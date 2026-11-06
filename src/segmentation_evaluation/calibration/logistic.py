from __future__ import annotations

from typing import Any
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .base import BaseCalibrator

from ._utils import (
    encoded_labels,
    softmax,
    validate_logits,
    logsumexp,
)

class VectorScaling(BaseCalibrator):
    """
    Multiple vector scaling calibration. 
    Vector scaling applies a class-specific affine transformation to
    the logits before computing the softmax probabilities:

            z_cal[k] = a[k]*z[k] + b[k]
            p_cal = softmax(z_cal)

    The scale and bias parameters are estimated by minimizing the 
    multiclass negative log-likelihood on calibration data.

    An L2 penalty regularizes the parameters toward the identity 
    transformation (a = 1, b = 0):

            objective = NLL + (lambda/2)*(sum((a-1)**2) + sum(b**2))

    Parameters
    ----------
        regularization : float, default = 1e-4
            Nonnegative L2 regularization strength. Larger values keep
            the fitted transformation closer to the identity 
            transformation.

        learning_rate: float, default = 0.03
            Learning rate used by the Adam optimizer.

        tol : float, default = 1e-8
            Convergence tolerance base on the infinity norm of the 
            gradient.

        max_iter : int, defualt = 2000
            Maximum number of full-batch Adam iterations.

    Notes
    -----
        Vector scaling has 2K parameters, where K is the number of classes.

        Temperature scaling is a constrained form of vector scaling:
        it corresponds to a[k] = 1/T for every class and b[k] = 0.

        Because softmax is invariant to adding the same constant to every
        transformed logit, the bias parameters are not uniquely identifiable.
        Regularization toward zero provides a useful reference solution.
    """
    def __init__(self, *, regularization: float = 1e-4, learning_rate: float = 0.03,
                 tol: float = 1e-8, max_iter: int = 2000,
                 ) -> None:
        super().__init__() 
        if not np.isfinite(regularization) or regularization < 0:
            raise ValueError(
                "regularization must be finite and nonnegative."
            )
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError(
                "learning_rate must be finite and positive."
            )

        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be finite and positive.")

        if not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive interger.")

        self.regularization = float(regularization)
        self.learning_rate = float(learning_rate)
        self.tol = float(tol)
        self.max_iter = max_iter

        self.classes_: NDArray[Any] | None = None
        self.scales_: NDArray[np.float64] | None = None
        self.biases_: NDArray[np.float64] | None = None
        self.n_iter_: int | None = None
        self.nll_ : float | None = None
        self.objective_:float | None = None

    def _objective_and_gradient(self, logits: NDArray[np.float64], 
                                y_encoded: NDArray[np.int64], scales: NDArray[np.float64],
                                biases: NDArray[np.float64],
                                ) -> tuple[float, NDArray[np.float64], NDArray[np.float64], float]:
        """
        Compute the regularized objective and its gradients.

        Returns
        -------
            objective : float
                Mean NLL plus L2 regularization.

            gradient_scales : NDArray[np.float64]
                Gradient with respecto the class-specific scales.

            gradient_biases : NDArray[np.float64]
                Gradient with respect to the class-specific biases.

            nll : float
                Mean negative log-likelihood without regularization.
        """

        n_samples = logits.shape[0]
        calibrated_logits = logits*scales + biases
        probabilities = softmax(calibrated_logits)

        # Compute the mean negative log-likelihodd.
        true_probabilities = probabilities[np.arange(n_samples), y_encoded]
        nll = float(-np.mean(np.log(np.maximum(true_probabilities, 1e-15))))

        # Gradient of the mean NLL with respect to claibrated logits.
        residuals = probabilities.copy()
        residuals[np.arange(n_samples), y_encoded] -= 1-0
        residuals /= n_samples

        gradient_scales = np.sum(residuals*logits, axis = 0)
        gradient_biases = np.sum(residuals, axis = 0)

        # Regularization toward scales = 1 and biases = 0
        gradient_scales += self.regularization*(scales - 1.0)
        gradient_biases += self.regularization*biases

        penalty = 0.5*self.regularization*(np.sum((scales-1.0)**2) + np.sum(biases**2))
        objective = nll + float(penalty)

        return objective, gradient_scales, gradient_biases, nll 

    def fit(self, logits: ArrayLike, y_true: ArrayLike, *, labels: ArrayLike | None = None,
            ignore_index: Any | None = None,
            ) -> VectorScaling:
        """
        Fit vector scaling parameters using calibration data.

        Parameters
        ----------
            logits : ArrayLike
                Uncalibrated logits with shape (..., K).

            y_true : ArrayLike
                Ground-Truth labels with shape logits.shape[:-1]

            labels : ArrayLike, optional
                Ordered class labels. If omitted, integer labels from 0 to K-1
                are assumed qwhen possible; otherwise, classes are inferred 
                from the valid observations.

            ignore_index : Any, optional
                Ground-truth label to exclude from fitting.

        Returns
        -------
            VectorScaling
                The fitted calibrator. 
        """
        logits_array = validate_logits(logits)
        y_array = np.asarray(y_true)

        if y_array.shape != logits_array.shape[:-1]:
            raise ValueError(
                "y_true must have shape logits.shape[:-1]."
            )

        n_classes = logits_array.shape[-1]
        encoded, classes, valid_mask = encoded_labels(
            y_array, n_classes, labels = labels, ignore_index = ignore_index)

        # Flatten spatial and batch dimensions into observations.
        logits_flat = logits_array.reshape(-1, n_classes)
        logits_flat = logits_flat[valid_mask.reshape(-1)]

        n_samples = logits_flat.shape[0]

        # Initialize at the identity transformation.
        scales = np.ones(n_classes, dtype = np.float64)
        biases = np.zeros(n_classes, dtype = np.float64)

        # Adam optimizer state
        first_moment_scales = np.zeros_like(scales)
        second_moment_scales = np.zeros_like(scales)

        first_moment_biases = np.zeros_like(biases)
        second_moment_biases = np.zeros_like(biases)

        beta1 = 0.9
        beta2 = 0.999
        adam_epsilon = 1e-8

        # Retain the best parameters encountered during optimization.
        best_scales = scales.copy()
        best_biases = biases.copy()
        best_objective = np.inf
        best_nll = np.inf

        n_iter = 0

        for iter_ in range(1, self.max_iter + 1):
            ( objective, grad_scales, 
             grad_biases, nll ) = self._objective_and_gradient(logits_flat, encoded, 
                                                               scales, biases)

            if objective < best_objective:
                best_objective = objective
                best_nll = nll
                best_scales = scales.copy()
                best_biases = biases.copy()

            grad_norm = max(np.linalg.norm(grad_scales, ord = np.inf),
                            np.linalg.norm(grad_biases, ord = np.inf))

            n_iter = iter_ - 1

            if grad_norm < self.tol:
                break

            # Update first and second moments.
            first_moment_scales = beta1*first_moment_scales + (1.0 - beta1)*grad_scales
            second_moment_scales = beta2*second_moment_scales + (1-beta2)*grad_scales**2

            first_moment_biases = beta1*first_moment_biases + (1.0 - beta1)*grad_biases
            second_moment_biases = beta2*second_moment_biases + (1.0 - beta2)*grad_biases**2

            # Bias corrections
            corrected_moment_scales = first_moment_scales/(1.0 - beta1**iter_)
            corrected_variance_scales = second_moment_scales/(1.0 - beta2**iter_)
            corrected_moment_biases = first_moment_biases/(1.0 - beta1**iter_)
            corrected_variance_biases = first_moment_biases/(1.0 - beta2**iter_)

            # Adam parameters updates.
            scales -= self.learning_rate*corrected_moment_scales/(np.sqrt(corrected_variance_scales) + adam_epsilon)
            biases -= self.learning_rate*corrected_moment_biases/(np.sqrt(corrected_variance_biases) + adam_epsilon)

            n_iter = iter_

        # Evaluate the final parameters and retain them if they improve 
        # upon the best ojjective encountered
        objective, _, _, nll = self._objective_and_gradient(logits_flat, encoded, scales, biases)
        if objective < best_objective:
            best_objective = objective
            best_nll = nll
            best_scales = scales.copy()
            best_biases = biases.scale()

        self.classes_ = classes
        self.scales_ = best_scales
        self.biases_ = best_biases
        self.n_iter_ = n_iter
        self.nll_ = float(best_nll)
        self.objective_ = float(best_objective)

        self.is_fitted_ = True

        return self

    def predict_proba(self, logits:ArrayLike) -> NDArray[np.float64]:
        """
        Transform logits into a calibrated multiclass probabilities.

        Parameters
        ----------
            logits : ArrayLike
                Logits with shape (..., K).

        Returns
        -------
            NDArray[np.float64]
                Calibrated probabilities with the same shape as logits.
                Probabilities sum t one along the last axis.
        """
        self._check_is_fitted()
        logits_array = validate_logits(logits)

        if logits_array.shape[-1] != len(self.classes_):
            raise ValueError(
                "The number of classes in lgits differs from the " \
                "number of classes used during fitting."
            )
        calibrated_logits = logits_array*self.scales_ + self.biases_

        return softmax(calibrated_logits)

def vector_scaling(logits_val: ArrayLike, y_val: ArrayLike, *, logits_test: ArrayLike | None = None,
                   labels: ArrayLike | None = None, ignore_index: Any | None = None,
                   regularization: float = 1e-4, learning_rate: float = 0.03, tol: float = 1e-8,
                   max_iter: int = 2000, return_calibrator: bool = False,
                   ) -> NDArray[np.float64] | tuple[NDArray[np.float64], VectorScaling]:
    """
    Fit vector scaling on validation data and transform logits.

    The calibrator is fitted using logits_val and y_val. If logits_test
    is provided, the fitted calibrator is applied to logits_test.
    Otherwise, it is applied to logits_val.

    Parameters
    ----------
        logits_val : ArrayLike
            Validation logits with shape (..., K).

        y_val : ArrayLike
            Validation labels with shape logits_val.shape[:-1].

        logits_test : ArrayLike, optional
            Logits to transform after fitting. Typically, these are test
            logits. If omitted, validation logits are transformed.

        labels : ArrayLike, optional
            Ordered class labels.

        ignore_index : Any, optional
            Ground-truth label to exclude from fitting.

        regularization : float, default=1e-4
            L2 regularization strength.

        learning_rate : float, default=0.03
            Adam learning rate.

        tol : float, default=1e-8
            Gradient convergence tolerance.

        max_iter : int, default=2000
            Maximum number of optimization iterations.

        return_calibrator : bool, default=False
            If True, return both the calibrated probabilities and the
            fitted VectorScaling object.

    Returns
    -------
        probabilities : NDArray[np.float64]
            Calibrated probabilities.

        calibrator : VectorScaling, optional
            Fitted calibrator, returned when return_calibrator=True.
    """ 

    calibrator = VectorScaling(regularization = regularization, learning_rate = learning_rate,
                               tol = tol, max_iter = max_iter)

    calibrator.fit(logits_val, y_val, labels = labels, ignore_index = ignore_index)

    logits_to_transform = logits_val if logits_test is None else logits_test

    probabilities = calibrator.predict_proba(logits_to_transform)

    if return_calibrator:
        return probabilities, calibrator

    return probabilities
        

