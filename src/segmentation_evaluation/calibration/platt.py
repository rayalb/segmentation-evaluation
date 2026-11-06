"""
Platt-style probability calibration methods.

This modelu implements multiclass temperature scaling, a constrained 
form of logit-based calibration.

Temperature scaling transforms model logits using a single positive 
scalar parameter T:

    p_k = exp(z_k/T)/sum_j exp(z_j/T)

The temperature is estimated by minimizing the negative log-likelihood 
on a calibration dataset.
"""

from __future__ import annotations
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .base import BaseCalibrator

from ._utils import (
    encoded_labels,
    logsumexp,
    softmax,
    validate_logits)

class TemperatureScaling(BaseCalibrator):
    """
    Multiclass temperature scaling.

    Temperature scaling calibrates a model's logits by dividing them
    b a single positive scalar temperature before applying softmax.

    The temperature is fitted by minimizing the negative log-likelihood
    on a calibration dataset.

    Parameters
    ----------
        temperature_bounds : tuple of float, default = (0.05, 20.0)
            Lower and upper bounds for the temperature search interval.
            Both values must be finite and strictly positive, and the
            lower bound must be smaller than the upper bound.

        tol : float, default = 1e-6
            Tolerance used by the golden-section search algorithm.
            The optimization is performed in log.temperature space.

        max_iter : int, default = 200
            Maximum number of iterations for the golden-section search.

    Attributes
    ----------
        temperature_ : float or None
            Estimated temperature after fitting.

        classes_ : ndarray or None
            Class labels corresponding to the logit columns.

        n_iter : int or None
            Number of optimization iteration performed during fitting.

        nll_ : float or None
            Minimum validation negative log-likelihood found by the optimizer.
        
        is_fitted_ : bool
            Indicated whether the calibrator has been successfully fitted.

    Notes
    -----
        Temperature scaling preserves the predicted class:
            argmax_ z_k = argmax_k z_k/T, for every T>0.

        Therefore, temperature scaling changes the confidence of predictions
        but does not change their argmax classification.
    """

    def __init__(self, temperature_bounds: tuple[float, float] = (0.05, 20.0),
                 tol: float = 1e-6, max_iter: int = 200) -> None:
        super().__init__()
        lower, upper = temperature_bounds

        if not (
            np.isfinite(lower) 
            and np.isfinite(upper) 
            and 0.0<lower<upper
        ):
            raise ValueError(
                "temperature_bounds must contain two finite, positive "
                "values with lower < upper. "
            )
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError("tol must be finite and strictly positive.")
        if (
            isinstance(max_iter, bool) 
            or not isinstance(max_iter, (int, np.integer)) 
            or max_iter < 1
        ):
            raise ValueError("max_iter must be a positive integer.")

        self.temperature_bounds = (float(lower), float(upper))
        self.tol = float(tol)
        self.max_iter = int(max_iter)

        self.temperature_: float | None = None
        self.classes_ : NDArray[Any] | None = None
        self.n_iter_ : int | None = None
        self.nll_ : float | None = None

    @staticmethod
    def _validate_logits(logits: ArrayLike) -> NDArray[np.float64]:
        """
        Validate and convert logits to float64.

        Parameters
        ----------
            logits : array-like, shape (..., K).
                Model logits.

        Returns
        -------
            logits : ndarray of shape (..., K)
                Validated logits represented as float64.
        """
        logits_array = np.asarray(logits, dtype = np.float64)

        if logits_array.ndim < 1:
            raise ValueError("logits must have at least one dimension.")
        if logits_array.shape[-1] < 2:
            raise ValueError("logits must contain at least two classes.")
        if not np.all(np.isfinite(logits_array)):
            raise ValueError("logits must contain only finite values.")

        return logits_array

    @staticmethod
    def _softmax(logits: NDArray[np.float64], temperature: float) -> NDArray[np.float64]:
        """
        Compute numerically stable temperature-scaled softmax.
        
        Parameters
        ----------
            logits : ndarray of shape (..., K)
                Model logits.

            temperature : float
                Positive temperature parameter.

        Returns
        -------
            probabilities : ndarray of shape (..., K)
                Temperatur-scaled class probabilities.
        """
        scaled_logits = logits / temperature
        shifted_logits = scaled_logits - np.max(scaled_logits, axis = -1, keepdims = True)
        exp_logits = np.exp(shifted_logits)
        return exp_logits / np.sum(exp_logits, axis = -1, keepdims = True)

    @staticmethod
    def _encode_labels(y_true: ArrayLike, n_classes: int, labels: ArrayLike | None,
                       ignore_index: Any | None,
                       ) -> tuple[NDArray[np.int64], NDArray[Any]]:
        """
        Validate and encode ground-truth labels.

        Parameters
        ----------
            y_true : array-like
                Ground-truth labels.
            
            n_classes : int
                Numeber of classes represented by the logit column.

            labels : array-like or None
                Class labels corresponding to the logit columns.
                If None, integer labels in [o, n_classes) are assumed.
                Otherwise, all n_classes labels must be represented in
                y_true so that the class ordering can be inferred.

            ignore_index : scalar or None
                Ground-truth label to exclude from calibration.

        Returns
        -------
            encoded_labels : ndarray of shape (N_valid)
                Integer-encoded labels in [0, n_classes).
            
            classes : ndarray of shape (n_classes,)
                Class labels in the same order as the logit columns.
        """
        y_array = np.asarray(y_true)

        if ignore_index is None:
            valid_mask = np.ones(y_array.shape, dtype = bool)
        else:
            valid_mask = y_array != ignore_index

        y_valid = y_array[valid_mask].reshape(-1)

        if y_valid.size == 0:
            raise ValueError("No valid observations remain after applyinf ignore_index")

        if labels is None:
            # Default convention: integer class indices 0, ... K-1.
            if (
                np.issubdtype(y_valid.dtype, np.integer) 
                and np.all(y_valid >= 0) 
                and np.all(y_valid < n_classes)
            ):
                classes = np.arange(n_classes)
            else:
                inferred_classes = np.unique(y_valid)
                if inferred_classes.size != n_classes:
                    raise ValueError(
                        "Could not infer the class ordering from y_true. "
                        "Provide labels explicitly, with one label for "
                        "each logit column." 
                    )
                classes = inferred_classes
        else:
            classes = np.asarray(labels)
            if classes.ndim != 1 or classes.size != n_classes:
                raise ValueError(
                    "labels must be a one-dimensional array containing "
                    "one label for each logit column. "
                )

            for ii in range(n_classes):
                if np.any(classes[ii] == classes[:ii]):
                    raise ValueError("labels must contain unique values.")

        # Map original labels to their corresponding logit-column indices.
        matches = y_valid[:, np.newaxis] == classes[np.newaxis, :]

        if not np.all(np.sum(matches, axis = 1) == 1):
            raise ValueError(
                "y_true contains labels that cannot be matched unequely "
                "to the provided classes."
            )
        encoded_labels = np.argmax(matches, axis = 1).astype(np.int64)
        return encoded_labels, classes

    @staticmethod
    def _negative_log_likelihood(logits: NDArray[np.float64], y_encoded: NDArray[np.int64],
                                 temperature: float,
                                 ) -> float:
        """
        Compute the mean negative log-likelihood.

        The calculation uses log-sum-exp estabilization.

        Parameters
        ----------
            logits : ndarray of shape (N, K)
                Flattened model logits.
            
            y_encoded : ndarray of shape (N, )
                Integer-encoded ground-truth labels.
            
            temperature : float
                Positive temperature parameter.

        Returns
        -------
            nll: float
                Mean negative log-likelihood.
        """

        scaled_logits = logits/temperature

        max_logits = np.max(scaled_logits, axis = 1, keepdims = True)
        shifted_logits = scaled_logits - max_logits
        log_sum_exp = np.log(np.sum(np.exp(shifted_logits), axis = 1)) + max_logits[:, 0]

        true_logits = scaled_logits[np.arange(y_encoded.size), y_encoded]

        return float(np.mean(log_sum_exp - true_logits))

    def _golden_section_search(self, objective : Any, lower: float, upper: float,
                               ) -> tuple[float, int]:
        """
        Minimize a scalar objective using golden-section search.

        Parameters
        ----------
            objective : callable
                Scalar objetive function.

            lower : float
                Lower bound of the search interval.

            upper : float
                Upper bound of the search interval.

        Returns
        -------
            optimum : float
                Estimated minimizer in the search interval.

            n_iter : int
                Number of iterations performed.

        Notes
        -----
            The objective is assumed to be unimodal over the search interval.
        """
        golden_ratio = (np.sqrt(5.0) - 1.0)/2.0

        x1 = upper - golden_ratio*(upper - lower)
        x2 = lower + golden_ratio*(upper - lower)

        f1 = objective(x1)
        f2 = objective(x2)
        n_iter = 0

        while (upper - lower) > self.tol and n_iter < self.max_iter:
            if f1 <= f2:
                upper = x2
                x2 = x1
                f2 = f1

                x1 = upper - golden_ratio*(upper-lower)
                f1 = objective(x1)
            else:
                lower = x1
                x1 = x2
                f1 = f2
                x2 = lower + golden_ratio*(upper-lower)
                f2 = objective(x2)

            n_iter += 1

        candidates = np.array([lower, upper, x1, x2])
        candidate_values = np.array([objective(value) for value in candidates])

        optimum = float(candidates[np.argmin(candidate_values)])
        return optimum, n_iter 

    def fit(self, logits: ArrayLike, y_true: ArrayLike, *,
            labels: ArrayLike | None = None,
            ignore_index: Any | None = None,
            ) -> TemperatureScaling:
        """
        Estimate the optimal temperature using calibration dataset.

        Parameters
        ----------
            logits : array-like, shape (..., K)
                Uncalibrated model logits. The last dimension represents
                the class dimension.
            
            y_true : array-like of shape (...)
                Ground-truth labels corresponding to the observations.
            
            labels : array-like of shape (K,), optional
                Class labels corresponding to the logit columns.
                If omitted, integer labels 0, ..., K-1 are assumed when
                possible. Otherwise, the class ordering is inferred from
                y_true if all classes are represented.

            ignore_index : scalar, optional
                Ground-truth label to ignore during fitting.

        Returns
        -------
            self : TemperatureScaling
                The fitted calibrator.

        Notes
        -----
            The optimization minimizes:

                    T* = argmin_T NLL(softmax(logits / T), y_true)

                    subject to: temperature_bounds[0] <= T <= temperature_bounds[1].

            Fitting should be performed using validation data, not test data.
        """
        
        self.temperature_ = None
        self.classes_ = None
        self.n_iter_ = None
        self.nll_ = None

        logits_array = self._validate_logits(logits)
        y_array = np.array(y_true)

        if y_array.shape != logits_array.shape[:-1]:
            raise ValueError(
                "y_true must have the same shape as logits excluding "
                "the last (class) dimension. "
                f"Received y_true shape {y_array.shape} and logits "
                f"shape {logits_array.shape}."
            )
        n_classes = logits_array.shape[-1]
        y_encoded, classes = self._encode_labels(y_true = y_true, n_classes = n_classes,
                                                 labels = labels, ignore_index = ignore_index)

        # Remove ignored onservation from the logits as well.
        if ignore_index is None:
            valid_logits = logits_array.reshape(-1, n_classes)
        else:
            valid_mask = (y_array != ignore_index).reshape(-1)
            valid_logits = logits_array.reshape(-1, n_classes)[valid_mask]

        lower_temp, upper_temp = self.temperature_bounds

        # Optimize in log-temperature space.
        lower = float(np.log(lower_temp))
        upper = float(np.log(upper_temp))

        def objective(log_temperature: float) -> float:
            temperature = float(np.exp(log_temperature))
            return self._negative_log_likelihood(
                valid_logits, y_encoded, temperature)

        optimal_log_temperature, n_iter = self._golden_section_search(
            objective = objective, lower = lower, upper = upper
        )

        optimal_temperature = float(np.exp(optimal_log_temperature))

        self.temperature_ = optimal_temperature
        self.classes_ = classes
        self.n_iter_ = n_iter
        self.nll_ = objective(optimal_log_temperature)
        self.is_fitted_ = True

        return self

    def predict_proba(self, logits: ArrayLike) -> NDArray[np.float64]:
        """
        Apply the fitted temperature to model logits.

        Paramenters
        -----------
            logits : array-like, shape (..., K)
                Uncalibrated model logits.
        
        Returns
        -------
            probabilities : ndarray, shape (... K)
                Calibrated class probabilities.     
        """
        self._check_is_fitted()

        logits_array = self._validate_logits(logits)

        if logits_array.shape[-1] != self.classes_.size:
            raise ValueError(
                "The number of logit columns does not match the number "
                "of classes used during fitting."
            )

        return self._softmax(logits=logits_array,
            temperature=self.temperature_
        )

def temperature_scaling(logits_val: ArrayLike, y_val: ArrayLike, *, 
                        logits_test: ArrayLike | None = None,
                        labels: ArrayLike | None = None,
                        ignore_index: Any | None = None,
                        temperature_bounds: tuple[float, float] = (0.05, 20.0),
                        tol: float = 1e-6, max_iter: int = 200,
                        return_calibrator: bool = False,
                        ) -> NDArray[np.float64] | tuple[NDArray[np.float64], TemperatureScaling]:
    """
    Fit and apply multiclass temperature scaling.

    Parameters
    ----------
        logits_val : array-like of shape (..., K)
            Validation logits used to estimate the temperature.

        y_val : array-like of shape (...)
            Validation ground-truth labels.

        logits_test : array-like of shape (..., K), optional
            Logits to calibrate after fitting.
            If None, the fitted calibrator is applied to logits_val.

        labels : array-like of shape (K,), optional
            Class labels corresponding to the logit columns.

        ignore_index : scalar, optional
            Ground-truth label to ignore during fitting.

        temperature_bounds : tuple of float, default=(0.05, 20.0)
            Lower and upper bounds for the temperature search interval.

        tol : float, default=1e-6
            Tolerance used by the golden-section search.

        max_iter : int, default=200
            Maximum number of optimization iterations.

        return_calibrator : bool, default=False
            If True, return both calibrated probabilities and the fitted
            TemperatureScaling object.

    Returns
    -------
        probabilities : ndarray
            Calibrated probabilities for logits_test, or logits_val if
            logits_test is None.

        calibrator : TemperatureScaling, optional
            Returned only when return_calibrator=True.
    """
    calibrator = TemperatureScaling(
        temperature_bounds=temperature_bounds,
        tol=tol,
        max_iter=max_iter,
    )

    calibrator.fit(
        logits=logits_val,
        y_true=y_val,
        labels=labels,
        ignore_index=ignore_index,
    )

    logits_to_transform = (
        logits_val if logits_test is None else logits_test
    )

    probabilities = calibrator.predict_proba(logits_to_transform)

    if return_calibrator:
        return probabilities, calibrator

    return probabilities


def _stable_sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
        Compute the logistic sigmoid without overflow.
    """
    return np.exp(-np.logaddexp(0.0, -x))

def _binary_platt_fit(scores: NDArray[np.float64], targets: NDArray[np.float64], *,
                      tol: float, regularization: float, max_iter: int,
                      ) -> tuple[NDArray[np.float64], int, float]:
    """
    Fit a binary logistic calibrator using Newton's method.

    The model is
    
        P(Y=1 | z) = sigmoid(a*z+b).
    
    A small L2 penalty stabilizes the optimization, particulary when the binary data 
    are nearly or completely separable

    Parameters:
    ----------
        scores : array-like

        targets : array-like

        tol : float, default = 1e-8
            Convergence tolerance for Newton's method.

        regularization : float, default = 1e-6
            L2 regularization applied to the slope and intercept parameters
            of each binary logistic model.

        max_iter : max_iter : int, default = 200
            Maximum number of Newton iterations for each class.

    Returns
    -------
        parameters : array-like
            Array[a, b]
        
        n_iter : int
            Number of Newton iterations.
        
        nll : float
            Mean binary negative log-likelihood without the regularization term.
    """
    X = np.column_stack((scores, np.ones_like(scores)))
    theta = np.array([1.0, 0.0], dtype = np.float64)

    Id = np.eye(2, dtype = np.float64)

    def objective(params: NDArray[np.float64]) -> float:
        linear = X @ params
        loss = np.mean(np.logaddexp(0.0, linear) - targets*linear)
        penalty = 0.5*regularization*np.sum(params**2)
        return float(loss + penalty)

    n_iter = 0
    for ii in range(max_iter):
        linear = X @ theta
        probabilities = _stable_sigmoid(linear)

        gradient = X.T @ (probabilities - targets)/len(targets) + regularization*theta
        weights = probabilities*(1.0 - probabilities)
        hessian = X.T @ (weights[:, None]*X)/len(targets) + regularization*Id

        try:
            direction = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            direction = np.linalg.pinv(hessian) @ gradient

        if np.linalg.norm(gradient, ord = np.inf) < tol:
            break

        # Backtracking line search.
        current_objective = objective(theta)
        step_size = 1.0

        while step_size > 1e-10:
            candidate = theta - step_size*direction

            if objective(candidate) < current_objective:
                break

            step_size *= 0.5

        if step_size <= 1e-10:
            break

        theta = candidate
        n_iter = ii + 1
        if np.linalg.norm(step_size*direction) < tol:
            break

    final_linear = X @ theta
    final_nll = np.mean(np.logaddexp(0.0, final_linear) - targets*final_linear)
    return theta, n_iter, float(final_nll)

class OVRPlattScaling(BaseCalibrator):
    """
    Multiclass One-vs-Rest Platt scaling.

    A separate binary logistic calibrator is fitted for each class.
    The resulting binary probabilities are normalized to obtain a
    multiclass probability distribution.

    For class k, the input score is the class-versus-rest logit:

        s_k = z_k - logsumexp(z_j, j != k)

    The binary calibration model is:

        q_k = sigmoid(a_k*s_k + b_k).

    The final multiclass probabilities are:

        p_k = q_k/sum_j q_j.

    Parametes
    ---------
        regularization : float, default = 1e-6
            L2 regularization applied to the slope and intercept parameters
            of each binary logistic model.

        tol : float, default = 1e-8
            Convergence tolerance for Newton's method.

        max_iter : int, default = 200
            Maximum number of Newton iterations for each class.
        
    Notes
    -----
        This method is more flexible than temperature scaling because each
        class has its own  slope and intercept.
    """
    def __init__(self, *, regularization: float = 1e-6, tol: float = 1e-8,
                 max_iter: int = 200) -> None:
        super().__init__()

        if not np.isfinite(regularization) or regularization < 0:
            raise ValueError("regularization must be finite and nonnegative.")

        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be finite and positive.")

        if not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")

        self.regularization = float(regularization)
        self.tol = float(tol)
        self.max_iter = max_iter

        self.classes_: NDArray[Any] | None = None
        self.parameters_ : NDArray[np.float64] | None = None
        self.slopes_: NDArray[np.float64] | None = None
        self.intercepts_ : NDArray[np.float64] | None = None
        self.n_iter_: NDArray[np.int64] | None = None
        self.nll_: float | None = None

    @staticmethod
    def _class_vs_rest_scores(logits: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute class-versus-rest logits.
        Return an array with the same shape as logits
        """
        n_classes = logits.shape[-1]
        scores = np.empty_like(logits)
        for kk in range(n_classes):
            other_logits = np.concatenate((logits[..., :kk], logits[..., kk+1:]),
                                          axis = -1)
            max_other = np.max(other_logits, axis = -1, keepdims = True)
            logsumexp_other = max_other[..., 0]+np.log(np.sum(np.exp(other_logits - max_other),
                                                              axis = -1))
            scores[..., kk] = logits[..., kk] - logsumexp_other
        return scores

    def fit(self, logits:ArrayLike, y_true: ArrayLike, *,
            labels: ArrayLike | None = None, ignore_index: Any | None = None,
            ) -> OVRPlattScaling:
        """
        Fit one binary Platt calibrator per class.

        Parameters
        ----------
            logits : ArrayLike
                Uncalibrated logits with shape (..., K).

            y_true : ArrayLike
                Ground-truth labels with shape logits.shape[:-1].

            labels : ArrayLike, optional
                Complete ordered list of class labels. If omitted, integer
                labels 0, ..., K-1 are assumed when applicable; otherwise,
                classes are inferred from valid observations.

            ignore_index : Any, optional
                Ground-truth label to exclude from fitting.

        Returns
        -------
            OVERPlattScaling
                The fitted calibrator.
        """
        logits_arrays = validate_logits(logits)
        if np.asarray(y_true).shape != logits_arrays.shape[:-1]:
            raise ValueError("y_true must have shape logits.shape[:-1]")

        n_classes = logits_arrays.shape[-1]
        encoded, classes, valid_mask = encoded_labels(
            y_true, n_classes, labels = labels, ignore_index = ignore_index)

        scores = self._class_vs_rest_scores(logits_arrays)
        scores = scores[valid_mask].reshape(-1, n_classes)

        parameters = np.empty((n_classes, 2), dtype = np.float64)
        iterations = np.empty(n_classes, dtype = np.int64)
        class_nlls = np.empty(n_classes, dtype = np.float64)

        for kk in range(n_classes):
            targets = (encoded == kk).astype(np.float64)
            if np.unique(targets).size < 2:
                raise ValueError(
                    f"Class {classes[kk]!r} must have both positive "
                    "and negative observations in the fitting data."
                )
            parameters[kk], iterations[kk], class_nlls[kk] = _binary_platt_fit(
                scores[:, kk], targets, regularization = self.regularization,
                tol = self.tol, max_iter = self.max_iter
            )

        self.classes_ = classes
        self.parameters_ = parameters
        self.slopes_ = parameters[:, 0].copy()
        self.intercepts_ = parameters[:, 1].copy()
        self.n_iter_ = iterations
        self.nll_ = float(np.mean(class_nlls))
        self.is_fitted_ = True

        return self

    def predict_proba(self, logits: ArrayLike) -> NDArray[np.float64]:
        """
        Transform logits into calibrated multiclass probabilities.

        Parameters
        ----------
            logits : ArrayLike
                Logits with shape (..., K).
        
        Returns
        -------
            NDArray[np.float64]
                Calibrated probabilities with the same shape as logits.
                Probabilities sum to one along the last axis.
        """
        self._check_is_fitted()
        logits_array = validate_logits(logits)

        if logits_array.shape[-1] != len(self.classes_):
            raise ValueError(
                "The number of classes in logits differs from the " \
                "number of classes used during fitting."
            )
        scores = self._class_vs_rest_scores(logits_array)
        linear = scores*self.slopes_ + self.intercepts_

        log_q = -np.logaddexp(0.0, -linear)

        #max_log_q = np.max(log_q,  axis = -1, keepdims = True)
        log_normalizer = logsumexp(log_q, axis = -1, keepdims = True)#max_log_q + np.log(np.sum(np.exp(log_q - max_log_q),
                                         #          axis = -1, keepdims = True))
        probabilities = np.exp(log_q - log_normalizer)
        return probabilities

def ovr_platt_scaling(logits_val: ArrayLike, y_val: ArrayLike, *,
                      logits_test : ArrayLike | None = None, labels: ArrayLike | None = None,
                      ignore_index: Any | None = None, regularization: float = 1e-6,
                      tol: float = 1e-8, max_iter: int = 200, 
                      return_calibrator: bool = False,
                      ) -> NDArray[np.float64] | tuple[NDArray[np.float64], OVRPlattScaling]:
    """
    Fit OVR Platt scaling on validation data and transform logits.

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

        calibrator : OVRPlattScaling, optional
            Returned when return_calibrator = True.
    """
    calibrator = OVRPlattScaling(regularization = regularization, tol = tol,
                                 max_iter = max_iter)

    calibrator.fit(logits_val, y_val, labels = labels, ignore_index = ignore_index)

    logits_to_transform = logits_val if logits_test is None else logits_test
    probabilities = calibrator.predict_proba(logits_to_transform)
    if return_calibrator:
        return probabilities, calibrator

    return probabilities
        

            


