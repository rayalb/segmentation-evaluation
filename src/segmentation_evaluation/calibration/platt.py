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
        self.is_fitted_ = False
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