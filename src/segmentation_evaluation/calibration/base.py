"""
Base classes for probability calibration.

This module defines the abstract interfaces shared by all calibration
methods implemented

A calibration follows the general workflow:
    1. Fit calibration parameters using a validation dataset-
    2. Apply the fitted transformation to a new model logits
    3. Return calibrated class probabilities.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

CalibratorT = TypeVar("CalibratorT", bound = "BaseCalibrator")

class BaseCalibrator(ABC):
    """
    Abstract base class for probability calibrators.

    All calibration methods should inherit from this class
    and implement the fit() and predict_proba() methods.

    Parameters
    ----------
        None

    Attributes
    ----------
        is_fitted_ : bool
            Indicates whether the calibrator has been successfully fitted.

    Notes
    -----
        The base class does not impose a particular shape on the input logits.
        Concrete implementations must document their expected input shapes.

        The base class does not assume that the input logits have been normalized
        or converted into probabilities.
    """
    def __init__(self) -> None:
        """
        Initialize an unfitted calibrator.
        """
        self.is_fitted_: bool = False

    @abstractmethod
    def fit(self, logits: ArrayLike, y_true: ArrayLike, **fit_params: Any,
            ) -> CalibratorT:
        """
        Fit the calibrator using a validation dataset.

        Parameters
        ----------
            logits : array-like
                Uncalibrated model logits.

            y_true : array-like
                Ground-truth class labels corresponding to the logits.

            **fit_params : Any
                Additional parameters specific to the calibration method.

        Returns
        -------
            CalibratorT
                The fitted calibrator.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, logits: ArrayLike) -> NDArray[np.float64]:
        """
        Transform logits into calibrated class probabilities.

        Parameters
        ----------
            logits : array-like
                Uncalibrated model logits.

        Returns
        -------
            probabilities : ndarray of shape (..., K)
                Calibrated class probabilities.
        """
        raise NotImplementedError

    def fit_transform(self, logits: ArrayLike, y_true: ArrayLike, **fit_params: Any,
                      ) -> NDArray[np.float64]:
        """
        Fit the calibrator and transform the same dataset.

        Parameters
        ----------
            logits : array-like
                Uncalibrated model logits.

            y_true : array-like
                Ground-truth class labels corresponding to the logits.

            **fit_params : Any
                Additonal parameters passed to fit()

        Returns
        -------
            probabilities : ndarray of shape (..., K)
                Calibrated class probabilities.
        
        Notes
        -----
            For inbiased evaluation, calibration parameters should be fitted using
            validation data and then applied to an independent test set through 
            predict_proba().
        """

        self.fit(logits, y_true, **fit_params)
        return self.predict_proba(logits)

    def _check_is_fitted(self) -> None:
        """
        Check whether the calibrator has been fitted.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted. "
                "Call fit() before predict_proba()." 
            )
        