from __future__ import annotations

from typing import Any
import numpy as np

from numpy.typing import ArrayLike, NDArray

def validate_logits(logits: ArrayLike) -> NDArray[np.float64]:
        
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

def encoded_labels(y_true: ArrayLike, n_classes: int, *, labels: ArrayLike | None,
                    ignore_index: Any | None,
                    ) -> tuple[NDArray[np.int64], NDArray[Any], NDArray[np.bool_]]:
    """
    Encode class labels as integer indices.

    The encoding follows the order specified by ``classes`` rather than
    assuming that labels are consecutive integers.

    Parameters
    ----------
        y_true : ArrayLike
            Ground-truth labels.

        n_classes : int
            Number of classes in the logits.

        labels : ArrayLike, optional
            Ordered class labels. If omitted, integer labels from 0 to 
            n_classes - 1 are asummed when possible. Otherwise, the classes
            are inferred from the valid observations.

        ignore_index : Any, optional
            Label to exclude from calibration.

    Return
    ------
        encoded : NDArray[np.int64]
            Encoded valid labels, with values in [0, n_classes - 1].

        classes : NDArray[Any]
            Ordered class labels.

        valid_mask : NDArray[np.bool_]
            Boolean mask identifying valid observations in y_true.
    """

    y_array = np.asarray(y_true)

    if ignore_index is None:
        valid_mask = np.ones(y_array.shape, dtype = bool)
    else:
        valid_mask = y_array != ignore_index

    y_valid = y_array[valid_mask]
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
        if len(np.unique(classes)) != n_classes:
            raise ValueError("labels must contain unique values.")
    
    encoded_labels = np.empty(y_valid.shape, dtype = np.int64)
    for ii, label in enumerate(classes):
        encoded_labels[y_valid == label] = ii
    
    known = np.zeros(y_valid.shape, dtype = bool)
    
    for label in classes:
        known |= y_valid == label
    
    if not np.all(known):
        raise ValueError("y_true contains labels not included in labels.")
                
    return encoded_labels, classes, valid_mask 

def softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute numerically stable softmax probabilities.

    Parameters
    ----------
        logits: NDArray[np.float64]

    Returns
    -------
        NDArray[np.float64]
            Probability array with the same shape as logits.
            Probabilities sum to one along the last axis.
    """
    shifted = logits - np.max(logits, axis = -1, keepdims = True)
    exp_logits = np.exp(shifted)

    return exp_logits/np.sum(exp_logits, axis = -1, keepdims = True)


def logsumexp(values: ArrayLike, *, axis: int | None = None, keepdims: bool = False,
             ) -> NDArray[np.float64]:
    """
    Compute the numerically stable log-sum-exp function.

    The log-sum-exp is defined as

        logsumexp(x) = log(sum_i exp(x_i)).

    It is commonly used when computing log probabilities and
    cross-entropy losses.

    Parameters
    ----------
        values : array-like
            Input array.
        axis : int or None, default=None
            Axis or axes over which to compute the sum. If None, the
            operation is performed over the entire array.
        keepdims : bool, default=False
            If True, retain reduced dimensions with size one.

    Returns
    -------
        NDArray[np.float64]
            Log-sum-exp result.
    """
    values = np.asarray(values, dtype=np.float64)

    if not np.all(np.isfinite(values)):
        raise ValueError("values must contain only finite values.")

    maximum = np.max(values, axis=axis, keepdims=True)

    result = maximum + np.log(
        np.sum(np.exp(values - maximum), axis=axis, keepdims=True)
    )

    if not keepdims and axis is not None:
        result = np.squeeze(result, axis=axis)
    elif not keepdims and axis is None:
        result = np.squeeze(result)

    return result
