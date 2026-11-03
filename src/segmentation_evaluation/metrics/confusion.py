
from __future__ import annotations

from typing import Any, Literal 

import numpy as np

from numpy.typing import ArrayLike, NDArray


Normalization = Literal["true", "pred", "all"]

def confusion_matrix(y_true: NDArray[Any], y_pred: NDArray[Any], *, num_classes: int | None = None,
                    labels: list[Any] | NDArray[Any] | None = None,
                    decision_labels: list[Any] | NDArray[Any] | None = None,
                    ignore_index: Any | None = None,
                    normalize: Normalization | None = None) -> NDArray:
    """
    Compute a confusion matrix for a classification or segmentation task.

    Rows correspond to true classes and columns to predicted classes.

    Parameters
    ----------
        y_true : array-like
                Ground-truth class labels. Can have any shape.
        y_pred : array-like
                Predicted class labels. Must have the same shape as y_true.
        num_classes: int, optional
                Number of classes. Assumes labels are intengers in [0, num_classes).
                Cannot be used together with `labels`.
        labels : array-like, optional
                Explicit class labels and their order in the confusion matrix.
                Useful when class labels are not consecutive integers.
        decision_labels : array-like, optional
                Explicit labels for the predictions decisions (columns).
                This can include an abstention label.
        ignore_index : int, optional
                Ground-truth label to ignore. Predictions at those position are also
                excluded.
        normalize : {None, "true", "pred", "all"}, optional
                Normalization method:
                    - None, raw counts.
                    - "true", normalize eaxh row (recall-oriented normalization).
                    - "pred", normalize each column (presicion-oriented normalization).
                    - "all", normalize by the total number of observations.

    Returns
    -------
        numpy.ndarray
            Confusion matrix of shape (K, J), where
                - K = number of true classes
                - J = number of prediction decisions
    Notes
    -----
        Matrix convention:
            cm[ii, jj] = number of observations with true class ii and 
                            predicted class jj.
        If labels and decision_labels are not provided, their values are 
        inferred independently from y_true and y_pred.
        To guarantee a fixed set of classes or decisions, explicetly provide 
        the corresponding labels.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape."
            f"Got {y_true.shape} and {y_pred.shape}."
        )

    if y_true.size == 0:
        raise ValueError("y_true and y_pred cannot be empty.")

    '''
    if not np.issubdtype(y_true.dtype, np.integer):
        raise TypeError("y_true must constain integer class labels.")

    if not np.issubdtype(y_pred.dtype, np.integer):
        raise TypeError("y_pred must contain integer class labels.")
    '''
    if labels is not None and num_classes is not None:
        raise ValueError("Specify either labels or num_classes, not both.")

    if labels is not None:
        class_labels = np.asarray(labels)

        if class_labels.ndim != 1 or class_labels.size == 0:
            raise ValueError("labels must be a non-empty one-dimension sequence.")

        if not np.issubdtype(class_labels.dtype, np.integer):
            raise TypeError("labels must contain integers.")

        if np.unique(class_labels).size != class_labels.size:
            raise ValueError("labels must constain unique values.")

    elif num_classes is not None:
        if (
            not isinstance(num_classes, (int, np.integer)) 
            or num_classes <= 0 
            or isinstance(num_classes, (bool, np.bool_))
        ):                                                          
            raise ValueError("num_classes must be a positive integer.")

        class_labels = np.arange(num_classes)
    else:
        class_labels = np.unique(y_true)

    if decision_labels is not None:
        decision_labels = np.asarray(decision_labels)
    else:
        decision_labels = np.unique(y_pred)


    if ignore_index is not None:
        valid = y_true != ignore_index
        y_true = y_true[valid]
        y_pred = y_pred[valid]

    if y_true.size == 0:
        raise ValueError("No observations remain after applying ignore_index.")

    if class_labels.ndim != 1 or class_labels.size == 0:
        raise ValueError("labels must be non-empty 1D array")

    if decision_labels.ndim != 1 or decision_labels.size == 0:
        raise ValueError("decision_labels must be a non-empty 1D array.")

    if np.unique(class_labels).size != class_labels.size:
        raise ValueError("True class labels must be unique")

    if np.unique(decision_labels).size != decision_labels.size:
        raise ValueError("Decisions labels must be unique")

    # Map class labels to matrix indices.
    label_to_index = {
        label: index for index, label in enumerate(class_labels.tolist())
    }

    decision_to_index = {
        label: index for index, label in enumerate(decision_labels.tolist())
    }

    try:
        true_indices = np.array(
            [label_to_index[label] for label in y_true.ravel()],
            dtype = np.int64)
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "y_true contains labels not included in labels."
        ) from exc
    try:
        decision_indices = np.array(
            [decision_to_index[label] for label in y_pred.ravel()],
            dtype = np.int64)
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "y_pred contains decision not included in decision_labels"
        ) from exc

    num_true_classes = len(class_labels)
    num_decisions = len(decision_labels)

    # Each pair (true class, predicted class) is encoded as one integer.
    encoded = true_indices*num_decisions + decision_indices

    cm = np.bincount(encoded, minlength = num_true_classes*num_decisions
                     ).reshape(num_true_classes, num_decisions)

    if normalize is None:
        return cm

    cm = cm.astype(np.float64)

    if normalize == "true":
        denominator = cm.sum(axis = 1, keepdims = True)
    elif normalize == "pred":
        denominator = cm.sum(axis = 0, keepdims = True)
    elif normalize == "all":
        denominator = cm.sum()
    else:
        raise ValueError(
            "normalize must be None, 'true', 'pred', or 'all'."
        )
    return np.divide(cm, denominator, out = np.zeros_like(cm),
                     where = denominator != 0)
