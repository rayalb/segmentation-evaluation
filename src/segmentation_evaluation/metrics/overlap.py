"""
Overlap metrics for segmentation: 
    - Dice. 
    - Intersection over Union (IoU).

The module supports two aggregation strategies:

    - "global":
        All pixels from all images are considered together. The overlap
        counts are accumulated before computing the metric.

    - "per_image":
        The metric is computed independently for each image, and the
        resulting scores are averaged across images.

Metrics can be computed for individual classes or averaged across classes.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_inputs(y_true: ArrayLike, y_pred: ArrayLike, labels: ArrayLike | None = None, 
                     ignore_index: Any | None = None,
                     aggregation: Literal["global", "per_image"] = "global",
                     ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """
    Validation segmentation arrays and determine the evaluated classes.
    
    Parameters
    ----------
        y_true : array-like
        Ground-truth segmentation labels.

        y_pred : array-like
            Predicted segmentation labels. Must have the same shape as
            ``y_true``.

        labels : array-like, optional
            Class labels to evaluate. If None, the classes are inferred from
            the ground-truth and predicted labels.

        ignore_index : int or str, optional
            Label to ignore in the ground-truth segmentation. Pixels for which
            ``y_true == ignore_index`` are excluded from the evaluation.

        aggregation : {"global", "per_image"}, default="global"
            Aggregation strategy. For ``"per_image"``, the first array
            dimension is interpreted as the image dimension.

    Returns
    -------
        y_true : numpy.ndarray
            Validated ground-truth array.

        y_pred : numpy.ndarray
            Validated prediction array.

        labels : numpy.ndarray
            Class labels used to compute the metrics.
   """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    if y_true.size == 0:
        raise ValueError("y_true and y_pred cannot be empty.")

    if aggregation not in ("global", "per_image"):
        raise ValueError("aggregation must be 'global' or 'per_image'.")

    if aggregation == "per_image" and y_true.ndim < 2:
        raise ValueError(
            "For per_image aggregation, the first axis must represent images."
        )

    if labels is None:
        if ignore_index is not None:
            valid = y_true != ignore_index
            y_true_valid = y_true[valid]
            y_pred_valid = y_pred[valid]
        else:
            y_true_valid = y_true.ravel()
            y_pred_valid = y_pred.ravel()

        labels = np.unique(
            np.concatenate([y_true_valid.ravel(), y_pred_valid.ravel()])
        )
    else:
        labels = np.asarray(labels)

    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional sequence.")

    if np.unique(labels).size != labels.size:
        raise ValueError("labels must contain unique values.")

    return y_true, y_pred, labels

def _compute_per_class(y_true: ArrayLike, y_pred: ArrayLike, labels: ArrayLike, 
                       ignore_index: Any | None = None,
                       ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute Dice and IoU independently for each specified class.

    Paramters
    ---------
        y_true : array-like
            Ground-truth segmentation labels.

        y_pred : array-like
            Predicted segmentation labels.

        labels : array-like
            Classes for which the metrics are computed.

        ignore_index : int or str, optional
            Ground-truth label to exclude from the calculations.

    Returns
    -------
        dice : numpy.ndarray
            Dice coefficient for each class, in the same order as ``labels``.

        iou : numpy.ndarray
            IoU for each class, in the same order as ``labels``.

    Notes
    -----
        If a class is absent from both ground truth and prediction, its
        denominator is zero and the corresponding score is returned as NaN.

        The treatment of these NaN values is handled by the public metric
        functions through the ``zero_division`` parameter.
    """

    dice = np.empty(len(labels), dtype = float)
    iou = np.empty(len(labels), dtype = float)

    if ignore_index is not None:
        valid = y_true != ignore_index
        y_true = y_true[valid]
        y_pred = y_pred[valid]
    else:
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()

    for kk, label in enumerate(labels):
        true_mask = y_true == label
        pred_mask = y_pred == label

        tp = np.count_nonzero(true_mask & pred_mask)
        fp = np.count_nonzero(~true_mask & pred_mask)
        fn = np.count_nonzero(true_mask & ~pred_mask)

        dice_denominator = 2*tp + fp + fn
        iou_denominator = tp + fp + fn

        dice[kk] = 2*tp/dice_denominator if dice_denominator > 0 else np.nan
        iou[kk] = tp/iou_denominator if iou_denominator > 0 else np.nan

    return dice, iou

def _aggregate_scores(scores: NDArray[np.float64], average: Literal["macro"] | None, 
                      zero_division: float) -> float | NDArray[np.float64]:
    """
    Aggregate class-wise scores according to the requensted convention.
    
    Parameters
    ----------
        scores : numpy.ndarray
            Metric values for individual classes.

        average : {"macro", None}
            If ``"macro"``, return the arithmetic mean across classes.
            If None, return the individual class scores.

        zero_division : float
            Value assigned to classes with zero denominator.

    Returns
    -------
        float or numpy.ndarray
            Aggregated metric or individual class-wise scores.
    """

    scores = np.where(np.isnan(scores), zero_division, scores)

    if average is None:
        return scores

    if average == "macro":
        return float(np.mean(scores))

    raise ValueError("average must be 'macro' or None.")

def dice_score(y_true: ArrayLike, y_pred: ArrayLike, *,
               labels: ArrayLike | None = None, average: Literal["macro"] | None = "macro",
               aggregation: Literal["global", "per_image"] = "global",
               ignore_index: Any | None = None, zero_division: float = 0.0,
               ) -> float | NDArray[np.float64]:
    """
    Compute the Dice coefficient for segmentation predictions.

    The Dice coefficient measures the overlap between the predicted and
    ground-truth regions of a class:

        Dice_k = 2 * TP_k / (2 * TP_k + FP_k + FN_k)

    Its values range from 0 to 1, where:

        - 1 indicates perfect overlap.
        - 0 indicates no overlap.

    Parameters
    ----------
        y_true : array-like
            Ground-truth segmentation labels.

        y_pred : array-like
            Predicted segmentation labels. Must have the same shape as
            ``y_true``.

        For ``aggregation="per_image"``, the first dimension represents
        independent images. For example, an array with shape
        ``(N, H, W)`` represents N segmentation maps.

        labels : array-like, optional
            Classes to evaluate. If None, classes are inferred from the
            ground-truth and predicted arrays.

            This parameter can be used to exclude the background class.
            For example, ``labels=[1, 2]`` evaluates only foreground classes
            1 and 2.

        average : {"macro", None}, default="macro"
            Class aggregation strategy:

            - ``"macro"``: return the unweighted arithmetic mean of the
                class-wise Dice coefficients.
            - ``None``: return one Dice coefficient per class, preserving
                the order specified by ``labels``.

        aggregation : {"global", "per_image"}, default="global"
            Image aggregation strategy:

            - ``"global"``: accumulate TP, FP, and FN across all images before
                    computing Dice. This is also known as pooled or micro-style
                    aggregation at the class level.
            - ``"per_image"``: compute Dice independently for each image and
                average the resulting scores across images.

            Note that global and per-image aggregation generally produce
            different results because they assign different weights to images.

        ignore_index : int or str, optional
            Ground-truth label to ignore. Pixels for which
            ``y_true == ignore_index`` are excluded from the calculation.

        zero_division : float, default=0.0
            Score assigned to a class when it is absent from both ground truth
            and prediction.

        Returns
        -------
            float or numpy.ndarray
                If ``average="macro"``, returns a scalar Dice score.

                If ``average=None``, returns a one-dimensional array containing
                the Dice score for each evaluated class.
    """

    if not (np.isscalar(zero_division) and (
        np.isfinite(zero_division) or np.isnan(zero_division))):
        raise ValueError("zero_division must be a finite number or np.nan.")

    y_true, y_pred, labels = _validate_inputs(y_true, y_pred, labels, ignore_index,
                                              aggregation)

    if aggregation == "global":
        dice, _ = _compute_per_class(y_true, y_pred, labels, ignore_index)
    else:
        image_scores = []
        for ii in range(y_true.shape[0]):
            dice, _ = _compute_per_class(y_true[ii], y_pred[ii], labels,
                                         ignore_index)
            image_scores.append(dice)

        dice = np.nanmean(np.asarray(image_scores), axis = 0)

    return _aggregate_scores(dice, average, zero_division)

def iou_score(y_true: ArrayLike, y_pred: ArrayLike, *,
               labels: ArrayLike | None = None, average: Literal["macro"] | None = "macro",
               aggregation: Literal["global", "per_image"] = "global",
               ignore_index: Any | None = None, zero_division: float = 0.0,
               ) -> float | NDArray[np.float64]:
    """
    Compute Intersection over Union (IoU) for segmentation predictions.

    IoU, also known as the Jaccard index, measures the intersection
    relative to the union of the predicted and ground-truth regions:

        IoU_k = TP_k / (TP_k + FP_k + FN_k)

    Its values range from 0 to 1, where:

        - 1 indicates perfect overlap.
        - 0 indicates no overlap.

    Parameters
    ----------
        y_true : array-like
            Ground-truth segmentation labels.

        y_pred : array-like
            Predicted segmentation labels. Must have the same shape as
            ``y_true``.

            For ``aggregation="per_image"``, the first dimension represents
            independent images.

        labels : array-like, optional
            Classes to evaluate. If None, classes are inferred from the
            ground-truth and predicted arrays.

            Use this parameter to exclude the background class, for example
            ``labels=[1, 2]``.

        average : {"macro", None}, default="macro"
            Class aggregation strategy:

                - ``"macro"``: return the unweighted arithmetic mean of the
                                class-wise IoU values.
                - ``None``: return one IoU value per class.

        aggregation : {"global", "per_image"}, default="global"
            Image aggregation strategy:

                - ``"global"``: accumulate intersection and union counts across
                        all images before computing IoU.
                - ``"per_image"``: compute IoU independently for each image and
                        average the resulting scores across images.

        ignore_index : int or str, optional
            Ground-truth label to ignore. Pixels for which
            ``y_true == ignore_index`` are excluded from the calculation.

        zero_division : float, default=0.0
            Score assigned to a class when it is absent from both ground truth
            and prediction.

    Returns
    -------
        float or numpy.ndarray
            If ``average="macro"``, returns a scalar IoU score.

            If ``average=None``, returns a one-dimensional array containing
                the IoU value for each evaluated class.

    Notes
    -----
        For each class, Dice and IoU are related by:

            Dice = 2 * IoU / (1 + IoU)

        This relationship holds for class-wise scores computed from the same
        intersection and union counts. It does not generally hold between
        separately averaged Dice and IoU values.
    """

    if not (np.isscalar(zero_division) and (
        np.isfinite(zero_division) or np.isnan(zero_division)
    )):
        raise ValueError("zero_division must be a finite number or np.nan.")

    y_true, y_pred, labels = _validate_inputs(y_true, y_pred, labels, ignore_index,
                                              aggregation)

    if aggregation == "global":
        _, iou = _compute_per_class(y_true, y_pred, labels, ignore_index)

    else:
        image_scores = []

        for ii in range(y_true.shape[0]):
            _, iou = _compute_per_class(y_true[ii], y_pred[ii], labels,
                                        ignore_index)
            image_scores.append(iou)

        iou = np.nanmean(np.asarray(image_scores), axis = 0)

    return _aggregate_scores(iou, average, zero_division)

    

    



