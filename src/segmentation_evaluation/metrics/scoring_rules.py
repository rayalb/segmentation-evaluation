"""
Proper scoring rules for evaluating predictions.

A scoring rule evaluates a probabilistic prediction by comparing
the predicted probability distribution with the observed outcome.

Let: p = (p_1, ..., p_K)

be the predicted probability vector for K classes, and let 
y denote the observed class. 

The following scoring rule are implemented:

    - Logarithm score (cross-entropy)
    - Multiclass Brier score
    - Spherical score
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

def _validate_inputs(y_true: ArrayLike, probabilities: ArrayLike,
                     labels: ArrayLike | None = None,
                     ignore_index: Any | None = None,
                     aggregation: Literal["global", "per_image"] = "global"
                     ) -> tuple[NDArray[Any], NDArray[np.float64] ,NDArray[Any]]:
    """
    Validate ground-truth labels and predicted class probabilities,

    Parameters
    ----------
        y_true : ArrayLike
            Ground-truth class labels.
        
        probabilities : ArrayLike
            Predicted class probabilities. The last dimension 
            represent the K classes. All probabilities must be finite
            and non-negative, and must sum to one along the last axis.

        labels : ArrayLike, optional
            Class labels corresponding to the probability columns.
            If None, labels are assumed to be encoded as integers
            0, ..., K-1.
            If provided, ``labels[k]`` indentifies the class represented by
            probability column k.
        
        ignore_index : Any, optional
            Ground.truth label to ignore. Observations with 
            ``y_true == ignore_index`` are excluded from the evaluation.

        aggregation : {"global", "per_image"}, default = "global"
            Aggregation strategy.
            For "per_image", the first dimension represents independent images.

    Returns
    -------
        y_indices : NDArray[np.int64]
            Integer-encoded ground-truth labels.
        
        probabilities : NDArray[np.float64]
            Validated probability array.
        
        image_indices : NDArray[np.int64]
            Image index associated with each valid observations.
    """

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype = np.float64)

    if probabilities.ndim < 2:
        raise ValueError(
            "probabilities must have at least two dimensions, "
            "including the class dimension."
        )
    if y_true.shape != probabilities.shape[:-1]:
        raise ValueError(
            "y_true shape must match probabilities.shape[:-1]."
        )
    if y_true.size == 0 or probabilities.size == 0:
        raise ValueError("y_true and probabilities cannot be empty")
    if aggregation not in ("global", "per_image"):
        raise ValueError("aggregation must be 'global' or 'per_image'")

    if aggregation == "per_image" and y_true.ndim < 2:
        raise ValueError(
            "For 'per_image' aggregation, the first axis must "
            "represent images."
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must contain only finite values.")
    if np.any(probabilities < 0):
        raise ValueError("probabilities must be non.negative")
    if not np.allclose(probabilities.sum(axis = -1), 1.0,
                       rtol = 1e-7, atol = 1e-8):
        raise ValueError("probabilities must sum to one along the last axis.")

    n_classes = probabilities.shape[-1]

    if n_classes < 2:
        raise ValueError("At least two classes are required.")

    if labels is None:
        labels = np.arange(n_classes)
    else:
        labels = np.asarray(labels)

    if labels.ndim != 1 or labels.size != n_classes:
        raise ValueError(
            "labels must be one-dimension and have one entry "
            "per probability column. "
        )
    if np.unique(labels).size != labels.size:
        raise ValueError("labels must contain unique values.")

    # Identify observations that should be evaluated.
    if ignore_index is not None:
        valid = y_true != ignore_index
    else:
        valid = np.ones(y_true.shape, dtype = bool)

    if not np.any(valid):
        raise ValueError(
            "No valid observations remain abster applying ignore_index."
        )

    # Convert ground-truth labels to probability-column indices.
    y_indices = np.full(y_true.shape, -1, dtype = np.int64)

    for kk, label in enumerate(labels):
        y_indices[y_true == label] = kk

    unknown = valid & (y_indices == -1)

    if np.any(unknown):
        raise ValueError(
            "y_true contains labels not present in labels."
        )

    # Flatten observations while preserving the image association.
    y_indices = y_indices[valid]
    probabilities = probabilities[valid]

    if aggregation == "per_image":
        image_grid = np.broadcast_to(
            np.arange(y_true.shape[0]).reshape((-1,) + (1,)*(y_true.ndim - 1)), y_true.shape)
        image_indices = image_grid[valid]
    else:
        image_indices = np.zeros(y_indices.shape, dtype = np.int64)

    return y_indices, probabilities, image_indices

def _aggregate_scores(scores: NDArray[np.float64], image_indices: NDArray[np.int64],
                      aggregation: Literal["global", "per_image"], 
                      average: Literal["mean"] | None, n_images: int,
                      ) -> float | NDArray[np.float64]:
    """
    Agregate observatons-level scores.

    Parameters
    ----------
        scores : NDArray[np.float64]
            Score associated with each valid observation.

        image_indices : NDArray[np.int64]
            Image index associated with each observation.

        aggregation : {"global", "per_image"}
            Aggregation strategy.

        average : {"mean", None}
            If "mean", return a scalar average.
            If None, return the individual scores or per-image averages.

        n_images : int
            Total number of image in the original input.

    Returns
    -------
        float or NDArray[np.float64]
            Aggregated score or individual observation scores.
    """

    if aggregation == "per_image":
        sums = np.bincount(image_indices, weights = scores, minlength = n_images)
        counts = np.bincount(image_indices, minlength = n_images)
        image_scores = np.full(n_images, np.nan, dtype = np.float64)
        valid_images = counts > 0
        image_scores[valid_images] = sums[valid_images]/counts[valid_images]

        scores = image_scores

    if average is None:
        return scores
    if average == "mean":
        return float(np.nanmean(scores))

    raise ValueError("average must be 'mean' or None.")

def cross_entropy(y_true: ArrayLike, probabilities: ArrayLike, *, 
                  labels : ArrayLike | None = None, 
                  average: Literal["mean"] | None = "mean", 
                  aggregation: Literal["global", "per_image"] = "global",
                  ignore_index: Any | None = None, eps: float = 1e-15,
                  ) -> float | NDArray[np.float64]:
    """
    Compute the multiclass logarithmic scoring rule (cross-entropy).

    For an observation with true class y, the logarithm loss is:

            L(p, y) = -log(p_y)

    where p_y is the predicted probability assigned to the true class.

    The logarithmic score is strictly proper: its expected value is 
    uniquely minimized when the reported probabilities equal the true
    conditional probabilities.

    Parameters
    ----------
        y_true : ArrayLike
            Ground-truth class labels.

        probabilities : ArrayLike
            Predicted class probabilities with shape (..., K). The last
            dimension represents the classes.

        labels : ArrayLike, optional
            Class labels correspondinf to the probability columns. 
            If None, integer labels 0, ..., K-1 are assumed.

        average : {"mean", None}, default = "mean"
            Observation aggregation strategy:
                - "mean": return the mean score.
                - None : return individual observation scores for global
                        aggregation, or one mean score per image for 
                        per-image aggregation.

        aggregation : {"global", "per_image"}, default = "global"
            Observation aggregaton strategy:
                - "global": average over all valid observations.
                - "per_image": average observations within each image first,
                    then average the rulting scores.

        ignore_index : Any, optional.
            Ground-truth label to ignore.

        eps : float, default = 1e-15
            Lower bound used for numerical stability when evaluating 
            log(p_y). Must be strictly positive and smaller than one.

    Returns
    -------
        float or NDArray[np.float64]
            Mean cross-entropy or individual observation/per-image scores.

    Notes
    -----
        Cross-entropy is a loss, so lower values indicates better 
        probabilistic predictions.  
    """

    if not np.isfinite(eps) or not 0 < eps < 1:
        raise ValueError("eps must be finite and satisfy 0<eps<1.")

    y_indices, probabilities, image_indices = _validate_inputs(
        y_true, probabilities, labels, ignore_index, aggregation
    )

    true_probabilities = probabilities[np.arange(len(y_indices)), y_indices]

    scores = -np.log(np.maximum(true_probabilities, eps))
    return _aggregate_scores(scores, image_indices, aggregation, average,
                             np.asarray(y_true).shape[0] if aggregation == "per_image" else 1)

def brier_score(y_true: ArrayLike, probabilities: ArrayLike, *,
                labels: ArrayLike | None = None, average: Literal["mean"] | None = "mean",
                aggregation: Literal["global", "per_image"] = "global", 
                ignore_index: Any | None = None) -> float | NDArray[np.float64]:
    """
    Compute the multiclass Brier Score.

    The Brier score measures the squared difference between the predicted probability vector and 
    the one-hot encoded outcome.

    For and observation with true class y, its is defined as:
    
            BS(p, y) = sum_k (p_k - o_k)^2

    where o_k = 1 if y = k, 0 otherwise.

    The multiclass Brier score is strictly proper. Is expected value is uniquely minimized when the 
    reported probabilities equal the true conditional probabilities.

    Parameters
    ----------
        y_true : ArrayLike
            Ground-truth class labels.

        probabilites : ArrayLike
            Predicted class probabilities with shape (..., K).

        labels : ArrayLike, optional
            Class labels corresponding to the probability columns.
            If None, integer labels 0, ..., K-1 are assumed.

        average : {"mean", None}, default = "mean"
            Observations aggregation strategy:
                - "mean": return the mean score.
                - None : return individual observation scores for global
                        aggregation, or one mean score per image for 
                        per-image aggregation.
                
        aggregation : {"global", "per_image"}, default = "global"
            Observation aggregaton strategy:
                - "global": average over all valid observations.
                - "per_image": average observations within each image first,
                        then average the rulting scores.
                
        ignore_index : Any, optional.
            Ground-truth label to ignore.
                
    Returns
    -------
        float or NDArray[np.float64]
            Mean Brier score or individual observation/per-image scores.
    
    Notes
    -----
        This implemetation uses the sum over classes, not the mean over classes.
        Therefore, for K classes, the Brier score range from 0 to 2.

        Lower values indicates beterr probabilitistic predictions.
    """
    y_indices, probabilities, image_indices = _validate_inputs(
        y_true, probabilities, labels, ignore_index, aggregation
    )
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(y_indices)), y_indices] = 1.0

    scores = np.sum((probabilities - one_hot)**2, axis = -1)

    return _aggregate_scores(scores, image_indices, aggregation, average,
                             np.asarray(y_true).shape[0] if aggregation == "per_image" else 1)

def spherical_score(y_true: ArrayLike, probabilities: ArrayLike, *, 
                    labels: ArrayLike | None = None, 
                    average: Literal["mean"] | None = "mean",
                    aggregation: Literal["global", "per_image"] = "global",
                    ignore_index: Any | None = None) -> float | NDArray[np.float64]:
    """
    Compute the multiclass spherical scorinig rule.

    The spherical score is defined as:

        SS(p, y) = p_y/||p||_2

    where: p_y = probability assigned to the true class, 
        ||p||_2 = sqrt(sum_k p_k^2).

    The spherical score is strictly proper. Its expected value is uniquely 
    maximized when the reported probabilities equal the true conditional
    probablities.

    Parameters
    ----------
        y_true : ArrayLike
            Ground-truth class labels.

        probabilities : ArrayLike
            predicted class probabilities with shape (..., K)

        labels : ArrayLike, optional
            Class labels corresponding to the probability columns.
            If None, integer labels 0, ..., K-1 are assumed.

        average : {"mean", None}, default = "mean"
            Observations aggregation strategy:
                - "mean": return the mean score.
                - None : return individual observation scores for global
                        aggregation, or one mean score per image for 
                        per-image aggregation.
                
        aggregation : {"global", "per_image"}, default = "global"
            Observation aggregaton strategy:
                - "global": average over all valid observations.
                - "per_image": average observations within each image first,
                        then average the rulting scores.
                
        ignore_index : Any, optional.
            Ground-truth label to ignore.
                
    Returns
    -------
        float or NDArray[np.float64]
            Mean spherical score or individual observation/per-image scores.

    Notes
    -----
        The spherical score is a reward rather than a loss. Higher values 
        indicate better probabilistic predictions.

        For a uniform distribution over K classes, the score is 1/sqrt(K).
        For a perfectly certain and correct prediction, the score is 1.
    """

    y_indices, probabilities, image_indices = _validate_inputs(
        y_true, probabilities, labels, ignore_index, aggregation
    )

    true_probabilities = probabilities[np.arange(len(y_indices)), y_indices]

    norms = np.linalg.norm(probabilities, axis = -1)

    scores = true_probabilities/norms

    return _aggregate_scores(scores, image_indices, aggregation, average,
                             np.asarray(y_true).shape[0] if aggregation == "per_image" else 1)



    