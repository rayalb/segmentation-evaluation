from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

def _validate_matrices(confusion: ArrayLike, cost_matrix: ArrayLike,
                       ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Validate  a confusion matrix and its corresponding cost matrix.

    The matrices must have the same shape (K, J)

    Parameters
    ----------
        confusion : array-like
            Confusion matrix.
        cost_matrix :  array-like
            cost matrix, negative costs are allowed.

    Notes
    -----
        Rows =  true classes
        Columns = prediction decisions
    """

    cm = np.asarray(confusion, dtype = np.float64)
    costs = np.asarray(cost_matrix, dtype = np.float64)

    if cm.ndim != 2:
        raise ValueError("Confusion must be a 2D array.")

    if costs.ndim != 2:
        raise ValueError("Cost matrix must be a 2D array.")

    if cm.shape != costs.shape:
        raise ValueError(
            f"Confusion matrix and costs matrix must have the same shape."
            f"Got {cm.shape} and {costs.shape}"
        )

    if not np.all(np.isfinite(cm)):
        raise ValueError("Confusion matrix must contain finite values.")

    if not np.all(np.isfinite(costs)):
        raise ValueError("Cost_matrix must contain finite values.")

    if np.any(cm < 0):
        raise ValueError("Confusion matrix cannot contain negative values.")

    if cm.sum() <= 0:
        raise ValueError("Confusion matrix must contain at least one observation.")

    return cm, costs

def expected_cost(confusion: ArrayLike, cost_matrix: ArrayLike) -> float:
    """
    Compute the empirical expected cost of a classifier.

    Parameters
    ----------
        confusion : array-like, shape (K, J)
                Confusion matrix. Rows are true classes and columns are predicted classes.
        cost_matrix : array-like, shape (K, J)
                Cost matrix. cost_matrix[ii, jj] is the cost of predictinj jj 
                            when the true class is ii.
    
    Returns
    -------
        float
            Empirical average cost per observation.
    """

    cm, costs = _validate_matrices(confusion, cost_matrix)

    return float(np.sum(cm * costs)/cm.sum())

def best_constant_cost(confusion: ArrayLike, cost_matrix: ArrayLike, *,
                       return_class: bool = False) -> float | tuple[float, int]:
    """
    Compute the minimum expected cost of a constant classifier.
    The constant classifier always predicts the same class, regardless of the input.
    
    Class priors are estimated from the row sums of the confusion matrix.
    
    Parameters
    ----------
        confusion : array-like, shape (K, J)
                Confusion matrix.
        cost_matrix : array-like, shape (K, J) 
                Cost matrix.
        return_class : bool, default = False
                If true, also return the class that minimizes the constant
                classifier's expected cost.
    
    Returns
    -------
        float or tuple
            Minimum constant decision cost. If return_class = True,
            return (minimum_cost, optimal_constant_class).
    """

    cm, costs = _validate_matrices(confusion, cost_matrix)
    class_priors = cm.sum(axis = 1)/cm.sum()

    # Expected cost for each possible constant prediction
    constant_cost = class_priors @ costs
    best_class = int(np.argmin(constant_cost))
    minimum_cost = float(constant_cost[best_class])

    if return_class:
        return minimum_cost, best_class
    return minimum_cost

def normalized_expected_cost(confusion: ArrayLike, cost_matrix: ArrayLike, *,
                             return_components: bool = False,
                            ) -> float | dict[str, float | int]:
    """
    Compute the normalized expected cost (NEC)

    NEC is the empirical expected cost divided by the minimum expected
    cost achievable by a constant classifier that has access to the 
    emprical class priors.

    Parameters
    ----------
        confusion : array-like, shape (K, J)
                Confusion matrix.
        cost_matrix : array-like, shape (K, J)
                Cost matrix.
        returns_components : bool, default = False
            If True, return NEC together with its components.

    Returns
    -------
        float or dict
            Normalized expected cost, or a dictionary containing:
                - expected cost
                - best constant cost
                - normalized expected cost
                - best constant class
    """
    cm, costs = _validate_matrices(confusion, cost_matrix)
    ec = expected_cost(cm, costs)
    baseline, best_class = best_constant_cost(cm, costs, return_class = True)

    if baseline <= 0:
        raise ValueError(
            "NEC requieres a strictly best constant cost."
            f"Got baseline cost = {baseline}."
            "Consider transforming the cost matrix."
        )

    nec = ec/baseline

    if return_components:
        return {
            "expected cost": ec,
            "best constant cost": baseline,
            "normalized expected cost": nec,
            "best constant class": best_class,
        }

    return float(nec)

def transform_cost_matrix(cost_matrix: ArrayLike, *, 
                          scale: float = 1.0) -> NDArray[np.float64]:
    """
    Transform a cost matrix using row-wise minimum substraction.
    
    Transformed cost are defines as:
        c_new[ii, jj] = scale * (c[ii, jj]-min_j(c[ii, jj]))

    This transformation: 
        - Produces nonnegative costs.
        - Preserves Bayes-optimal decisions.
        - Allows negative entries in the original cost matrix.

    Parameters
    ----------
        cost_matrix : array-like, shape (K, J)
            original cost matrix.
        scale : float, default = 1.0
            Positive scaling factor.
    
    Returns
    -------
        ndarray
            Transformed cost matrix.
    """

    cost_matrix = np.asarray(cost_matrix, dtype = np.float64)

    if cost_matrix.ndim != 2:
        raise ValueError("cost_matrix must be a 2D array")

    if cost_matrix.shape[0] == 0 or cost_matrix.shape[1] == 0:
        raise ValueError("cost_matrix cannot have empry dimensions.")

    if not np.all(np.isfinite(cost_matrix)):
        raise ValueError("cost_matrix must contain only finite values.")

    if not np.isscalar(scale) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a finite positive scalar.")

    row_minima = np.min(cost_matrix, axis = 1, keepdims = True)

    return scale*(cost_matrix-row_minima)


    
