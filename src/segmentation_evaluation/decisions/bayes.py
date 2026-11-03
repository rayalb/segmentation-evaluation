from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

def _validate_inputs(probabilities: ArrayLike, cost_matrix: ArrayLike
                     ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Validate posterior probabilities and a cost matrix.
    
    Parameters
    ----------
        probabilities : array-like, shape (..., K)
            Posterior probabilities for K true classes.
            
            The last dimension represents the true class.
            All preceding dimensions represent observations.
            
        cost_matrix : array-like, shape (K, J)
            Cost matrix, rows represent true classes and 
            columns represent available decisions.
            Negative costs are allowed.
            
    Returns
    -------
        probabilities : ndarray
            Validated probabilities.
            
        cost_matrix : ndarray
            Validated cost matrix
    """

    probabilities = np.asarray(probabilities, dtype = np.float64)
    cost_matrix = np.asarray(cost_matrix, dtype = np.float64)

    if probabilities.ndim < 1:
        raise ValueError("probabilities must have at least one dimension.")

    if cost_matrix.ndim != 2:
        raise ValueError("cost_matrix must be a 2D array.")

    if probabilities.shape[-1] != cost_matrix.shape[0]:
        raise ValueError(
            "The last dimension of probabilities must match the "
            "number of rows in cost matrix."
            f"Got {probabilities.shape[-1]} and "
            f"{cost_matrix.shape[0]}."
        )
    if cost_matrix.shape[0] == 0 or cost_matrix.shape[1] == 0:
        raise ValueError("cost_matrix cannot have empty dimensions.")

    if probabilities.size == 0:
        raise ValueError("probabilities cannot be empty")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must contain only finite values.")

    if not np.all(np.isfinite(cost_matrix)):
        raise ValueError("cost_matrix must contain only finite values.")

    if np.any(probabilities < 0):
        raise ValueError("probabilities cannot contain negative values.")

    # Each observation must have probabilities summing to one
    probabilities_sum = probabilities.sum(axis = -1)

    if not np.allclose(probabilities_sum, 1.0, rtol = 1e-7, atol = 1e-8):
        raise ValueError("Probabilities must sum to one along the last axis.")

    return probabilities, cost_matrix

def expected_risks(probabilities: ArrayLike, cost_matrix: ArrayLike
                   ) -> NDArray[np.float64]:
    """
    Calculate the posterior expected cost of every decision.
    
    R[n, j] = sum_i probabilities[n, i]*cost_matrix[i, j]
    
    Parameters
    ----------
        probabilities : array-like, shape (..., K)
            posterior probabilities.
        
        cost_matrix : array-like, shape (K, J)
            Cost matrix.
    
    Returns
    -------
        ndarray, shape (..., J)
            Posterior expected risks for every available decision.
    """

    probabilities, cost_matrix = _validate_inputs(probabilities, cost_matrix)

    return probabilities @ cost_matrix

def bayes_predict(probabilities: ArrayLike, cost_matrix: ArrayLike, *,
                  decision_labels: ArrayLike | None = None,
                  return_risks: bool = False,
                  ) -> NDArray | tuple[NDArray, NDArray[np.float64]]:
    """
    Compute Bayes - optimal decision by minimizing posterior expected cost.

    The selected decision is:
        j* = argmin_j sum_i p_i*c_ij

    Parameters
    ----------
        probabilities : array-like, shape (..., K)
            Posterior probabilities.
    
        cost_matrix : array-like, shape (K, J)
            Cost matrix. NEgative costs and rectangula matrices are
            supported.
        
        decision_labels : array-like, shape (J,), optional
            Labels associated with the available decisions.
            If None, integer decision indices are returned:
            0, 1, ..., J-1

            This can be used to assign a label such as "abstain"
            to an additional decision.

        return_risks : bool, default = False
            If True, also return the posterior expected risk
            associated with every decision.

    Returns
    -------
        predictions : ndarray, shape (...)
            Optimal decision indices or decision labels.

        risks : ndarray, shape (..., J), optional
            Posterior expected cost for all decisions.
            returned only when return_risks = True.

    Notes
    -----
        If multiple decisions have exactly the same minimum expected cost, 
        np.argmin selects the first one.
    """ 

    risks = expected_risks(probabilities, cost_matrix)

    # Select the decision with minimum expected risk.
    decision_indices = np.argmin(risks, axis = -1)

    # Optionally map decision to user - define labels
    if decision_labels is not None:
        decision_labels = np.asarray(decision_labels)
        if decision_labels.ndim != 1:
            raise ValueError("decision_labels must be 1D array.")

        if decision_labels.size != risks.shape[-1]:
            raise ValueError(
                "decision_labels must contain one label"
                " for each column of cost_matrix."
            )
        if np.unique(decision_labels).size != decision_labels.size:
            raise ValueError("decision_labels must be unique.")

        predictions = decision_labels[decision_indices]

    else:
        predictions = decision_indices

    if return_risks:
        return predictions, risks

    return predictions

    