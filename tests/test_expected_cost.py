import numpy as np 
import pytest 

from segmentation_evaluation.metrics import (
    expected_cost,
    best_constant_cost,
    normalized_expected_cost,
    transform_cost_matrix,
)

@pytest.fixture
def example_matrices():
    cm = np.array([[8, 2], [1, 9]])
    cost_matrix = np.array([[0, 1], [3, 0]])

    return cm, cost_matrix


def test_expected_cost(example_matrices):
    cm, costs = example_matrices
    result = expected_cost(cm , costs)
    assert result == pytest.approx(0.25)

def test_best_constant_cost(example_matrices):
    cm, costs = example_matrices

    result = best_constant_cost(cm, costs)

    # Priors = [0.5, 0.5]
    # Always predict 0: 0.5*0 + 0.5*3 = 1.5
    # Always predict 1: 0.5*1 + 0.5*0 = 0.5
    assert result == pytest.approx(0.5)

def test_best_constant_class(example_matrices):
    cm, costs = example_matrices

    result, best_class = best_constant_cost(cm, costs, return_class = True)

    assert result == pytest.approx(0.5)
    assert best_class == 1

def test_normalized_expected_cost(example_matrices):
    cm, costs = example_matrices
    result = normalized_expected_cost(cm, costs)
    assert result == pytest.approx(0.25 /0.5)

def test_return_components(example_matrices):
    cm, costs = example_matrices
    result = normalized_expected_cost(cm, costs, return_components = True)

    assert result["expected cost"] == pytest.approx(0.25)
    assert result["best constant cost"] == pytest.approx(0.5)
    assert result["normalized expected cost"] == pytest.approx(0.25 /0.5)
    assert result["best constant class"] == 1

def test_zero_baseline():
    cm = np.array([[0, 1], [0, 1]])
    costs = np.zeros((2, 2))

    with pytest.raises(ValueError):
        normalized_expected_cost(cm, costs)

def test_rectangular_cost_matrix():
    cm = np.array([[8, 1, 1], [1, 8, 1]])
    cost_matrix = np.array([[0, 1, 0.3], [1, 0, 0.3]])

    result = expected_cost(cm, cost_matrix)
    assert result == pytest.approx(0.13)

def test_abstention_as_constant_decision():
    cm = np.array([[8, 1, 1], [1, 8, 1]])
    cost_matrix = np.array([[0, 1, 0.3], [1, 0, 0.3]])

    baseline, decision = best_constant_cost(cm, cost_matrix,
                                            return_class = True)

    assert baseline == pytest.approx(0.3)
    assert decision == 2

def test_negative_costs():
    cm = np.array([[8, 2], [1, 9]])
    cost_matrix = np.array([[-2, 1], [0, -1]])
    result = expected_cost(cm, cost_matrix)

    assert result == pytest.approx(-1.15)

def test_transform_cost_matrix():
    cost_matrix = np.array([[-2, 1, 0], [0, -1, 2]])
    expected = np.array([[0, 3, 2], [1, 0, 3]])
    result = transform_cost_matrix(cost_matrix)
    np.testing.assert_allclose(result, expected)

def test_transform_with_scale():
    cost_matrix = np.array([[-2, 1], [0, -1]])
    results = transform_cost_matrix(cost_matrix, scale = 2.0)
    expected = np.array([[0, 6], [2, 0]])
    np.testing.assert_allclose(results, expected)

def test_transformed_cost_are_nonnegative():
    cost_matrix = np.array([[-5, -2, 1], [3, -4, 0]])
    transformed = transform_cost_matrix(cost_matrix)
    assert np.all(transformed>=0)

def test_negative_baseline_raises():
    cm = np.array([[5, 0], [0, 5]])
    cost_matrix = np.array([[-2, 1], [-2, 1]])

    with pytest.raises(ValueError):
        normalized_expected_cost(cm, cost_matrix)

def test_mismatched_shapes():
    cm = np.array([[5, 2], [1, 4]])
    cost_matrix = np.array([[0, 1, 2], [1, 0, 2]])

    with pytest.raises(ValueError):
        expected_cost(cm, cost_matrix)

def test_negative_confusion_entries():
    cm = np.array([[5, -1], [1, 4]])
    cost_matrix = np.zeros((2, 2))

    with pytest.raises(ValueError):
        expected_cost(cm, cost_matrix)

def test_non_finite_costs():
    cm = np.array([[5, 2], [1, 4]])
    cost_matrix = np.array([[0, np.inf], [1, 0]])

    with pytest.raises(ValueError):
        expected_cost(cm, cost_matrix)

def test_invalid_scale():
    cost_matrix = np.array([[0, 1], [1, 0]])

    with pytest.raises(ValueError):
        transform_cost_matrix(cost_matrix, scale=0)

    with pytest.raises(ValueError):
        transform_cost_matrix(cost_matrix, scale=-1)



    