import numpy as np 
import pytest 
     
from segmentation_evaluation.decisions import (
    bayes_predict,
    expected_risks,
)

def test_equal_cost_classification():
    probabilities = np.array([
        [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]
    ])
    cost_matrix = np.array([[0, 1], [1, 0]])
    predictions = bayes_predict(probabilities, cost_matrix)
    expected = np.array([0, 1, 0])

    np.testing.assert_array_equal(predictions, expected)

def test_asymetric_costs():
    probabilities = np.array([[0.8, 0.2]])
    cost_matrix = np.array([[0, 1], [5, 0]])
    predictions = bayes_predict(probabilities, cost_matrix)

    # Risk of decision 0 = 0.8*0 + 0.2*5 = 1.0
    # Risk of decision 1 = 0.8*1 + 0.2*0 = 0.8

    np.testing.assert_array_equal(predictions, [1])

def test_rectangular_cost_matrix():
    probabilities = np.array([
        [0.8, 0.2], [0.4, 0.6]
    ]) 
    cost_matrix = np.array([
        [0, 1, 0.3], [1, 0, 0.3]
    ])
    predictions = bayes_predict(probabilities, cost_matrix)

    # First observation: decision 0
    # Second observation: abstetion (decision 2)

    np.testing.assert_array_equal(predictions, [0, 2])

def test_abstention_label():
    probabilities = np.array([
        [0.8, 0.2], [0.4, 0.6]
    ])
    cost_matrix = np.array([
        [0, 1, 0.3], [1, 0, 0.3]
    ])
    predictions = bayes_predict(probabilities, cost_matrix,
                                decision_labels = [0, 1, "abstain"])
    np.testing.assert_array_equal(predictions, np.array([0, "abstain"]))

def test_expected_risks():
    probabilities = np.array([
            [0.8, 0.2], [0.4, 0.6]
        ])
    cost_matrix = np.array([
            [0, 1, 0.3], [1, 0, 0.3]
        ])
    risks = expected_risks(probabilities, cost_matrix)

    expected = np.array([
        [0.2, 0.8, 0.3], [0.6, 0.4, 0.3]
    ])
    np.testing.assert_allclose(risks, expected)

def test_return_risks():
    probabilities = np.array([[0.8, 0.2]])
    cost_matrix = np.array([
            [0, 1], [1, 0]
        ])
    predictions, risks = bayes_predict(probabilities, cost_matrix,
                                      return_risks = True)
    np.testing.assert_array_equal(predictions, [0])
    np.testing.assert_allclose(risks, [[0.2, 0.8]])

def test_image_shaped_probabilities():
    probabilities = np.array([
        [
            [0.8, 0.2], [0.3, 0.7],
        ],
        [
            [0.6, 0.4], [0.1, 0.9],
        ],
    ])
    cost_matrix = np.array([
        [0, 1], [1, 0],
    ])
    predictions = bayes_predict(probabilities, cost_matrix)
    expected = np.array([[0, 1], [0, 1]])

    np.testing.assert_array_equal(predictions, expected)
    assert predictions.shape == (2, 2)

def test_invalid_probability_shape():
    probabilities = np.array([0.8, 0.2])

    cost_matrix = np.array([
        [0, 1], [1, 0],
    ])

    # A one-dimensional probability vector is valid.
    predictions = bayes_predict(probabilities, cost_matrix)

    assert predictions.ndim == 0

def test_probabilities_do_not_sum_to_one():
    probabilities = np.array([
        [0.8, 0.3],
    ])

    cost_matrix = np.array([
        [0, 1], [1, 0],
    ])

    with pytest.raises(ValueError):
        bayes_predict(probabilities, cost_matrix)

def test_negative_probabilities():
    probabilities = np.array([
        [1.2, -0.2],
    ])

    cost_matrix = np.array([
        [0, 1], [1, 0],
    ])

    with pytest.raises(ValueError):
        bayes_predict(probabilities, cost_matrix)

def test_non_finite_probabilities():
    probabilities = np.array([
        [np.nan, 0.5],
    ])

    cost_matrix = np.array([
        [0, 1], [1, 0],
    ])

    with pytest.raises(ValueError):
        bayes_predict(probabilities, cost_matrix)

def test_incompatible_cost_matrix():
    probabilities = np.array([
        [0.8, 0.2],
    ])

    cost_matrix = np.array([
        [0, 1, 2],
    ])

    with pytest.raises(ValueError):
        bayes_predict(probabilities, cost_matrix)

def test_invalid_decision_labels():
    probabilities = np.array([
        [0.8, 0.2],
    ])

    cost_matrix = np.array([
        [0, 1], [1, 0],
    ])

    with pytest.raises(ValueError):
        bayes_predict(probabilities, cost_matrix,
            decision_labels=[0, 1, 2]
        )


def test_duplicate_decision_labels():
    probabilities = np.array([
        [0.8, 0.2],
    ])

    cost_matrix = np.array([
        [0, 1], [1, 0],
    ])

    with pytest.raises(ValueError):
        bayes_predict(probabilities, cost_matrix,
            decision_labels=[0, 0]
        )