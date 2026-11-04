import numpy as np 
import pytest 

from segmentation_evaluation.metrics import(
    cross_entropy,
    brier_score,
    spherical_score,
)

def test_perfect_predictions():
    y_true = np.array([0, 1, 2])
    probabilities = np.eye(3)

    assert cross_entropy(y_true, probabilities) == pytest.approx(0.0)
    assert brier_score(y_true, probabilities) == pytest.approx(0.0)
    assert spherical_score(y_true, probabilities) == pytest.approx(1.0)

def test_cross_entropy():
    y_true = np.array([0, 1])
    probabilities = np.array([
        [0.8, 0.2], [0.3, 0.7]
    ])
    expected = (-np.log(0.8)-np.log(0.7))/2
    assert cross_entropy(y_true, probabilities) == pytest.approx(expected)

def test_brier_score():
    y_true = np.array([0, 1])
    probabilities = np.array([
        [0.8, 0.2], [0.3, 0.7]
    ])
    expected = ((0.8-1)**2 + 0.2**2 + 0.3**2 + (0.7-1)**2)/2
    assert brier_score(y_true, probabilities) == pytest.approx(expected)

def test_spherical_score():
    y_true = np.array([0])
    probabilities = np.array([[0.8, 0.2]])

    expected = 0.8 / np.sqrt(0.8**2 + 0.2**2)

    assert spherical_score(y_true, probabilities) == pytest.approx(expected)

def test_arbitrary_labels():
    y_true = np.array([10, 20])
    probabilities = np.array([
            [0.8, 0.2], [0.3, 0.7]
        ])
    expected = (-np.log(0.8)-np.log(0.7))/2
    assert cross_entropy(y_true, probabilities, labels = [10, 20]) == pytest.approx(expected)

def test_ignore_index():
    y_true = np.array([0, 1, 255])
    probabilities = np.array([
        [0.8, 0.2], [0.3, 0.7], [0.1, 0.9]
    ])
    result = cross_entropy(y_true, probabilities, ignore_index = 255)
    expected = (-np.log(0.8)-np.log(0.7))/2
    assert result == pytest.approx(expected)

def test_individual_scores():
    y_true = np.array([0, 1])
    probabilities = np.array([
        [0.8, 0.2], [0.3, 0.7]
    ])
    result = cross_entropy(y_true, probabilities, average = None)
    np.testing.assert_allclose(result, [-np.log(0.8), -np.log(0.7)])

def test_per_image_individual_scores():
    y_true = np.array([[0, 0], [1, 1]])

    probabilities = np.array([
        [[0.9, 0.1], [0.9, 0.1]],
        [[0.2, 0.8], [0.2, 0.8]],
    ])

    result = cross_entropy(y_true, probabilities, aggregation="per_image", 
                           average=None)

    expected = [-np.log(0.9), -np.log(0.8)]

    np.testing.assert_allclose(result, expected)

def test_probability_shape_mismatch():
    with pytest.raises(ValueError):
        cross_entropy(
            np.array([0, 1]),
            np.array([[0.8, 0.2]]),
        )


def test_probabilities_do_not_sum_to_one():
    with pytest.raises(ValueError):
        cross_entropy(
            np.array([0]),
            np.array([[0.8, 0.8]]),
        )


def test_negative_probabilities():
    with pytest.raises(ValueError):
        brier_score(
            np.array([0]),
            np.array([[1.2, -0.2]]),
        )


def test_non_finite_probabilities():
    with pytest.raises(ValueError):
        spherical_score(
            np.array([0]),
            np.array([[np.nan, 1.0]]),
        )


def test_invalid_labels():
    with pytest.raises(ValueError):
        cross_entropy(
            np.array([0, 2]),
            np.array([
                [0.8, 0.2],
                [0.3, 0.7],
            ]),
        )


def test_invalid_eps():
    with pytest.raises(ValueError):
        cross_entropy(
            np.array([0]),
            np.array([[0.8, 0.2]]),
            eps=0,
        )


def test_all_observations_ignored():
    with pytest.raises(ValueError):
        cross_entropy(
            np.array([255, 255]),
            np.array([
                [0.8, 0.2],
                [0.3, 0.7],
            ]),
            ignore_index=255,
        )
    