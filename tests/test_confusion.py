import numpy as np 
import pytest 

from segmentation_evaluation.metrics import confusion_matrix

def test_basic_confusion_matrix():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2, 2, 0])

    expected = np.array([[1, 1, 0],
                         [0, 1, 1],
                         [1, 0, 1]])

    results = confusion_matrix(y_true, y_pred, num_classes = 3)
    np.testing.assert_array_equal(results, expected)


def test_rectangular_confusion_matrix():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 3, 1, 3, 2, 0])

    expected = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [1, 0, 1, 0]])

    result = confusion_matrix(y_true, y_pred, labels = [0, 1, 2],
                              decision_labels = [0, 1, 2, 3])

    np.testing.assert_array_equal(result, expected)
    assert result.shape == (3, 4)

def test_string_abstention_label():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, "abstain", 1, "abstain"])

    result = confusion_matrix(y_true, y_pred, labels = [0, 1],
                              decision_labels = [0, 1, "abstain"])

    expected = np.array([
        [1, 0, 1], [0, 1, 1]
    ])
    np.testing.assert_array_equal(result, expected)

def test_normalize_true():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    results = confusion_matrix(
        y_true, y_pred, num_classes = 2, normalize = "true")

    expected = np.array([[0.5, 0.5], [0.0, 1.0]])

    np.testing.assert_allclose(results, expected)

def test_normalize_pred():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    
    results = confusion_matrix(
        y_true, y_pred, num_classes = 2, normalize = "pred")
    
    expected = np.array([[1.0, 1/3], [0.0, 2/3]])
    
    np.testing.assert_allclose(results, expected)

def test_normalize_all():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
        
    results = confusion_matrix(
        y_true, y_pred, num_classes = 2, normalize = "all")
    
    expected = np.array([[0.25, 0.25], [0.0, 0.5]])
    np.testing.assert_allclose(results, expected)
        

def test_ignore_index():
    y_true = np.array([0, 1, 2, 255])
    y_pred = np.array([0, 1, 1, 2])

    result = confusion_matrix(y_true, y_pred, num_classes = 3,
                              ignore_index = 255)
    assert result.sum() == 3

def test_explicit_labels():
    y_true = np.array([1, 1, 3, 3, 5, 5])
    y_pred = np.array([1, 3, 3, 5, 5, 1])

    result = confusion_matrix(
        y_true, y_pred, labels = [1, 3, 5], decision_labels = [1, 3, 5]
    )    
    expected = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    np.testing.assert_array_equal(result, expected)

def test_invalid_shapes():
    with pytest.raises(ValueError):
        confusion_matrix(np.array([0, 1]), np.array([0, 1, 2]),
                         num_classes = 3)

def test_invalid_normalization():
    with pytest.raises(ValueError):
        confusion_matrix(np.array([0, 1]), np.array([0, 1]),
                         num_classes = 2, normalize = "invalid")

def test_unknown_predicition_label():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 2])

    with pytest.raises(ValueError):
        confusion_matrix(y_true, y_pred, labels = [0, 1],
                         decision_labels = [0, 1])

def test_empty_after_ignore_index():
    y_true = np.array([255, 255])
    y_pred = np.array([0, 1])
    with pytest.raises(ValueError):
        confusion_matrix(y_true, y_pred, ignore_index = 255)

        

