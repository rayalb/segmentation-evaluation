import numpy as np 
import pytest 


from segmentation_evaluation.metrics import dice_score, iou_score

def test_perfect_segmentation():
    y = np.array([[0, 0, 1], [1, 2, 2]])

    assert dice_score(y, y) == pytest.approx(1.0)
    assert iou_score(y, y) == pytest.approx(1.0)

def test_dice_and_iou_binary():
    y_true = np.array([1, 1, 1, 0])
    y_pred = np.array([1, 1, 0, 0])

    assert dice_score(
        y_true, y_pred, labels = [1]
        ) == pytest.approx(4/5)
    assert iou_score(
            y_true, y_pred, labels = [1]
            ) == pytest.approx(2/3)

def test_macro_average():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    expected_dice = (2/3 + 4/5)/2
    expected_iou = (1/2 + 2/3)/2

    assert dice_score(y_true, y_pred) == pytest.approx(expected_dice)
    assert iou_score(y_true, y_pred) == pytest.approx(expected_iou)

def test_per_class_score():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    dice = dice_score(y_true, y_pred, average = None)
    iou = iou_score(y_true, y_pred, average = None)

    np.testing.assert_allclose(dice, [2/3, 4/5])
    np.testing.assert_allclose(iou, [1/2, 2/3])

def test_exclude_one_class():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    result = dice_score(y_true, y_pred, labels = [1, 2])
    expected = (4/5 + 2/3)/2

    assert result == pytest.approx(expected)

def test_global_vs_per_image():
    y_true = np.array([
        [1, 1, 1, 1], [1, 0, 0, 0]
    ])
    y_pred = np.array([
        [1, 1, 1, 1], [0, 0, 0, 0]
    ])
    global_dice = dice_score(y_true, y_pred, labels = [1],
                             aggregation = "global")
    mean_image_dice = dice_score(y_true, y_pred, labels = [1],
                                 aggregation = "per_image")

    assert global_dice == pytest.approx(8/9)
    assert mean_image_dice == pytest.approx(0.5)

def test_ignore_index():
    y_true = np.array([0, 1, 1, 255])
    y_pred = np.array([0, 1, 0, 1])

    result = dice_score(y_true, y_pred, labels = [1],
                        ignore_index = 255)
    assert result == pytest.approx(2/3)

def test_absent_class():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])

    result = dice_score(y_true, y_pred, labels = [1],
                        zero_division = 1.0)

    assert result == pytest.approx(1.0)

def test_shape_mismatch():
    with pytest.raises(ValueError):
        dice_score(np.array([0, 1]), np.array([0, 1, 1]))

def test_invalid_aggregation():
    with pytest.raises(ValueError):
        dice_score(
            np.array([0, 1]), np.array([0, 1]),
            aggregation = "invalid"
        )

def test_per_image_requieres_multiple_dimensions():
    with pytest.raises(ValueError):
        dice_score(np.array([0, 1]), np.array([0, 1]),
                   aggregation = "per_image")



        