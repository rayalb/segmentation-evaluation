import numpy as np 
import pytest 

from segmentation_evaluation.calibration import (
    OVRPlattScaling,
    ovr_platt_scaling,
)

@pytest.fixture
def calibration_data():
    rng = np.random.default_rng(43)

    n_samples = 300
    n_classes = 3

    y_true = rng.integers(0, n_classes, size = n_samples)

    # Generate logits with deliberately imperfect confidence
    logits = rng.normal(size = (n_samples, n_classes))
    logits[np.arange(n_samples), y_true] += 1
    return logits, y_true

def test_initial_state():
    calibrator = OVRPlattScaling()
    assert calibrator.is_fitted_ is False
    assert calibrator.parameters_ is None

def test_predict_before_fit_raises():
    calibrator = OVRPlattScaling()

    with pytest.raises(RuntimeError):
        calibrator.predict_proba(np.zeros((10, 3)))

def test_fit_returns_self(calibration_data):
    logits, y_true  = calibration_data
    calibrator = OVRPlattScaling()

    result = calibrator.fit(logits, y_true)
    assert result is calibrator
    assert calibrator.is_fitted_ 

def test_parameters_have_correct_shape(calibration_data):
    logits, y_true = calibration_data
    calibrator = OVRPlattScaling().fit(logits, y_true)

    assert calibrator.parameters_.shape == (3, 2)
    assert calibrator.slopes_.shape == (3,)
    assert calibrator.intercepts_.shape == (3,)
    assert calibrator.n_iter_.shape == (3,)

def test_predict_proba_shape(calibration_data):
    logits, y_true = calibration_data
    calibrator = OVRPlattScaling().fit(logits, y_true)
    probabilities = calibrator.predict_proba(logits)

    assert probabilities.shape == logits.shape

def test_probabilities_are_valid(calibration_data):
    logits, y_true = calibration_data
    calibrator = OVRPlattScaling().fit(logits, y_true)

    probabilities = calibrator.predict_proba(logits)

    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)

    np.testing.assert_allclose(
        probabilities.sum(axis = -1), 1.0, atol = 1e-12
    )

def test_segmentation_shape():
    rng = np.random.default_rng(10)

    logits = rng.normal(size=(4, 20, 30, 3))
    y_true = rng.integers(0, 3, size=(4, 20, 30))

    calibrator = OVRPlattScaling().fit(logits, y_true)
    probabilities = calibrator.predict_proba(logits)

    assert probabilities.shape == (4, 20, 30, 3)
    np.testing.assert_allclose(
        probabilities.sum(axis=-1), 1.0, atol=1e-12,
    )

def test_arbitrary_class_labels():
    rng = np.random.default_rng(11)

    logits = rng.normal(size=(200, 3))
    y_true = rng.choice(
        np.array([1, 3, 9]),
        size=200,
    )

    calibrator = OVRPlattScaling().fit(
        logits,
        y_true,
        labels=[1, 3, 9],
    )

    assert calibrator.classes_.tolist() == [1, 3, 9]

    probabilities = calibrator.predict_proba(logits)
    assert probabilities.shape == (200, 3)

def test_convenience_function_returns_calibrator(calibration_data):
    logits, y_true = calibration_data

    probabilities, calibrator = ovr_platt_scaling(
        logits,
        y_true,
        return_calibrator=True,
    )

    assert isinstance(calibrator, OVRPlattScaling)
    assert calibrator.is_fitted_

    np.testing.assert_allclose(
        probabilities,
        calibrator.predict_proba(logits),
    )


def test_validation_fit_test_transform(calibration_data):
    logits, y_true = calibration_data

    logits_val = logits[:200]
    y_val = y_true[:200]

    logits_test = logits[200:]

    probabilities, calibrator = ovr_platt_scaling(
        logits_val,
        y_val,
        logits_test=logits_test,
        return_calibrator=True,
    )

    assert probabilities.shape == logits_test.shape
    assert calibrator.is_fitted_


def test_missing_class_raises():
    rng = np.random.default_rng(13)

    logits = rng.normal(size=(100, 3))
    y_true = rng.integers(0, 2, size=100)

    with pytest.raises(ValueError, match="both positive and negative"):
        OVRPlattScaling().fit( logits, y_true, labels=[0, 1, 2],
        )


def test_invalid_logits():
    calibrator = OVRPlattScaling()

    with pytest.raises(ValueError):
        calibrator.fit(
            np.zeros((10, 1)),
            np.zeros(10),
        )

def test_invalid_regularization():
    with pytest.raises(ValueError):
        OVRPlattScaling(regularization=-1.0)


def test_invalid_tolerance():
    with pytest.raises(ValueError):
        OVRPlattScaling(tol=0.0)

def test_invalid_max_iter():
    with pytest.raises(ValueError):
        OVRPlattScaling(max_iter=0)


