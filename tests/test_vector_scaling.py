import numpy as np
import pytest

from segmentation_evaluation.calibration import (
    VectorScaling,
    vector_scaling,
)


@pytest.fixture
def calibration_data():
    rng = np.random.default_rng(42)

    n_samples = 300
    n_classes = 3

    y_true = rng.integers(0, n_classes, size=n_samples)

    logits = rng.normal(size=(n_samples, n_classes))
    logits[np.arange(n_samples), y_true] += 1.0

    return logits, y_true


def test_initial_state():
    calibrator = VectorScaling()

    assert calibrator.is_fitted_ is False
    assert calibrator.scales_ is None
    assert calibrator.biases_ is None


def test_predict_before_fit_raises():
    calibrator = VectorScaling()

    with pytest.raises(RuntimeError):
        calibrator.predict_proba(np.zeros((10, 3)))


def test_fit_returns_self(calibration_data):
    logits, y_true = calibration_data

    calibrator = VectorScaling()
    result = calibrator.fit(logits, y_true)

    assert result is calibrator
    assert calibrator.is_fitted_


def test_fitted_parameters_have_correct_shape(calibration_data):
    logits, y_true = calibration_data

    calibrator = VectorScaling().fit(logits, y_true)

    assert calibrator.scales_.shape == (3,)
    assert calibrator.biases_.shape == (3,)
    assert isinstance(calibrator.n_iter_, int)
    assert np.isfinite(calibrator.nll_)
    assert np.isfinite(calibrator.objective_)


def test_predict_proba_shape(calibration_data):
    logits, y_true = calibration_data

    calibrator = VectorScaling().fit(logits, y_true)
    probabilities = calibrator.predict_proba(logits)

    assert probabilities.shape == logits.shape


def test_probabilities_are_valid(calibration_data):
    logits, y_true = calibration_data

    calibrator = VectorScaling().fit(logits, y_true)
    probabilities = calibrator.predict_proba(logits)

    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)

    np.testing.assert_allclose(
        probabilities.sum(axis=-1),
        1.0,
        atol=1e-12,
    )


def test_segmentation_shape():
    rng = np.random.default_rng(10)

    logits = rng.normal(size=(4, 20, 30, 3))
    y_true = rng.integers(0, 3, size=(4, 20, 30))

    calibrator = VectorScaling().fit(logits, y_true)
    probabilities = calibrator.predict_proba(logits)

    assert probabilities.shape == logits.shape

    np.testing.assert_allclose(
        probabilities.sum(axis=-1),
        1.0,
        atol=1e-12,
    )


def test_arbitrary_class_labels():
    rng = np.random.default_rng(11)

    logits = rng.normal(size=(200, 3))
    y_true = rng.choice([1, 3, 9], size=200)

    calibrator = VectorScaling().fit(
        logits,
        y_true,
        labels=[1, 3, 9],
    )

    assert calibrator.classes_.tolist() == [1, 3, 9]

    probabilities = calibrator.predict_proba(logits)

    assert probabilities.shape == (200, 3)


def test_ignore_index():
    rng = np.random.default_rng(12)

    logits = rng.normal(size=(100, 3))
    y_true = rng.integers(0, 3, size=100)

    y_true[-10:] = 255

    calibrator = VectorScaling().fit(
        logits,
        y_true,
        ignore_index=255,
    )

    probabilities = calibrator.predict_proba(logits)

    assert probabilities.shape == (100, 3)


def test_convenience_function(calibration_data):
    logits, y_true = calibration_data

    probabilities = vector_scaling(logits, y_true)

    assert probabilities.shape == logits.shape

    np.testing.assert_allclose(
        probabilities.sum(axis=-1),
        1.0,
        atol=1e-12,
    )


def test_convenience_function_returns_calibrator(calibration_data):
    logits, y_true = calibration_data

    probabilities, calibrator = vector_scaling(
        logits,
        y_true,
        return_calibrator=True,
    )

    assert isinstance(calibrator, VectorScaling)
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

    probabilities, calibrator = vector_scaling(
        logits_val,
        y_val,
        logits_test=logits_test,
        return_calibrator=True,
    )

    assert probabilities.shape == logits_test.shape
    assert calibrator.is_fitted_


def test_invalid_logits_shape():
    with pytest.raises(ValueError):
        VectorScaling().fit(
            np.zeros((10,)),
            np.zeros(10),
        )


def test_single_class_raises():
    with pytest.raises(ValueError, match="at least two classes"):
        VectorScaling().fit(
            np.zeros((10, 1)),
            np.zeros(10),
        )


def test_nonfinite_logits():
    logits = np.zeros((20, 3))
    logits[0, 0] = np.nan

    y_true = np.arange(20) % 3

    with pytest.raises(ValueError, match="finite"):
        VectorScaling().fit(logits, y_true)


def test_invalid_target_shape():
    logits = np.zeros((20, 3))
    y_true = np.zeros(10)

    with pytest.raises(ValueError, match="shape"):
        VectorScaling().fit(logits, y_true)


def test_all_observations_ignored():
    logits = np.zeros((20, 3))
    y_true = np.full(20, 255)

    with pytest.raises(ValueError, match="No valid observations"):
        VectorScaling().fit(
            logits,
            y_true,
            ignore_index=255,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"regularization": -1.0},
        {"learning_rate": 0.0},
        {"tol": 0.0},
        {"max_iter": 0},
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        VectorScaling(**kwargs)