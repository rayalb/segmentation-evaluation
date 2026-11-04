import numpy as np 
import pytest 

from segmentation_evaluation.calibration import(
    TemperatureScaling,
    temperature_scaling,
)


def test_inital_state():
    calibrator = TemperatureScaling()
    assert calibrator.is_fitted_ is False
    assert calibrator.temperature_ is None
    assert calibrator.classes_ is None

def test_fit_returns_calibrator():
    logits = np.array([
        [2.0, 1.0], [1.0, 2.0], [2.0, 1.0], [1.0, 2.0]
    ])
    y_true = np.array([0, 1, 0, 1])
    calibrator = TemperatureScaling()
    result = calibrator.fit(logits, y_true)
    assert result is calibrator
    assert calibrator.is_fitted_ is True
    assert calibrator.temperature_ is not None

def test_temperature_parameter():
    logits = np.array([
        [2.0, 1.0], [1.0, 2.0], [2.0, 1.0], [1.0, 2.0]
    ])
    y_true = np.array([0, 1, 0, 1])

    bounds = (0.1, 10.0)
    calibrator = TemperatureScaling(temperature_bounds = bounds)
    calibrator.fit(logits, y_true)

    assert bounds[0] <= calibrator.temperature_ <= bounds[1]

def test_predict_proba_before_fit():
    calibrator = TemperatureScaling()
    logits = np.array([[1.0, 2.0]])

    with pytest.raises(RuntimeError):
        calibrator.predict_proba(logits)

def test_predict_proba_shape_and_normalization():
    logits_val = np.array([
        [2.0, 1.0, 0.5], [0.5, 2.0, 1.0],
        [1.0, 0.5, 2.0], [1.0, 1.0, 0.5]
    ])
    y_val = np.array([0, 1, 2, 0])
    logits_test = np.array([
        [1.0, 2.0, 0.5], [0.5, 1.0, 2.0]
    ])
    calibrator = TemperatureScaling()
    calibrator.fit(logits_val, y_val)
    calib_probs = calibrator.predict_proba(logits_test)

    assert calib_probs.shape == (2, 3)
    assert np.all(calib_probs >= 0.0)
    assert np.all(calib_probs <= 1.0)
    assert np.allclose(calib_probs.sum(axis = -1), 1.0)

def test_temperature_scaling_preserves_argmax():
    logits_val = np.array([
        [2.0, 1.0], [1.0, 2.0], [2.0, 1.0], [1.0, 2.0]
    ])
    y_val = np.array([0, 1, 0, 1])

    logits_test = np.array([
        [1.0, 2.0], [2.0, 1.0], [0.5, 1.5]
    ])
    calibrator = TemperatureScaling()
    calibrator.fit(logits_val, y_val)

    probs = calibrator.predict_proba(logits_test)
    assert np.array_equal(
        np.argmax(logits_test, axis = -1),
        np.argmax(probs, axis = -1)
    )

def test_segmentation_input():
    rng = np.random.default_rng(42)

    logits_val = rng.normal(size=(2, 4, 5, 3))
    y_val = rng.integers(0, 3, size=(2, 4, 5))

    logits_test = rng.normal(size=(1, 6, 7, 3))

    calibrator = TemperatureScaling()
    calibrator.fit(logits_val, y_val)

    probabilities = calibrator.predict_proba(logits_test)

    assert probabilities.shape == (1, 6, 7, 3)
    assert np.allclose(probabilities.sum(axis=-1), 1.0)


def test_ignore_index():
    logits = np.array([
        [2.0, 1.0],
        [1.0, 2.0],
        [2.0, 1.0],
        [1.0, 2.0],
    ])

    y_true = np.array([0, 1, 255, 255])

    calibrator = TemperatureScaling()

    calibrator.fit(
        logits,
        y_true,
        ignore_index=255,
    )

    assert calibrator.is_fitted_
    assert calibrator.temperature_ is not None


def test_arbitrary_class_labels():
    logits = np.array([
        [2.0, 1.0, 0.5],
        [0.5, 2.0, 1.0],
        [1.0, 0.5, 2.0],
    ])

    y_true = np.array([10, 20, 30])

    calibrator = TemperatureScaling()

    calibrator.fit(
        logits,
        y_true,
        labels=[10, 20, 30],
    )

    assert np.array_equal(
        calibrator.classes_,
        np.array([10, 20, 30]),
    )

def test_convenience_function():
    logits_val = np.array([
        [2.0, 1.0],
        [1.0, 2.0],
        [2.0, 1.0],
        [1.0, 2.0],
    ])

    y_val = np.array([0, 1, 0, 1])

    logits_test = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
    ])

    probabilities = temperature_scaling(
        logits_val,
        y_val,
        logits_test=logits_test,
    )

    assert probabilities.shape == (2, 2)
    assert np.allclose(probabilities.sum(axis=-1), 1.0)


def test_convenience_function_returns_calibrator():
    logits_val = np.array([
        [2.0, 1.0],
        [1.0, 2.0],
        [2.0, 1.0],
        [1.0, 2.0],
    ])

    y_val = np.array([0, 1, 0, 1])

    probabilities, calibrator = temperature_scaling(
        logits_val,
        y_val,
        return_calibrator=True,
    )

    assert isinstance(calibrator, TemperatureScaling)
    assert calibrator.is_fitted_
    assert probabilities.shape == logits_val.shape


def test_fit_transform():
    logits = np.array([
        [2.0, 1.0],
        [1.0, 2.0],
        [2.0, 1.0],
        [1.0, 2.0],
    ])

    y_true = np.array([0, 1, 0, 1])

    calibrator = TemperatureScaling()

    probabilities = calibrator.fit_transform(logits, y_true)

    assert calibrator.is_fitted_
    assert probabilities.shape == logits.shape
    assert np.allclose(probabilities.sum(axis=-1), 1.0)


def test_invalid_logits_shape():
    calibrator = TemperatureScaling()

    logits = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
    ])

    y_true = np.array([0])

    with pytest.raises(ValueError):
        calibrator.fit(logits, y_true)


def test_non_finite_logits():
    calibrator = TemperatureScaling()

    logits = np.array([
        [1.0, np.nan],
        [2.0, 1.0],
    ])

    y_true = np.array([0, 1])

    with pytest.raises(ValueError):
        calibrator.fit(logits, y_true)


def test_invalid_temperature_bounds():
    with pytest.raises(ValueError):
        TemperatureScaling(temperature_bounds=(0.0, 10.0))

    with pytest.raises(ValueError):
        TemperatureScaling(temperature_bounds=(10.0, 1.0))


def test_invalid_tolerance():
    with pytest.raises(ValueError):
        TemperatureScaling(tol=0.0)


def test_invalid_max_iter():
    with pytest.raises(ValueError):
        TemperatureScaling(max_iter=0)


def test_all_observations_ignored():
    logits = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
    ])

    y_true = np.array([255, 255])

    calibrator = TemperatureScaling()

    with pytest.raises(ValueError):
        calibrator.fit(
            logits,
            y_true,
            ignore_index=255,
        )

    