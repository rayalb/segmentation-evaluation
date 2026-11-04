import numpy as np
import pytest

from segmentation_evaluation.calibration import BaseCalibrator


class DummyCalibrator(BaseCalibrator):
    
    def fit(self, logits, y_true, **fit_params):
        self.is_fitted_ = True
        return self

    def predict_proba(self, logits):
        self._check_is_fitted()

        logits = np.asarray(logits, dtype = np.float64)
        exp_logits = np.exp(logits - np.max(logits, axis = -1, keepdims = True))
        return exp_logits/np.sum(exp_logits, axis = -1, keepdims = True)

def test_base_calibrator_is_abstract():
    with pytest.raises(TypeError):
        BaseCalibrator()

def test_initial_state():
    calibrator = DummyCalibrator()
    assert calibrator.is_fitted_ is False

def test_predict_proba_before_fit():
    calibrator = DummyCalibrator()
    logits = np.array([[1.0, 2.0]])
    with pytest.raises(RuntimeError):
        calibrator.predict_proba(logits)

def test_fit_returns_calibrator():
    calibrator = DummyCalibrator()
    logits = np.array([[1.0, 2.0]])
    y_true = np.array([1])
    result = calibrator.fit(logits, y_true)
    assert result is calibrator
    assert calibrator.is_fitted_ is True

def test_predict_proba_after_fit():
    calibrator = DummyCalibrator()
    logits = np.array([
        [1.0, 2.0], [2.0, 1.0]
    ])
    y_true = np.array([1, 0])
    calibrator.fit(logits, y_true)

    probabilities = calibrator.predict_proba(logits)
    assert probabilities.shape == (2, 2)
    assert np.allclose(probabilities.sum(axis = -1), 1.0)
    assert np.all(probabilities >= 0.0)

def test_fit_transform():
    calibrator = DummyCalibrator()
    logits = np.array([
        [1.0, 2.0], [2.0, 1.0]
    ])
    y_true = np.array([1, 0])
    probabilities = calibrator.fit_transform(logits, y_true)
    assert calibrator.is_fitted_ is True
    assert probabilities.shape == (2, 2)
    assert np.allclose(probabilities.sum(axis = -1), 1.0)
    assert np.all(probabilities >= 0.0)
