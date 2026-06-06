
# tests/test_predict.py
from app.predict import predict_rock_class

def test_prediction_returns_integer():
    result = predict_rock_class(15.0, 23.0, 0.30, 2.15)
    assert isinstance(result, int), 'Prediction must be an integer'

def test_prediction_in_valid_range():
    result = predict_rock_class(15.0, 23.0, 0.30, 2.15)
    assert 1 <= result <= 10, f'Class must be 1-10, got {result}'

def test_different_inputs_return_valid_classes():
    for gr, res, phie, rhob in [
        (20.0, 100.0, 0.25, 2.20),
        (80.0, 5.0,   0.10, 2.55),
        (14.0, 23.0,  0.33, 2.09),
    ]:
        result = predict_rock_class(gr, res, phie, rhob)
        assert 1 <= result <= 10
