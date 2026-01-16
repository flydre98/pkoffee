
def test_compute_r2():
    from pkoffee import metrics
    compute_r2 = metrics.compute_r2
    import numpy as np

    # Test case 1: Perfect prediction
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([3, -0.5, 2, 7])
    r2 = compute_r2(y_true, y_pred)
    assert r2 == 1.0, f"Expected R² of 1.0, got {r2}"

    # Test case 2: No correlation
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([4, 3, 2, 1])
    r2 = compute_r2(y_true, y_pred)
    assert r2 < 0.0, f"Expected R² less than 0.0, got {r2}"

    print("All tests passed!")

def test_compute_rmse():
    from pkoffee import metrics
    compute_rmse = metrics.compute_rmse
    import numpy as np

    # Test case 1: Perfect prediction
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([3, -0.5, 2, 7])
    rmse = compute_rmse(y_true, y_pred)
    assert rmse == 0.0, f"Expected RMSE of 0.0, got {rmse}"

    # Test case 2: Some error
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([2, 2, 2, 2])
    rmse = compute_rmse(y_true, y_pred)
    expected_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    assert np.isclose(rmse, expected_rmse), f"Expected RMSE of {expected_rmse}, got {rmse}"

    print("All tests passed!")