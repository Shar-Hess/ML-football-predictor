#!/usr/bin/env python3
"""
Simple test script for the Small ML Library
"""

import numpy as np
from models.linear_regression import LinearRegression
from metrics.mean_squared_error import mean_squared_error
from utils.data_split import train_test_split

def test_linear_regression():
    """Test linear regression with simple data"""
    print("Testing Linear Regression...")
    
    # Simple linear relationship: y = 2x + 1
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([3, 5, 7, 9, 11])
    
    # Create and train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Test predictions
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    
    print(f"  Weights: {model.weights}")
    print(f"  Bias: {model.bias:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  Predictions: {predictions}")
    
    # Test on new data
    new_X = np.array([[0], [6]])
    new_predictions = model.predict(new_X)
    print(f"  New predictions: {new_predictions}")
    
    return mse < 0.1  # Should be very low for perfect linear data

def test_data_split():
    """Test train-test split functionality"""
    print("\nTesting Data Split...")
    
    X = np.arange(20).reshape(-1, 1)
    y = np.arange(20)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"  Original size: {len(X)}")
    print(f"  Training size: {len(X_train)}")
    print(f"  Test size: {len(X_test)}")
    
    # Check that sizes add up correctly
    total_split = len(X_train) + len(X_test)
    correct_split = total_split == len(X)
    
    print(f"  Split sizes correct: {correct_split}")
    
    return correct_split

def test_mean_squared_error():
    """Test MSE calculation"""
    print("\nTesting Mean Squared Error...")
    
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    
    mse = mean_squared_error(y_true, y_pred)
    expected_mse = np.mean((y_true - y_pred) ** 2)
    
    print(f"  Calculated MSE: {mse:.4f}")
    print(f"  Expected MSE: {expected_mse:.4f}")
    print(f"  MSE calculation correct: {abs(mse - expected_mse) < 1e-10}")
    
    return abs(mse - expected_mse) < 1e-10

def main():
    """Run all tests"""
    print("Small ML Library - Package Tests")
    print("=" * 40)
    
    tests = [
        test_linear_regression,
        test_data_split,
        test_mean_squared_error
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("  ✓ PASSED")
            else:
                print("  ✗ FAILED")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
    
    print("\n" + "=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed.")

if __name__ == "__main__":
    main()
