# Small ML Library - Usage Guide

## Quick Start

Your Python package is now fully set up and ready to use! Here are the different ways you can test it:

### 1. Run the Demo Script
```bash
python demo.py
```
This shows how to use the library with customizable synthetic data and analyzes your CSV file.

### 2. Run the Test Script
```bash
python test_package.py
```
This verifies that all components are working correctly.

### 3. Run the Example Script
```bash
python example_usage.py
```
This demonstrates both synthetic data and CSV data usage.

## Using the Package in Your Own Code

### Basic Import
```python
from models.linear_regression import LinearRegression
from metrics.mean_squared_error import mean_squared_error
from utils.data_split import train_test_split
```

### Simple Example
```python
import numpy as np

# Create some data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, predictions)
print(f"MSE: {mse}")
```

## Package Structure

```
Small-ML-Library/
├── models/
│   └── linear_regression.py    # Linear regression implementation
├── metrics/
│   └── mean_squared_error.py   # MSE calculation
├── utils/
│   └── data_split.py          # Train-test split functionality
├── test/                       # Test files
├── demo.py                     # Interactive demo
├── example_usage.py            # Usage examples
├── test_package.py             # Package tests
├── setup.py                    # Package installation
└── requirements.txt            # Dependencies
```

## What You Can Do

1. **Linear Regression**: Train models on your data
2. **Data Splitting**: Split data into training and test sets
3. **Evaluation**: Calculate mean squared error
4. **Customization**: Modify parameters and test different scenarios

## Next Steps

1. **Modify the demo**: Change parameters in `demo.py` to test different data
2. **Add your data**: Replace the synthetic data with your own CSV or numpy arrays
3. **Extend the library**: Add new models, metrics, or utilities
4. **Run experiments**: Test different data sizes, noise levels, and relationships

## Troubleshooting

- If you get import errors, make sure you're in the right directory
- The package is installed in development mode, so changes to the code will be reflected immediately
- All dependencies (numpy, pandas, scikit-learn) are automatically installed

Your package is ready to use! 🎉
