# Small ML Library

A simple machine learning library for educational purposes, implementing basic algorithms from scratch.

## Features

- **Linear Regression**: Simple linear regression implementation using normal equations
- **Metrics**: Mean squared error calculation
- **Utilities**: Train-test split functionality

## Installation

### From source
```bash
git clone <repository-url>
cd Small-ML-Library
pip install -e .
```

### Install dependencies only
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from small_ml_library import LinearRegression, mean_squared_error, train_test_split
import numpy as np

# Generate some sample data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X.flatten() + 1 + np.random.randn(100) * 0.5

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.4f}")
print(f"Model weights: {model.weights}")
print(f"Model bias: {model.bias}")
```

## Usage Examples

### Linear Regression

```python
from small_ml_library import LinearRegression

# Create model
model = LinearRegression()

# Train on data
X = [[1], [2], [3], [4], [5]]
y = [2, 4, 6, 8, 10]
model.fit(X, y)

# Make predictions
predictions = model.predict([[6], [7]])
print(predictions)  # [12, 14]
```

### Using the CSV data

```python
import pandas as pd
from small_ml_library import LinearRegression, mean_squared_error, train_test_split

# Load the moonDB.csv data
data = pd.read_csv('moonDB.csv')

# Assuming the CSV has columns 'feature' and 'target'
X = data[['feature']].values
y = data['target'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"MSE: {mse}")
```

## Development

### Running tests
```bash
pytest test/
```

### Install development dependencies
```bash
pip install -e ".[dev]"
```

## License

MIT License
# ML-football-predictor
