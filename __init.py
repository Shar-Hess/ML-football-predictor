"""
Small ML Library - A simple machine learning library for educational purposes
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Import main components for easy access
from .models.linear_regression import LinearRegression
from .metrics.mean_squared_error import mean_squared_error
from .utils.data_split import train_test_split

__all__ = [
    "LinearRegression",
    "mean_squared_error", 
    "train_test_split"
]
