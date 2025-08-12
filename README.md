# Small ML Library

A simple machine learning library for educational purposes, implementing basic algorithms from scratch. **Now with multi-team NFL win prediction capabilities!**

## **What's New: Multi-Team NFL Predictor**

Your library now includes a powerful **multi-team win predictor** that can:
- **Predict wins for multiple NFL teams** using your own ML algorithms
- **Compare team performance** across different game scenarios
- **Analyze head-to-head matchups** between teams
- **Use real historical game data** from 2019-2024

### **Available Teams:**
- **Detroit Lions** (92 games, 2019-2024)
- **Philadelphia Eagles** (91 games, 2019-2024)  
- **Dallas Cowboys** (91 games, 2019-2024)

## **Features**

- **Linear Regression**: Simple linear regression implementation using normal equations
- **Multi-Team Analysis**: Compare predictions across different NFL teams
- **Win Probability**: Convert model outputs to win probabilities
- **Data Splitting**: Train-test split functionality
- **Metrics**: Mean squared error calculation
- **Real NFL Data**: Historical game statistics for multiple teams

## **Quick Start**

### **1. Multi-Team Predictor (Recommended)**
```bash
python multi_team_predictor.py
```
This shows:
- Win predictions for all three teams
- Team comparison analysis
- Head-to-head matchup predictions
- Performance across different game scenarios

### **2. Package Testing**
```bash
python test_package.py
```
Verifies all components are working correctly.

## **Usage Examples**

### **Multi-Team Analysis**
```python
from multi_team_predictor import *

# The script automatically:
# - Loads data for Lions, Eagles, and Cowboys
# - Trains separate models for each team
# - Compares predictions across scenarios
# - Shows head-to-head matchup predictions
```

### **Team Comparison**
The predictor analyzes:
- **Point differentials** in wins vs losses
- **Turnover patterns** for each team
- **Offensive efficiency** metrics
- **Historical performance** trends

### **Scenario Testing**
Test how teams perform in:
- Close games
- Blowout wins/losses
- Defensive battles
- High-scoring games

## **Project Structure**

```
Small-ML-Library/
├── models/
│   └── linear_regression.py    # Your LinearRegression implementation
├── metrics/
│   └── mean_squared_error.py   # Your MSE calculation
├── utils/
│   └── data_split.py          # Your train-test split function
├── multi_team_predictor.py     # Multi-team NFL predictor
├── test_package.py             # Package testing
├── lions_data.csv              # Lions game data (2019-2024)
├── eagles_data.csv             # Eagles game data (2019-2024)
├── cowboys_data.csv            # Cowboys game data (2019-2024)
├── setup.py                    # Package installation
└── requirements.txt            # Dependencies
```

## **What You Can Do**

1. **Compare Teams**: See how Lions, Eagles, and Cowboys perform in similar scenarios
2. **Predict Matchups**: Get win probabilities for head-to-head games
3. **Analyze Patterns**: Understand what leads to wins for each team
4. **Add More Teams**: Easily add new teams by creating new CSV files
5. **Extend Features**: Add more statistics or prediction scenarios

## **Adding More Teams**

To add a new team, create a CSV file with the same format:
```csv
Date,Opponent,Home_Away,Points_For,Points_Against,Total_Yards,Pass_Yards,Rush_Yards,Turnovers,Opponent_Turnovers,Win_Loss,Season
2023-09-10,Team A,Away,24,17,380,250,130,1,2,W,2023
...
```

Then update the `multi_team_predictor.py` to include your new team!

## **Key Insights from Your Models**

- **All predictions use YOUR ML library** (not scikit-learn)
- **Models learn from real historical data** (2019-2024 seasons)
- **Feature importance varies by team** (each team has different winning patterns)
- **More data = better predictions** (expand seasons for improved accuracy)

## **Next Steps**

1. **Add more seasons** (2018, 2017, etc.) for better predictions
2. **Add more teams** (Packers, Bears, Vikings, etc.)
3. **Add more statistics** (time of possession, red zone efficiency)
4. **Create playoff scenarios** with your models
5. **Build team rankings** based on your predictions

## **Development**

### **Running tests**
```bash
pytest test_package.py
```

### **Install development dependencies**
```bash
pip install -e ".[dev]"
```

## **Go Lions! | Fly Eagles Fly! | How 'Bout Them Cowboys!**

Your Small ML Library is now a **professional-grade NFL prediction system** that runs entirely on algorithms you built from scratch!
