#!/usr/bin/env python3
"""
Multi-Team Win Predictor - Compare predictions across different teams
"""

import numpy as np
import pandas as pd
from models.linear_regression import LinearRegression
from metrics.mean_squared_error import mean_squared_error
from utils.data_split import train_test_split

def load_team_data(team_name, filename):
    """Load and prepare team game data"""
    print(f"Loading {team_name} Game Data...")
    
    try:
        data = pd.read_csv(filename)
        print(f"Loaded {len(data)} games from {data['Season'].min()}-{data['Season'].max()}")
        
        # Convert Win/Loss to numeric (1 for Win, 0 for Loss)
        data['Win_Numeric'] = (data['Win_Loss'] == 'W').astype(int)
        
        # Show some stats
        wins = data['Win_Numeric'].sum()
        total_games = len(data)
        win_rate = wins / total_games * 100
        
        print(f"{team_name} Record: {wins}-{total_games-wins} ({win_rate:.1f}% win rate)")
        
        return data
    except FileNotFoundError:
        print(f"ERROR: {filename} not found!")
        return None

def prepare_features(data):
    """Prepare features for win prediction"""
    # Create intuitive features
    data['Point_Diff'] = data['Points_For'] - data['Points_Against']
    data['Turnover_Diff'] = data['Opponent_Turnovers'] - data['Turnovers']
    data['Efficiency'] = data['Points_For'] / (data['Total_Yards'] + 1)
    
    # Select features
    feature_columns = ['Point_Diff', 'Turnover_Diff', 'Efficiency']
    
    # Prepare X (features) and y (target)
    X = data[feature_columns].values
    y = data['Win_Numeric'].values
    
    # Remove any rows with NaN values
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]
    
    return X, y, feature_columns

def train_team_model(X, y, team_name):
    """Train a model for a specific team"""
    print(f"\nTraining {team_name} Win Prediction Model...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"Training on {len(X_train)} games, testing on {len(X_test)} games")
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Evaluate
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    
    return model, X_test, y_test, test_pred

def predict_win_probability(model, features):
    """Convert model output to win probability"""
    raw_prediction = model.predict(features)[0]
    probability = 1 / (1 + np.exp(-raw_prediction))
    return probability

def compare_all_team_predictions(lions_model, eagles_model, cowboys_model, feature_names):
    """Compare how all three teams would perform in the same scenarios"""
    print("\nComparing All Team Predictions in Same Scenarios...")
    
    scenarios = [
        ("Close Win", [4, 1, 0.08]),
        ("Blowout Win", [21, 2, 0.09]),
        ("Close Loss", [-3, -1, 0.06]),
        ("Defensive Battle", [7, 1, 0.05]),
        ("High Scoring Game", [14, 0, 0.10])
    ]
    
    print(f"{'Scenario':<20} | {'Lions':<15} | {'Eagles':<15} | {'Cowboys':<15}")
    print("-" * 75)
    
    for name, stats in scenarios:
        features = np.array([stats])
        
        lions_prob = predict_win_probability(lions_model, features)
        eagles_prob = predict_win_probability(eagles_model, features)
        cowboys_prob = predict_win_probability(cowboys_model, features)
        
        lions_result = "WIN" if lions_prob > 0.6 else "LOSS" if lions_prob < 0.4 else "CLOSE"
        eagles_result = "WIN" if eagles_prob > 0.6 else "LOSS" if eagles_prob < 0.4 else "CLOSE"
        cowboys_result = "WIN" if cowboys_prob > 0.6 else "LOSS" if cowboys_prob < 0.4 else "CLOSE"
        
        print(f"{name:<20} | {lions_result:<15} | {eagles_result:<15} | {cowboys_result:<15}")
        print(f"{'':<20} | {lions_prob:.1%}{'':<12} | {eagles_prob:.1%}{'':<12} | {cowboys_prob:.1%}{'':<12}")

def analyze_all_team_differences(lions_data, eagles_data, cowboys_data):
    """Analyze key differences between all three teams"""
    print("\nAll Team Comparison Analysis...")
    
    lions_wins = lions_data[lions_data['Win_Loss'] == 'W']
    lions_losses = lions_data[lions_data['Win_Loss'] == 'L']
    eagles_wins = eagles_data[eagles_data['Win_Loss'] == 'W']
    eagles_losses = eagles_data[eagles_data['Win_Loss'] == 'L']
    cowboys_wins = cowboys_data[cowboys_data['Win_Loss'] == 'W']
    cowboys_losses = cowboys_data[cowboys_data['Win_Loss'] == 'L']
    
    print("Average Stats Comparison:")
    print(f"{'Stat':<25} | {'Lions (W)':<12} | {'Eagles (W)':<12} | {'Cowboys (W)':<12}")
    print("-" * 80)
    
    stats = [
        ('Points For', 'Points_For'),
        ('Points Against', 'Points_Against'),
        ('Point Differential', 'Point_Diff'),
        ('Turnover Differential', 'Turnover_Diff')
    ]
    
    for stat_name, col in stats:
        if col == 'Point_Diff':
            lions_data['Point_Diff'] = lions_data['Points_For'] - lions_data['Points_Against']
            eagles_data['Point_Diff'] = eagles_data['Points_For'] - eagles_data['Points_Against']
            cowboys_data['Point_Diff'] = cowboys_data['Points_For'] - cowboys_data['Points_Against']
        elif col == 'Turnover_Diff':
            lions_data['Turnover_Diff'] = lions_data['Opponent_Turnovers'] - lions_data['Turnovers']
            eagles_data['Turnover_Diff'] = eagles_data['Opponent_Turnovers'] - eagles_data['Turnovers']
            cowboys_data['Turnover_Diff'] = cowboys_data['Opponent_Turnovers'] - cowboys_data['Turnovers']
        
        lions_w_val = lions_wins[col].mean() if col in lions_wins.columns else 0
        eagles_w_val = eagles_wins[col].mean() if col in eagles_wins.columns else 0
        cowboys_w_val = cowboys_wins[col].mean() if col in cowboys_wins.columns else 0
        
        print(f"{stat_name:<25} | {lions_w_val:<12.1f} | {eagles_w_val:<12.1f} | {cowboys_w_val:<12.1f}")

def predict_all_head_to_head(lions_model, eagles_model, cowboys_model, feature_names):
    """Predict head-to-head matchups between all teams"""
    print("\nAll Head-to-Head Matchup Predictions...")
    
    # Lions vs Eagles
    print("Lions vs Eagles:")
    lions_scenario = [3, 1, 0.08]
    eagles_scenario = [-3, -1, 0.08]
    
    lions_win_prob = predict_win_probability(lions_model, np.array([lions_scenario]))
    eagles_win_prob = predict_win_probability(eagles_model, np.array([eagles_scenario]))
    
    if lions_win_prob > eagles_win_prob:
        winner = "Lions"
        margin = lions_win_prob - eagles_win_prob
    else:
        winner = "Eagles"
        margin = eagles_win_prob - lions_win_prob
    
    print(f"  Lions: {lions_win_prob:.1%} | Eagles: {eagles_win_prob:.1%} | Winner: {winner} (by {margin:.1%})")
    
    # Lions vs Cowboys
    print("Lions vs Cowboys:")
    cowboys_scenario = [-3, -1, 0.08]
    cowboys_win_prob = predict_win_probability(cowboys_model, np.array([cowboys_scenario]))
    
    if lions_win_prob > cowboys_win_prob:
        winner = "Lions"
        margin = lions_win_prob - cowboys_win_prob
    else:
        winner = "Cowboys"
        margin = cowboys_win_prob - lions_win_prob
    
    print(f"  Lions: {lions_win_prob:.1%} | Cowboys: {cowboys_win_prob:.1%} | Winner: {winner} (by {margin:.1%})")
    
    # Eagles vs Cowboys
    print("Eagles vs Cowboys:")
    if eagles_win_prob > cowboys_win_prob:
        winner = "Eagles"
        margin = eagles_win_prob - cowboys_win_prob
    else:
        winner = "Cowboys"
        margin = cowboys_win_prob - eagles_win_prob
    
    print(f"  Eagles: {eagles_win_prob:.1%} | Cowboys: {cowboys_win_prob:.1%} | Winner: {winner} (by {margin:.1%})")

def main():
    """Main function to run the multi-team predictor"""
    print("MULTI-TEAM WIN PREDICTOR")
    print("=" * 60)
    
    # Load all teams' data
    lions_data = load_team_data("Detroit Lions", "lions_data.csv")
    eagles_data = load_team_data("Philadelphia Eagles", "eagles_data.csv")
    cowboys_data = load_team_data("Dallas Cowboys", "cowboys_data.csv")
    
    if lions_data is None or eagles_data is None or cowboys_data is None:
        print("ERROR: Cannot proceed without all datasets!")
        return
    
    # Prepare features for all teams
    lions_X, lions_y, feature_names = prepare_features(lions_data)
    eagles_X, eagles_y, feature_names = prepare_features(eagles_data)
    cowboys_X, cowboys_y, feature_names = prepare_features(cowboys_data)
    
    # Train models for all teams
    lions_model, _, _, _ = train_team_model(lions_X, lions_y, "Lions")
    eagles_model, _, _, _ = train_team_model(eagles_X, eagles_y, "Eagles")
    cowboys_model, _, _, _ = train_team_model(cowboys_X, cowboys_y, "Cowboys")
    
    # Compare predictions across all teams
    compare_all_team_predictions(lions_model, eagles_model, cowboys_model, feature_names)
    
    # Analyze team differences
    analyze_all_team_differences(lions_data, eagles_data, cowboys_data)
    
    # Predict head-to-head matchups
    predict_all_head_to_head(lions_model, eagles_model, cowboys_model, feature_names)
    
    print("\n" + "=" * 60)
    print("Key Insights:")
    print("• All models use YOUR ML library (not scikit-learn)")
    print("• Compare how three teams perform in similar scenarios")
    print("• See which team is better at what types of games")
    print("• Predict head-to-head matchups between all teams!")
    
    print("\nGo Lions! | Fly Eagles Fly! | How 'Bout Them Cowboys!")

if __name__ == "__main__":
    main()
