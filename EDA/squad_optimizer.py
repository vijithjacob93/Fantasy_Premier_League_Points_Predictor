import pandas as pd
import xgboost as xgb
from pulp import LpProblem, LpVariable, lpSum, LpMaximize

# Load historical data
df = pd.read_csv('/Users/vijith.poovelil/Documents/sandbox/Fantasy-Premier-League/data/2023-24/gws/merged_gw.csv')
teams_df = pd.read_csv('/Users/vijith.poovelil/Documents/sandbox/Fantasy-Premier-League/data/2023-24/teams.csv')

# Sort by player and gameweek for rolling averages
df = df.sort_values(by=['name', 'gw'])

# Create rolling average features for historical data
rolling_features = ['total_points', 'goals_scored', 'assists', 'bps', 'ict_index']
for feature in rolling_features:
    df[f'{feature}_rolling_3'] = df.groupby('name')[feature].transform(lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))

# Add opponent difficulty and home/away advantage features to historical data
df = pd.merge(df, teams_df[['id', 'strength']], left_on='opponent_team', right_on='id', how='left')
df.rename(columns={'strength': 'opponent_strength'}, inplace=True)
df.drop('id', axis=1, inplace=True)

# Select features and target for model training
features = ['goals_scored', 'assists', 'minutes', 'goals_conceded', 'bonus', 'bps', 'ict_index', 'total_points_rolling_3', 'opponent_strength', 'was_home']
target = 'total_points'

# Drop rows with missing values (for initial rolling averages)
df.dropna(subset=features, inplace=True)

# Train XGBoost model on historical data
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_reg.fit(df[features], df[target])

# Load upcoming gameweek data
upcoming_gw = pd.read_csv('/Users/vijith.poovelil/Documents/sandbox/Fantasy-Premier-League/data/2024-25/gws/gw1.csv')

# Add opponent difficulty and home/away advantage features to upcoming gameweek data
upcoming_gw = pd.merge(upcoming_gw, teams_df[['id', 'strength']], left_on='opponent_team', right_on='id', how='left')
upcoming_gw.rename(columns={'strength': 'opponent_strength'}, inplace=True)
upcoming_gw.drop('id', axis=1, inplace=True)

# Map rolling average features from historical data to upcoming gameweek data
# Get the last available rolling average for each player from the historical data
last_gw_rolling_averages = df.groupby('name')[[f'{feature}_rolling_3' for feature in rolling_features]].last().reset_index()

# Merge these rolling averages into the upcoming_gw DataFrame
upcoming_gw = pd.merge(upcoming_gw, last_gw_rolling_averages, on='name', how='left')

# Handle players who might not have historical rolling average data (e.g., new players)
for feature in rolling_features:
    upcoming_gw[f'{feature}_rolling_3'] = upcoming_gw[f'{feature}_rolling_3'].fillna(0) # Fill NaN with 0 or a reasonable default

# Predict points for the upcoming gameweek
upcoming_gw['predicted_points'] = xgb_reg.predict(upcoming_gw[features])

# Optimize squad selection
prob = LpProblem("FPL Squad Optimization", LpMaximize)

players = list(upcoming_gw.index)
player_vars = LpVariable.dicts("Player", players, 0, 1, cat='Integer')

# Objective function
prob += lpSum([upcoming_gw['predicted_points'][i] * player_vars[i] for i in players])

# Constraints
prob += lpSum([upcoming_gw['value'][i] * player_vars[i] for i in players]) <= 1000  # Budget constraint
prob += lpSum([player_vars[i] for i in players]) == 15  # Total players

# Position constraints (example - adjust as per FPL rules)
prob += lpSum([player_vars[i] for i in players if upcoming_gw['position'][i] == 'GK']) == 2
prob += lpSum([player_vars[i] for i in players if upcoming_gw['position'][i] == 'DEF']) >= 5
prob += lpSum([player_vars[i] for i in players if upcoming_gw['position'][i] == 'DEF']) <= 5
prob += lpSum([player_vars[i] for i in players if upcoming_gw['position'][i] == 'MID']) >= 5
prob += lpSum([player_vars[i] for i in players if upcoming_gw['position'][i] == 'MID']) <= 5
prob += lpSum([player_vars[i] for i in players if upcoming_gw['position'][i] == 'FWD']) >= 3
prob += lpSum([player_vars[i] for i in players if upcoming_gw['position'][i] == 'FWD']) <= 3

# Solve the problem
prob.solve()

# Print the optimal squad
print("Optimal FPL Squad:")
for v in prob.variables():
    if v.varValue > 0:
        player_index = int(v.name.split('_')[1])
        print(f"{upcoming_gw.iloc[player_index]['name']} ({upcoming_gw.iloc[player_index]['position']}) - Predicted Points: {upcoming_gw.iloc[player_index]['predicted_points']:.2f}")