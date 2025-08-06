import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('/Users/vijith.poovelil/Documents/sandbox/Fantasy-Premier-League/data/2023-24/gws/merged_gw.csv')
teams_df = pd.read_csv('/Users/vijith.poovelil/Documents/sandbox/Fantasy-Premier-League/data/2023-24/teams.csv')

# Sort by player and gameweek
df = df.sort_values(by=['name', 'gw'])

# Create rolling average features
rolling_features = ['total_points', 'goals_scored', 'assists', 'bps', 'ict_index']
for feature in rolling_features:
    df[f'{feature}_rolling_3'] = df.groupby('name')[feature].transform(lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))

# Add opponent difficulty and home/away advantage features
df = pd.merge(df, teams_df[['id', 'strength']], left_on='opponent_team', right_on='id', how='left')
df.rename(columns={'strength': 'opponent_strength'}, inplace=True)
df.drop('id', axis=1, inplace=True)

# Select features and target
features = ['goals_scored', 'assists', 'minutes', 'goals_conceded', 'bonus', 'bps', 'ict_index', 'total_points_rolling_3', 'opponent_strength', 'was_home']
target = 'total_points'

# Drop rows with missing values
df.dropna(subset=features, inplace=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Train XGBoost model
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_reg.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = xgb_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse}')
