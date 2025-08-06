import pandas as pd

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

print(df[['name', 'gw', 'opponent_team', 'opponent_strength', 'was_home']].head(10))
