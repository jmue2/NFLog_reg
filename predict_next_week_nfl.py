"""
predict_next_week_nfl.py
script to predict the outcomes of next week's NFL games using a logistic regression model
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from data_extraction import dynamic_window_ewma_epa, dynamic_window_ewma_turnover
from cross_validation import read_data

def build_model(df, features, target='home_team_win'):
    """
    builds and returns a logistic regression model

    :param df (pd.DataFrame): DataFrame containing training data
    :param features (list): list of feature column names used for training
    :param target (str): target column name. Default is 'home_team_win'

    :ret clf LogisticRegression: a trained logistic regression model
    """
    X = df[features].values
    y = df[target].values

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    return clf

def compute_latest_ewma_epa(df):
    """
    computes the latest EWMA EPA values for each team using a dynamic window

    :param df (pd.DataFrame): DataFrame containing EPA data for each game, with EPA metrics for both home and away teams

    :ret ewma_df (pd.DataFrame): DataFrame containing the latest EWMA EPA values for each team
    """
    metrics = ['epa_rushing_offense', 'epa_passing_offense',
               'epa_rushing_defense', 'epa_passing_defense']
    results = []

    teams = set(df['home_team']).union(set(df['away_team']))
    
    for team in teams:
        team_ewma = {'team': team}
        
        for metric in metrics:
            # Get all games where the team was either home or away
            team_home = df[df['home_team'] == team][['season', 'week', f'{metric}_home']].rename(columns={f'{metric}_home': 'epa'})
            team_away = df[df['away_team'] == team][['season', 'week', f'{metric}_away']].rename(columns={f'{metric}_away': 'epa'})
            
            # Combine home and away games
            team_df = pd.concat([team_home, team_away], ignore_index=True)
            team_df = team_df.sort_values(['season', 'week']).reset_index(drop=True)
            
            # Shift the EPA values to simulate "going into next week"
            team_df['epa_shifted'] = team_df['epa'].shift(1)
            team_df = team_df.dropna(subset=['epa_shifted'])
            
            # Apply the dynamic window EWMA function
            team_df['ewma'] = dynamic_window_ewma_epa(team_df)
            
            # Get the latest EWMA value (going into the next week)
            latest_ewma = team_df['ewma'].iloc[-1]
            
            key = f'{metric}_ewma_dynamic_window'
            team_ewma[key] = latest_ewma
        
        results.append(team_ewma)
    
    ewma_df = pd.DataFrame(results)
    
    return ewma_df

def compute_latest_ewma_turnover(df):
    """
    computes the latest EWMA turnover rate for each team using a dynamic window

    :param df (pd.DataFrame): DataFrame containing turnover data for each game, with turnover metrics for both home and away teams

    :ret ewma_df (pd.DataFrame): DataFrame containing the latest EWMA turnover rate for each team
    """
    metrics = ['turnover_offense', 'turnover_defense']
    results = []

    teams = set(df['home_team']).union(set(df['away_team']))
    
    for team in teams:
        team_ewma = {'team': team}
        
        for metric in metrics:
            # Get all games where the team was either home or away
            team_home = df[df['home_team'] == team][['season', 'week', f'{metric}_home']].rename(columns={f'{metric}_home': 'turnover'})
            team_away = df[df['away_team'] == team][['season', 'week', f'{metric}_away']].rename(columns={f'{metric}_away': 'turnover'})
            
            # Combine home and away games
            team_df = pd.concat([team_home, team_away], ignore_index=True)
            team_df = team_df.sort_values(['season', 'week']).reset_index(drop=True)
            
            # Shift the turnover values to simulate "going into next week"
            team_df['turnover_shifted'] = team_df['turnover'].shift(1)
            team_df = team_df.dropna(subset=['turnover_shifted'])
            
            # Apply the dynamic window EWMA function
            team_df['ewma'] = dynamic_window_ewma_turnover(team_df)
            
            # Get the latest EWMA value (going into the next week)
            latest_ewma = team_df['ewma'].iloc[-1]
            
            key = f'{metric}_ewma_dynamic_window'
            team_ewma[key] = latest_ewma
        
        results.append(team_ewma)
    
    ewma_df = pd.DataFrame(results)
    
    return ewma_df

def main():
    """
    main function to predict the outcomes of next week's NFL games.

    reads the EPA data, builds a logistic regression model, computes the latest EWMA values for each team,
    merges these values with the next week's matchups, and predicts the winner and win probabilities.

    the results are saved to a CSV file
    """
    df = read_data()

    # build model
    target = 'home_team_win'
    features = [col for col in df.columns if 'ewma_dynamic_window' in col and ('_home' in col or '_away' in col)]
    clf = build_model(df, features, target)

    # read next week's matchups (CSV with columns ['season', 'week', 'home_team', 'away_team'])
    matchups = pd.read_csv('next_week_matchups.csv')

    next_week_epa_data = compute_latest_ewma_epa(df)
    next_week_turnover_data = compute_latest_ewma_turnover(df)

    # home team features for EPA
    home_epa_features = next_week_epa_data.copy()
    home_epa_features = home_epa_features.rename(columns={'team': 'home_team'})
    home_epa_features_columns = {col: col + '_home' for col in home_epa_features.columns if col != 'home_team'}
    home_epa_features = home_epa_features.rename(columns=home_epa_features_columns)

    # away team features for EPA
    away_epa_features = next_week_epa_data.copy()
    away_epa_features = away_epa_features.rename(columns={'team': 'away_team'})
    away_epa_features_columns = {col: col + '_away' for col in away_epa_features.columns if col != 'away_team'}
    away_epa_features = away_epa_features.rename(columns=away_epa_features_columns)

    # home team features for turnover
    home_turnover_features = next_week_turnover_data.copy()
    home_turnover_features = home_turnover_features.rename(columns={'team': 'home_team'})
    home_turnover_features_columns = {col: col + '_home' for col in home_turnover_features.columns if col != 'home_team'}
    home_turnover_features = home_turnover_features.rename(columns=home_turnover_features_columns)

    # away team features for turnover
    away_turnover_features = next_week_turnover_data.copy()
    away_turnover_features = away_turnover_features.rename(columns={'team': 'away_team'})
    away_turnover_features_columns = {col: col + '_away' for col in away_turnover_features.columns if col != 'away_team'}
    away_turnover_features = away_turnover_features.rename(columns=away_turnover_features_columns)

    # merge the latest EWMA EPA and turnover values with the matchups for home and away teams
    matchups = matchups.merge(home_epa_features, on='home_team', how='left')
    matchups = matchups.merge(away_epa_features, on='away_team', how='left')
    matchups = matchups.merge(home_turnover_features, on='home_team', how='left')
    matchups = matchups.merge(away_turnover_features, on='away_team', how='left')

    # prepare the features for prediction
    prediction_features = [col for col in matchups.columns if '_ewma_dynamic_window' in col]

    if not prediction_features:
        raise ValueError("No prediction features found in matchups DataFrame. Please check the column names.")

    matchups = matchups.dropna(subset=prediction_features)

    # save data used to make predictions
    output_directory = os.path.join('output', datetime.now().strftime('%Y-%m-%d'))
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    matchups.to_csv(os.path.join(output_directory, 'next_week_data.csv'))

    # predict outcomes
    X_next_week = matchups[prediction_features].values
    predictions = clf.predict(X_next_week)
    probabilities = clf.predict_proba(X_next_week)[:, 1]  # Probability that home team wins

    # add predictions to the DataFrame
    matchups['predicted_winner'] = np.where(predictions == 1, matchups['home_team'], matchups['away_team'])
    matchups['win_probability'] = np.where(predictions == 1, probabilities, 1 - probabilities)

    # write predictions to CSV
    output_file = os.path.join(output_directory, 'next_week_predictions.csv')
    output_columns = ['season', 'week', 'home_team', 'away_team', 'predicted_winner', 'win_probability']
    matchups[output_columns].to_csv(output_file, index=False)

    print(f"Predictions saved to '{output_file}'.")

if __name__ == '__main__':
    main()
