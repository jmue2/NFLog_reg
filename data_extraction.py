"""
data_extraction.py
script to pull and process NFL play-by-play (pbp) data using nfl_data_py
"""

import nfl_data_py as nfl
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

def import_latest_pbp_from(start_season):
    """
    imports NFL play-by-play data starting from the specified season up to the current year
    
    :param start_season (int): The first season to include in the imported data
    
    :ret data: pd.DataFrame containing play-by-play data from the specified seasons
    """
    current_year = datetime.now().year
    data = nfl.import_pbp_data(years=range(start_season, current_year + 1), downcast=True, cache=False)

    data = data.fillna(0)
    return data

def extract_schedule(data):
    """
    extracts schedule of games, including season, week, teams, and final scores
    
    :param data (pd.DataFrame): the play-by-play data
    
    :ret schedule: pd.DataFrame containing the schedule of games with scores and home team win indicator
    """
    schedule = data[['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score']]\
        .drop_duplicates().reset_index(drop=True)
    schedule['home_team_win'] = (schedule['home_score'] > schedule['away_score']).astype(int)

    return schedule

def extract_weekly_team_epa_data(data):
    """
    extracts weekly average EPA (Expected Points Added) for rushing and passing 
        performance by offense and defense
    
    :param data (pd.DataFrame): the play-by-play data
    
    :ret Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrames containing 
        weekly rushing and passing EPA for both offense and defense
    """
    rushing_offense_epa = data[data['rush_attempt'] == 1]\
        .groupby(['posteam', 'season', 'week'], as_index=False)['epa'].mean()

    rushing_defense_epa = data[data['rush_attempt'] == 1]\
        .groupby(['defteam', 'season', 'week'], as_index=False)['epa'].mean()

    passing_offense_epa = data[data['pass_attempt'] == 1]\
        .groupby(['posteam', 'season', 'week'], as_index=False)['epa'].mean()

    passing_defense_epa = data[data['pass_attempt'] == 1]\
        .groupby(['defteam', 'season', 'week'], as_index=False)['epa'].mean()
    
    return rushing_offense_epa, rushing_defense_epa, passing_offense_epa, passing_defense_epa

def extract_weekly_team_turnover_data(data):
    """
    extracts weekly turnover data for both offensive and defensive performance
    
    :param data (pd.DataFrame): the play-by-play data
    
    :ret Tuple[pd.DataFrame, pd.DataFrame]: DataFrames containing turnover counts 
        for offense and defense on a weekly basis
    """
    data['turnover'] = data[['interception', 'fumble_lost', 'safety']].any(axis=1).astype(int)

    turnover_count_offense = data.groupby(['posteam', 'season', 'week'], as_index=False)['turnover'].sum()
    turnover_count_defense = data.groupby(['defteam', 'season', 'week'], as_index=False)['turnover'].sum()
    
    scaler = MinMaxScaler()
    turnover_count_offense['turnover_normalized'] = scaler.fit_transform(turnover_count_offense[['turnover']])
    turnover_count_defense['turnover_normalized'] = scaler.fit_transform(turnover_count_defense[['turnover']])

    return turnover_count_offense, turnover_count_defense

def dynamic_window_ewma_epa(x):
    """
    calculates a rolling exponentially weighted moving average (EWMA) of EPA 
        with a dynamic window size based on the week of the season
    
    :param x (pd.DataFrame): DataFrame containing 'epa_shifted' and 'week'
    
    :ret pd.Series: A series of EWMA values
    """
    values = np.zeros(len(x))
    for i, (_, row) in enumerate(x.iterrows()):
        epa = x['epa_shifted'][:i+1]
        span = row['week'] if row['week'] > 10 else 16
        values[i] = epa.ewm(min_periods=1, span=span).mean().iloc[-1]
    return pd.Series(values, index=x.index)

def dynamic_window_ewma_turnover(x):
    """
    calculates a rolling exponentially weighted moving average (EWMA) of turnover rate 
        with a dynamic window size based on the week of the season
    
    :param x (pd.DataFrame): DataFrame containing 'turnover_shifted' and 'week'
    
    :ret pd.Series: A series of EWMA values for turnovers
    """
    values = np.zeros(len(x))
    for i, (_, row) in enumerate(x.iterrows()):
        epa = x['turnover_shifted'][:i+1]
        span = row['week'] if row['week'] > 5 else 5
        values[i] = epa.ewm(min_periods=1, span=span).mean().iloc[-1]
    return pd.Series(values, index=x.index)

def process_and_combine_turnover_data(turnover_count_offense, turnover_count_defense):
    """
    processes and combines turnover data for offense and defense, applying a dynamic window EWMA
    
    :param turnover_count_offense (pd.DataFrame): turnover data for offensive plays
    :param turnover_count_defense (pd.DataFrame): turnover data for defensive plays
    
    :ret turnover_counts (pd.DataFrame): A DataFrame containing combined turnover rates for each team, 
        including dynamic EWMA values
    """
    turnover_count_offense['turnover_shifted'] = turnover_count_offense.groupby('posteam')['turnover_normalized'].shift()
    turnover_count_defense['turnover_shifted'] = turnover_count_defense.groupby('defteam')['turnover_normalized'].shift()

    turnover_count_offense['turnover_ewma_dynamic_window'] = turnover_count_offense.groupby('posteam')\
        .apply(dynamic_window_ewma_turnover).values

    turnover_count_offense.rename(columns={'posteam': 'team'}, inplace=True)

    turnover_count_defense['turnover_ewma_dynamic_window'] = turnover_count_defense.groupby('defteam')\
        .apply(dynamic_window_ewma_turnover).values
    
    turnover_count_defense.rename(columns={'defteam': 'team'}, inplace=True)
    
    turnover_counts = turnover_count_offense.merge(
        turnover_count_defense, on=['team', 'season', 'week'], suffixes=('_offense', '_defense'))
    
    return turnover_counts

def process_and_combine_epa_data(rushing_offense_epa, rushing_defense_epa, passing_offense_epa, passing_defense_epa):
    """
    processes and combines EPA data for both rushing and passing, 
        applying a dynamic window EWMA for each category
    
    :param rushing_defense_epa (pd.DataFrame): Rushing EPA data for defense
    :param rushing_offense_epa (pd.DataFrame): Rushing EPA data for offense
    :param passing_defense_epa (pd.DataFrame): Passing EPA data for defense
    :param passing_offense_epa (pd.DataFrame): Passing EPA data for offense
    
    :ret epa (pd.DataFrame): A combined DataFrame of EPA values for both offense and defense
    """
    rushing_offense_epa['epa_shifted'] = rushing_offense_epa.groupby('posteam')['epa'].shift()
    rushing_defense_epa['epa_shifted'] = rushing_defense_epa.groupby('defteam')['epa'].shift()
    passing_offense_epa['epa_shifted'] = passing_offense_epa.groupby('posteam')['epa'].shift()
    passing_defense_epa['epa_shifted'] = passing_defense_epa.groupby('defteam')['epa'].shift()

    rushing_offense_epa['ewma_dynamic_window'] = rushing_offense_epa.groupby('posteam')\
        .apply(dynamic_window_ewma_epa).values
    rushing_defense_epa['ewma_dynamic_window'] = rushing_defense_epa.groupby('defteam')\
        .apply(dynamic_window_ewma_epa).values

    passing_offense_epa['ewma_dynamic_window'] = passing_offense_epa.groupby('posteam')\
        .apply(dynamic_window_ewma_epa).values
    passing_defense_epa['ewma_dynamic_window'] = passing_defense_epa.groupby('defteam')\
        .apply(dynamic_window_ewma_epa).values

    offense_epa = rushing_offense_epa.merge(
        passing_offense_epa, on=['posteam', 'season', 'week'], suffixes=('_rushing', '_passing')\
        ).rename(columns={'posteam': 'team'})
    defense_epa = rushing_defense_epa.merge(
        passing_defense_epa, on=['defteam', 'season', 'week'], suffixes=('_rushing', '_passing')\
        ).rename(columns={'defteam': 'team'})
    epa = offense_epa.merge(defense_epa, on=['team', 'season', 'week'], suffixes=('_offense', '_defense'))
    
    return epa

def main():
    """
    main function to import, process, and save NFL play-by-play data for modeling
    
    retrieves play-by-play data, processes schedule, EPA, and turnover data, and
        saves the combined dataset as a CSV file
    """
    start = 2009
    data = import_latest_pbp_from(start)

    schedule = extract_schedule(data)

    rushing_offense_epa, rushing_defense_epa, passing_offense_epa, passing_defense_epa =\
        extract_weekly_team_epa_data(data)

    turnover_count_offense, turnover_count_defense = extract_weekly_team_turnover_data(data)

    epa = process_and_combine_epa_data(rushing_offense_epa, rushing_defense_epa, passing_offense_epa, passing_defense_epa)
    epa_home = epa.rename(columns={'team': 'home_team'})
    epa_away = epa.rename(columns={'team': 'away_team'})

    df = schedule.merge(epa_home, on=['season', 'week', 'home_team'], how='left')\
        .merge(epa_away, on=['season', 'week', 'away_team'], how='left', suffixes=('_home', '_away'))

    turnover_counts = process_and_combine_turnover_data(turnover_count_offense, turnover_count_defense)
    turnover_counts_home = turnover_counts.rename(columns={'team': 'home_team'})
    turnover_counts_away = turnover_counts.rename(columns={'team': 'away_team'})

    df = df.merge(turnover_counts_home, on=['season', 'week', 'home_team'], how='left')\
        .merge(turnover_counts_away, on=['season', 'week', 'away_team'], how='left', suffixes=('_home', '_away'))

    first_season = epa['season'].min()
    df = df[df['season'] != first_season]

    today = datetime.now().strftime('%Y-%m-%d')
    output_dir = os.path.join('data', today)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df.to_csv(os.path.join(output_dir, 'model_data.csv'), index=False)
    print(f"Model data with game information saved to '{os.path.join(today, 'model_data.csv')}'")

if __name__ == '__main__':
    main()
