"""
curr_season_accuracy.py
script to analyze and evaluate model predictions for the current NFL season
"""

from datetime import datetime
import os

from sklearn.linear_model import LogisticRegression

from cross_validation import read_data

def build_model(df):
    """
    builds a logistic regression model using data from past seasons

    :param df (pd.DataFrame): DataFrame containing the model data

    :ret Tuple[LogisticRegression, np.ndarray, np.ndarray]: the trained LogisticRegression model,
        feature matrix X, and target vector y
    """
    target = 'home_team_win'
    features = [column for column in df.columns if 'ewma' in column and 'dynamic' in column]
    current_season = df['season'].max()

    X = df.loc[df['season'] != current_season, features].values
    y = df.loc[df['season'] != current_season, target].values

    clf = LogisticRegression()
    clf.fit(X, y)

    return clf, X, y

def main():
    """
    main function to evaluate prediction accuracy for the current NFL season

    reads the data, builds a logistic regression model, makes predictions for the
        current season, saves the results, and prints overall and week-by-week accuracy
    """
    df = read_data()

    clf, X, y = build_model(df)

    features = [column for column in df.columns if 'ewma' in column and 'dynamic' in column]
    current_season = df['season'].max()

    df_curr = df.loc[(df['season'] == current_season)].assign(
        predicted_winner = lambda x: clf.predict(x[features]),
        home_team_win_probability = lambda x: clf.predict_proba(x[features])[:, 1]
    )\
    [['home_team', 'away_team', 'week', 'predicted_winner', 'home_team_win_probability', 'home_team_win']]

    df_curr['actual_winner'] = df_curr.apply(lambda x: x.home_team if x.home_team_win else x.away_team, axis=1)
    df_curr['predicted_winner'] = df_curr.apply(lambda x: x.home_team if x.predicted_winner == 1 else x.away_team, axis=1)
    df_curr['win_probability'] = df_curr.apply(lambda x: x.home_team_win_probability if x.predicted_winner == x.home_team else 1 - x.home_team_win_probability, axis=1)
    df_curr['correct_prediction'] = (df_curr['predicted_winner'] == df_curr['actual_winner']).astype(int)

    df_curr = df_curr.drop(columns=['home_team_win_probability', 'home_team_win'])

    output_directory = os.path.join('output', datetime.now().strftime('%Y-%m-%d'))
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    df_curr.to_csv(os.path.join(output_directory, '2024_predictions.csv'), index=False)

    df_curr.sort_values(by='win_probability', ascending=False).reset_index(drop=True)\
        .to_csv(os.path.join(output_directory, 'sorted_2024_predictions.csv'), index=False)

    overall_accuracy = df_curr['correct_prediction'].mean()
    print(f"Overall accuracy for 2024: {overall_accuracy:.2%}")

    week_by_week_accuracy = df_curr.groupby('week')['correct_prediction'].mean()
    print("\nWeek-by-week accuracy for 2024:")
    for week, accuracy in week_by_week_accuracy.items():
        print(f"Week {int(week)}: {accuracy:.2%}")

    most_confident_predictions = df_curr.sort_values(by='win_probability', ascending=False).head(10)
    print("\nTop 10 most confident predictions:")
    print(most_confident_predictions[['week', 'home_team', 'away_team', 'predicted_winner', 'win_probability', 'correct_prediction']])

if __name__ == '__main__':
    main()