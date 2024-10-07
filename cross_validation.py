"""
cross_validation_analysis.py
script to perform cross-validation
"""

from datetime import datetime
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def read_data():
    """
    read the model data written by data_extraction.py

    :ret df (pd.DataFrame): the model data
    """
    today = datetime.now().strftime('%Y-%m-%d')
    data_directory = os.path.join('data', today)
    epa_data_path = os.path.join(data_directory, 'model_data.csv')

    if not os.path.exists(epa_data_path):
        raise FileNotFoundError(f"model_data.csv not found in directory '{data_directory}'. Please ensure the file exists.")

    df = pd.read_csv(epa_data_path)
    df = df.dropna()

    return df

def cross_validate_model(df, n_splits=5):
    """
    perform cross-validation using logistic regression

    :param df (pd.DataFrame): DataFrame containing the model data
    :param n_splits (int): number of splits for K-fold cross-validation (default: 5)

    :ret Tuple[LogisticRegression, float, List[float]]: the trained LogisticRegression model,
        average accuracy across all folds, and a list of accuracy scores for each fold
    """
    target = 'home_team_win'
    features = [column for column in df.columns if 'ewma' in column and 'dynamic' in column]

    X = df[features].values
    y = df[target].values

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    accuracy_scores = []
    fold = 1

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)
        print(f"Fold {fold}: Accuracy = {accuracy:.2%}")
        fold += 1

    avg_accuracy = np.mean(accuracy_scores)
    print(f"\nAverage cross-validation accuracy: {avg_accuracy:.2%}")

    return clf, avg_accuracy, accuracy_scores

def main():
    """
    main function to perform cross-validation on the model data and save the results

    reads the data, performs cross-validation, saves the results to a text file,
        and visualizes the feature importance based on the coefficients from the trained model
    """
    df = read_data()

    clf, avg_accuracy, accuracy_scores = cross_validate_model(df)

    output_directory = os.path.join('output', datetime.now().strftime('%Y-%m-%d'))
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    results_path = os.path.join(output_directory, 'cross_validation_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Average cross-validation accuracy: {avg_accuracy:.2%}\n")
        for i, score in enumerate(accuracy_scores, start=1):
            f.write(f"Fold {i}: Accuracy = {score:.2%}\n")

    print(f"\nCross-validation results saved to '{results_path}'")

    fig, ax = plt.subplots()

    features = [' '.join(word[0] for word in column.split('_')) for column in df.columns if 'ewma' in column and 'dynamic' in column]

    coef_ = clf.coef_[0]

    features_coef_sorted = sorted(zip(features, coef_), key=lambda x:x[-1], reverse=True)

    features_sorted = [feature for feature, _ in features_coef_sorted]
    coef_sorted = [coef for _, coef in features_coef_sorted]

    ax.set_title('Feature importance')

    ax.barh(features_sorted, coef_sorted)
    plt.show()

if __name__ == '__main__':
    main()
