# NFLog_reg

An end-to-end logistic regression model for predicting NFL games built using Python. This repository contains scripts for extracting and processing play-by-play (PBP) data, building models, and generating predictions for upcoming games.

## Overview

This repo includes the following scripts:

- **data_extraction.py**: Extracts NFL play-by-play data, computes advanced metrics (like EWMA of EPA and turnovers), and prepares the data for modeling.
- **cross_validation.py**: Performs cross-validation on the logistic regression model, evaluates accuracy, and visualizes feature importance.
- **curr_season_accuracy.py**: Evaluates the prediction accuracy for the current NFL season when trained on only data 2010-2023.
- **predict_next_week_nfl.py**: Generates predictions for the next week's NFL games, using the latest model data and dynamically computed metrics.

## Getting Started

1. **Clone the repository**:
```
git clone https://github.com/jmue2/NFLog_reg.git
```

2.  **Set up a virtual environment**:
```
python -m venv venv
```

3. **Activate the virtual environment**:
- On Windows:
  ```
  venv\Scripts\activate
  ```
- On macOS/Linux:
  ```
  source venv/bin/activate
  ```

4. **Install the required libraries**:
```
pip install -r requirements.txt
```

5. **Run the scripts**:
- To download and process the latest data, run:
  ```
  python data_extraction.py
  ```
- To analyze the model's performance using cross-validation, run:
  ```
  python cross_validation_analysis.py
  ```
- To see how the model performs on the current season, run:
  ```
  python curr_season_accuracy.py
  ```
- To predict outcomes for upcoming games, run:
  ```
  python predict_next_week_nfl.py
  ```

6. **Deactivate the virtual environment** when done:
```
deactivate
```

## Acknowledgements

This project was inspired by various resources and research on NFL game prediction using logistic regression:

- The article "[NFL Game Prediction Using Logistic Regression](https://opensourcefootball.com/posts/2021-01-21-nfl-game-prediction-using-logistic-regression/)" from Open Source Football provided valuable insights and methods for building a predictive model.
- Bouzianis, Stephen, "Predicting the Outcome of NFL Games Using Logistic Regression" (2019). Honors Theses and Capstones. 474. [Link to the paper](https://scholars.unh.edu/honors/474).
