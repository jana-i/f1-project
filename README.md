# F1 Data Analysis & Race Predictions

Exploratory data analysis of Formula 1 history (drivers, constructors, qualifying vs. race performance) 
combined with a small machine learning experiment that predicts average race lap times from qualifying times and sector times.


## Overview

The project has two parts:

Historical analysis: I merged four datasets (results, races, drivers, constructors) into one clean table of about 26,000 race results and explored who wins, which teams dominate, and how much starting position matters.
Prediction experiment: Using the FastF1 API, I pulled lap and sector data from past races, combined it with 2025 qualifying times, and trained a Gradient Boosting model to estimate each driver's average race lap time.

## Tech Stack
- Language: Python
- Data handling: Pandas, NumPy
- Visualization: Matplotlib, Seaborn
- Machine learning: scikit-learn (GradientBoostingRegressor)
- Data sources: Formula 1 historical results dataset (CSV files: results, races, drivers, constructors), and the FastF1 API for lap and sector timing
- Environment: Jupyter / Google Colab

## Part 1: Data Cleaning & Analysis

### Cleaning steps

- Merged results with race, driver and constructor tables
- Dropped unneeded columns, renamed columns, reordered them
- Removed the incomplete 2019 season data
- Replaced \N placeholders with proper missing values and converted columns to numeric types
- Final dataset: 26,339 rows x 15 columns

### Questions explored

- Which drivers have the most Grand Prix wins?
- Which constructors have the most wins?
- How strongly does starting (grid) position relate to finishing position?
- How has the average fastest-lap speed changed at each Grand Prix since 2004?

### Approach

- Loaded race data with FastF1 and converted lap and sector times to seconds
- Computed each driver's average sector times from the 2024 race
- Combined them with 2025 qualifying times
- Trained a GradientBoostingRegressor (200 estimators, learning rate 0.1) with an 80/20 train/test split
- Predicted average race lap time per driver and ranked the drivers

Result: Mean absolute error of 1.35 seconds on the test set.

### Limitations (what I would improve)

* I want to be upfront about the weak points of the prediction part:
* Very small dataset. There are about 20 drivers, so the training set has around 16 rows and the test set around 4. 
* A 1.35 s error on 4 samples is not a reliable measure of accuracy.
* Predicts lap time, not race results. The model estimates average lap time and does not account for pit stops, tyre strategy, safety cars, weather or reliability.
* Missing values filled with 0. Drivers without 2024 data (for example, rookies) get zeros for sector times, which distorts their predictions.

## What I Learned
- Cleaning and merging several related datasets with Pandas
- Visualizing distributions and relationships with Seaborn
- Pulling real-world timing data from an API (FastF1)
- Building and evaluating a regression model and recognizing the limits of a small dataset
