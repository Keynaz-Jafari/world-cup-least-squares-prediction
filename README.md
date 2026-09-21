# World Cup Match Prediction with Least-Squares Regression

A football match score prediction project implemented in Python using linear algebra and least-squares regression.

The project predicts the final score of a football match using statistics from the recent matches of both teams. It was developed without machine-learning libraries; the regression model is built directly using matrix operations and the Moore–Penrose pseudo-inverse.

## Overview

Each match is represented using data from the five most recent matches of both teams.

The original dataset contains 40 numerical features:

- 5 recent matches for Team 1
- 5 recent matches for Team 2
- 4 values for each historical match:
  - hosting status
  - first-team goals
  - second-team goals
  - match type

The training data also contains the final goals scored by both teams.

The project transforms these raw values into more informative statistical features and uses them to predict the score of a new match.

## Feature Engineering

For each team, the program extracts features such as:

- Average goals scored
- Average goals conceded
- Average goal difference
- Average total goals
- Weighted statistics based on match type
- Standard deviation of goals and goal difference
- Maximum and minimum goals
- Win, draw, and loss rates
- Recent form score
- Home, away, and neutral-game ratios
- Trends across recent matches

It also creates comparative features between the two teams, such as:

- attacking strength vs. opponent defense
- difference in recent form
- relative team performance

This keeps the underlying model linear while providing a richer representation of the input data. :chatgpt-content-reference{index="0"}

## Least-Squares Model

After feature extraction, the project constructs a design matrix:

```text
X ∈ R^(n × d)
