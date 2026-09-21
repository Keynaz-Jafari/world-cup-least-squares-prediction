#!/usr/bin/env python3
"""
Least-squares pseudo-inverse predictor for the World Cup score project.

The program intentionally avoids machine-learning libraries.  It uses linear
algebra directly: build a design matrix, compute a pseudo-inverse least-squares
solution, then convert continuous outputs into a discrete football score.

Default usage:
    python3 project.py

Optional usage:
    python3 project.py --train Dataset.txt --test DatasetTest.txt --out predictions.txt
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - fallback is for very small environments.
    np = None


FEATURE_COUNT = 40
TARGET_COUNT = 2


@dataclass
class MatchRow:
    team1: str
    team2: str
    features: List[float]
    target: Tuple[int, int] | None


@dataclass(frozen=True)
class ModelConfig:
    feature_mode: str
    ridge: float
    w_goals: float
    w_sum: float
    w_diff: float
    name: str


def numeric_tokens(row: Sequence[str]) -> List[float]:
    """Read numeric columns while ignoring empty tab fields from the source file."""
    values: List[float] = []
    for token in row[2:]:
        token = token.strip()
        if token == "":
            continue
        values.append(float(token))
    return values


def parse_dataset(path: str | Path, require_targets: bool) -> List[MatchRow]:
    rows: List[MatchRow] = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if not row or all(cell.strip() == "" for cell in row):
                continue
            if len(row) < 2:
                raise ValueError(f"{path}:{line_number}: every row must start with two team names")

            nums = numeric_tokens(row)
            if len(nums) < FEATURE_COUNT:
                raise ValueError(
                    f"{path}:{line_number}: expected at least {FEATURE_COUNT} numeric features, "
                    f"found {len(nums)}"
                )

            features = nums[:FEATURE_COUNT]
            target = None
            if len(nums) >= FEATURE_COUNT + TARGET_COUNT:
                target = (int(round(nums[FEATURE_COUNT])), int(round(nums[FEATURE_COUNT + 1])))
            elif require_targets:
                raise ValueError(
                    f"{path}:{line_number}: training rows must include two goal targets"
                )

            rows.append(MatchRow(row[0].strip(), row[1].strip(), features, target))
    if not rows:
        raise ValueError(f"{path}: no rows were found")
    return rows


def chunk_games(features: Sequence[float]) -> Tuple[List[List[float]], List[List[float]]]:
    if len(features) != FEATURE_COUNT:
        raise ValueError(f"expected {FEATURE_COUNT} features, got {len(features)}")
    games = [list(features[i : i + 4]) for i in range(0, FEATURE_COUNT, 4)]
    return games[:5], games[5:]


def viewed_score(game: Sequence[float]) -> Tuple[float, float, float, float]:
    """Convert one historical game to (goals_for, goals_against, diff, type_weight).

    Hosting 2 means the team under consideration was the second team in that
    historical match, so its goals are in the secondTeamGoals column.  Hosting
    0 or 3 is treated as neutral and the team is read as the first team.
    """
    hosting, first_goals, second_goals, game_type = game
    if int(round(hosting)) == 2:
        goals_for = second_goals
        goals_against = first_goals
    else:
        goals_for = first_goals
        goals_against = second_goals

    type_weight = {1: 1.0, 2: 1.35, 3: 1.8}.get(int(round(game_type)), 1.0)
    return goals_for, goals_against, goals_for - goals_against, type_weight


def safe_mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def safe_std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mu = safe_mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / len(values))


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    total = sum(weights)
    if total == 0:
        return safe_mean(values)
    return sum(value * weight for value, weight in zip(values, weights)) / total


def team_statistics(games: Sequence[Sequence[float]]) -> List[float]:
    rows = [viewed_score(game) for game in games]
    goals_for = [row[0] for row in rows]
    goals_against = [row[1] for row in rows]
    diffs = [row[2] for row in rows]
    totals = [gf + ga for gf, ga in zip(goals_for, goals_against)]
    type_weights = [row[3] for row in rows]

    # A mild recency curve.  If the file order changes, cross-validation can
    # prefer feature settings where these trend features matter less.
    recency = [1.0, 1.1, 1.2, 1.3, 1.4]
    combined_weights = [a * b for a, b in zip(type_weights, recency)]

    hosts = [1.0 if int(round(game[0])) == 1 else 0.0 for game in games]
    guests = [1.0 if int(round(game[0])) == 2 else 0.0 for game in games]
    neutrals = [1.0 if int(round(game[0])) in (0, 3) else 0.0 for game in games]
    official_or_wc = [1.0 if int(round(game[3])) >= 2 else 0.0 for game in games]

    wins = [1.0 if diff > 0 else 0.0 for diff in diffs]
    draws = [1.0 if diff == 0 else 0.0 for diff in diffs]
    losses = [1.0 if diff < 0 else 0.0 for diff in diffs]
    points = [3.0 if diff > 0 else 1.0 if diff == 0 else 0.0 for diff in diffs]

    first_part = slice(0, 3)
    last_part = slice(3, 5)
    gf_trend = safe_mean(goals_for[last_part]) - safe_mean(goals_for[first_part])
    ga_trend = safe_mean(goals_against[last_part]) - safe_mean(goals_against[first_part])
    diff_trend = safe_mean(diffs[last_part]) - safe_mean(diffs[first_part])

    return [
        safe_mean(goals_for),
        safe_mean(goals_against),
        safe_mean(diffs),
        safe_mean(totals),
        weighted_mean(goals_for, combined_weights),
        weighted_mean(goals_against, combined_weights),
        weighted_mean(diffs, combined_weights),
        weighted_mean(totals, combined_weights),
        safe_std(goals_for),
        safe_std(goals_against),
        safe_std(diffs),
        max(goals_for),
        max(goals_against),
        min(goals_for),
        min(goals_against),
        safe_mean(wins),
        safe_mean(draws),
        safe_mean(losses),
        safe_mean(points),
        weighted_mean(points, combined_weights),
        safe_mean(hosts),
        safe_mean(guests),
        safe_mean(neutrals),
        safe_mean(official_or_wc),
        goals_for[-1],
        goals_against[-1],
        diffs[-1],
        gf_trend,
        ga_trend,
        diff_trend,
    ]


def engineered_features(features: Sequence[float]) -> List[float]:
    first_games, second_games = chunk_games(features)
    first_stats = team_statistics(first_games)
    second_stats = team_statistics(second_games)

    diff_stats = [a - b for a, b in zip(first_stats, second_stats)]
    sum_stats = [a + b for a, b in zip(first_stats, second_stats)]

    first_attack = first_stats[0]
    first_defense = first_stats[1]
    second_attack = second_stats[0]
    second_defense = second_stats[1]
    first_weighted_attack = first_stats[4]
    first_weighted_defense = first_stats[5]
    second_weighted_attack = second_stats[4]
    second_weighted_defense = second_stats[5]

    match_features = [
        first_attack - second_defense,
        second_attack - first_defense,
        first_weighted_attack - second_weighted_defense,
        second_weighted_attack - first_weighted_defense,
        first_stats[2] - second_stats[2],
        first_stats[6] - second_stats[6],
        first_stats[18] - second_stats[18],
        first_stats[19] - second_stats[19],
        first_stats[27] - second_stats[27],
        first_stats[28] - second_stats[28],
        first_stats[29] - second_stats[29],
        abs(first_stats[2] - second_stats[2]),
        abs(first_attack - second_attack),
        abs(first_defense - second_defense),
    ]

    return first_stats + second_stats + diff_stats + sum_stats[:10] + match_features


def make_feature_vector(features: Sequence[float], mode: str) -> List[float]:
    raw = list(features)
    engineered = engineered_features(features)

    if mode == "raw":
        values = raw
    elif mode == "summary":
        values = engineered
    elif mode == "full":
        values = raw + engineered
    elif mode == "compact":
        values = raw + engineered[:30] + engineered[60:]
    else:
        raise ValueError(f"unknown feature mode: {mode}")

    # Add a small non-linear basis while keeping the final model linear in its
    # coefficients.  These are still ordinary least-squares features.
    if mode in {"summary", "full", "compact"}:
        selected = values[-14:]
        values = values + [x * x for x in selected] + [
            selected[i] * selected[j]
            for i, j in [(0, 1), (0, 2), (1, 3), (4, 5), (6, 7), (8, 10)]
        ]

    return values


def build_matrix(rows: Sequence[MatchRow], mode: str) -> List[List[float]]:
    return [make_feature_vector(row.features, mode) for row in rows]


def target_matrix(rows: Sequence[MatchRow]) -> List[List[float]]:
    targets: List[List[float]] = []
    for row in rows:
        if row.target is None:
            raise ValueError("training rows must have targets")
        g1, g2 = row.target
        targets.append([float(g1), float(g2), float(g1 + g2), float(g1 - g2)])
    return targets


def standardize_train_test(
    train_x: Sequence[Sequence[float]], test_x: Sequence[Sequence[float]]
) -> Tuple[List[List[float]], List[List[float]]]:
    if np is not None:
        train = np.asarray(train_x, dtype=float)
        test = np.asarray(test_x, dtype=float)
        mean = train.mean(axis=0)
        std = train.std(axis=0)
        std[std < 1e-12] = 1.0
        train = (train - mean) / std
        test = (test - mean) / std
        train = np.column_stack([np.ones(train.shape[0]), train])
        test = np.column_stack([np.ones(test.shape[0]), test])
        return train.tolist(), test.tolist()

    cols = list(zip(*train_x))
    means = [safe_mean(col) for col in cols]
    stds = [safe_std(col) or 1.0 for col in cols]

    def transform(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
        out: List[List[float]] = []
        for row in matrix:
            out.append([1.0] + [(value - mu) / sigma for value, mu, sigma in zip(row, means, stds)])
        return out

    return transform(train_x), transform(test_x)


def fit_pseudo_inverse(x: Sequence[Sequence[float]], y: Sequence[Sequence[float]], ridge: float):
    """Fit W in XW ~= Y using a pseudo-inverse least-squares solution."""
    if np is not None:
        x_array = np.asarray(x, dtype=float)
        y_array = np.asarray(y, dtype=float)
        if ridge > 0:
            penalty = math.sqrt(ridge) * np.eye(x_array.shape[1])
            penalty[0, 0] = 0.0  # do not penalize the intercept
            x_array = np.vstack([x_array, penalty])
            y_array = np.vstack([y_array, np.zeros((x_array.shape[1], y_array.shape[1]))])
        return np.linalg.pinv(x_array, rcond=1e-10).dot(y_array)

    return fit_pseudo_inverse_fallback(x, y, ridge if ridge > 0 else 1e-8)


def predict_matrix(x: Sequence[Sequence[float]], weights) -> List[List[float]]:
    if np is not None:
        return np.asarray(x, dtype=float).dot(weights).tolist()
    return matmul(x, weights)


def transpose(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    return [list(col) for col in zip(*matrix)]


def matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    b_t = transpose(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in b_t] for row in a]


def solve_linear_system(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    n = len(a)
    m = len(b[0])
    aug = [list(a[i]) + list(b[i]) for i in range(n)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-12:
            aug[pivot][col] = 1e-12
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [value / scale for value in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor:
                aug[row] = [rv - factor * pv for rv, pv in zip(aug[row], aug[col])]

    return [row[n : n + m] for row in aug]


def fit_pseudo_inverse_fallback(
    x: Sequence[Sequence[float]], y: Sequence[Sequence[float]], ridge: float
) -> List[List[float]]:
    n = len(x)
    p = len(x[0])
    if p <= n:
        x_t = transpose(x)
        xtx = matmul(x_t, x)
        for i in range(p):
            if i != 0:
                xtx[i][i] += ridge
        xty = matmul(x_t, y)
        return solve_linear_system(xtx, xty)

    xx_t = matmul(x, transpose(x))
    for i in range(n):
        xx_t[i][i] += ridge
    alpha = solve_linear_system(xx_t, y)
    return matmul(transpose(x), alpha)


def score_360(predicted: Tuple[int, int], actual: Tuple[int, int]) -> int:
    if predicted == actual:
        return 10
    pred_diff = predicted[0] - predicted[1]
    actual_diff = actual[0] - actual[1]
    if pred_diff == actual_diff:
        return 7
    if (pred_diff > 0 and actual_diff > 0) or (pred_diff < 0 and actual_diff < 0):
        return 5
    return 2


def choose_discrete_score(
    prediction: Sequence[float],
    config: ModelConfig,
    max_goal: int,
) -> Tuple[int, int]:
    goal1, goal2, total_goals, goal_diff = prediction
    max_goal = max(5, max_goal)
    best_pair = (0, 0)
    best_loss = float("inf")

    for first, second in itertools.product(range(max_goal + 1), repeat=2):
        loss = (
            config.w_goals * ((first - goal1) ** 2 + (second - goal2) ** 2)
            + config.w_sum * ((first + second - total_goals) ** 2)
            + config.w_diff * ((first - second - goal_diff) ** 2)
        )

        # Football scores are small.  This weak prior prevents unstable linear
        # extrapolation from choosing a wild score when several candidates are
        # close under the least-squares surrogate.
        loss += 0.03 * (first + second) ** 2
        if loss < best_loss:
            best_loss = loss
            best_pair = (first, second)

    return best_pair


def candidate_configs() -> List[ModelConfig]:
    feature_modes = ["summary", "compact", "full", "raw"]
    ridge_values = [0.0, 0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0]
    converters = [
        ("balanced", 1.0, 0.45, 1.4),
        ("difference", 0.65, 0.35, 2.4),
        ("goals", 1.6, 0.25, 0.9),
        ("sumdiff", 0.45, 0.75, 1.8),
    ]
    return [
        ModelConfig(mode, ridge, w_goals, w_sum, w_diff, f"{mode}-{label}-ridge{ridge:g}")
        for mode in feature_modes
        for ridge in ridge_values
        for label, w_goals, w_sum, w_diff in converters
    ]


def train_predict_with_config(
    train_rows: Sequence[MatchRow],
    test_rows: Sequence[MatchRow],
    config: ModelConfig,
) -> List[Tuple[int, int]]:
    train_raw_x = build_matrix(train_rows, config.feature_mode)
    test_raw_x = build_matrix(test_rows, config.feature_mode)
    train_x, test_x = standardize_train_test(train_raw_x, test_raw_x)
    weights = fit_pseudo_inverse(train_x, target_matrix(train_rows), config.ridge)
    continuous = predict_matrix(test_x, weights)
    max_goal = max(max(row.target or (0, 0)) for row in train_rows) + 2
    return [choose_discrete_score(row, config, max_goal) for row in continuous]


def evaluate_config(rows: Sequence[MatchRow], config: ModelConfig) -> Tuple[float, float, int]:
    total_score = 0
    total_mse = 0.0
    exact = 0

    for index in range(len(rows)):
        train_rows = list(rows[:index]) + list(rows[index + 1 :])
        test_rows = [rows[index]]
        prediction = train_predict_with_config(train_rows, test_rows, config)[0]
        actual = rows[index].target
        assert actual is not None
        total_score += score_360(prediction, actual)
        total_mse += (prediction[0] - actual[0]) ** 2 + (prediction[1] - actual[1]) ** 2
        exact += 1 if prediction == actual else 0

    return total_score / len(rows), total_mse / len(rows), exact


def select_model(rows: Sequence[MatchRow]) -> Tuple[ModelConfig, Tuple[float, float, int]]:
    best_config: ModelConfig | None = None
    best_metrics: Tuple[float, float, int] | None = None
    best_key: Tuple[float, float, int, float] | None = None

    for config in candidate_configs():
        avg_score, mse, exact = evaluate_config(rows, config)
        key = (avg_score, -mse, exact, -config.ridge)
        if best_key is None or key > best_key:
            best_key = key
            best_config = config
            best_metrics = (avg_score, mse, exact)

    assert best_config is not None and best_metrics is not None
    return best_config, best_metrics


def write_predictions(path: str | Path, predictions: Sequence[Tuple[int, int]]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for first, second in predictions:
            handle.write(f"{first}\t{second}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pseudo-inverse least-squares football predictor")
    parser.add_argument("--train", default="Dataset.txt", help="training file with 40 features + 2 goals")
    parser.add_argument("--test", default="DatasetTest.txt", help="test file with 40 features")
    parser.add_argument("--out", default="predictions.txt", help="output predictions file")
    parser.add_argument(
        "--config",
        default="auto",
        help="model config name from CV output, or 'auto' to select by leave-one-out",
    )
    args = parser.parse_args()

    train_rows = parse_dataset(args.train, require_targets=True)
    test_rows = parse_dataset(args.test, require_targets=False)

    if args.config == "auto":
        config, metrics = select_model(train_rows)
    else:
        matches = [config for config in candidate_configs() if config.name == args.config]
        if not matches:
            names = ", ".join(config.name for config in candidate_configs())
            raise ValueError(f"unknown config {args.config!r}. Available configs: {names}")
        config = matches[0]
        metrics = evaluate_config(train_rows, config)

    predictions = train_predict_with_config(train_rows, test_rows, config)
    write_predictions(args.out, predictions)

    avg_score, mse, exact = metrics
    print(f"training rows: {len(train_rows)}")
    print(f"test rows: {len(test_rows)}")
    print(f"selected model: {config.name}")
    print(f"leave-one-out 360 score: {avg_score:.3f} / 10")
    print(f"leave-one-out exact scores: {exact} / {len(train_rows)}")
    print(f"leave-one-out squared goal error: {mse:.3f}")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
