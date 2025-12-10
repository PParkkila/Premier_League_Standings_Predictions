# =============================================================
# PREMIER LEAGUE FINAL STANDINGS PREDICTION — MULTI-MODEL VERSION
# Including: RF, XGBoost, DecisionTree, LinearRegression, LogisticRegression,
# CatBoost Regressor & Classifier, with safe training wrappers.
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression

import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier


# =============================================================
# 1. BUILD MATCHDAY-CUMULATIVE TABLE
# =============================================================

matches = pd.read_csv("./standings/Premier_1993-2022.csv")
matches = matches[matches["Season_End_Year"] == 2020].copy()

matches["Match Date"] = pd.to_datetime(matches["Date"], dayfirst=True)

def pts(diff):
    if diff > 0: return 3
    if diff < 0: return 0
    return 1

matches["HomePts"] = (matches["HomeGoals"] - matches["AwayGoals"]).apply(pts)
matches["AwayPts"] = (matches["AwayGoals"] - matches["HomeGoals"]).apply(pts)

home = matches[["HomeTeam", "HomePts", "Match Date"]].rename(
    columns={"HomeTeam":"team", "HomePts":"points"}
)
away = matches[["AwayTeam", "AwayPts", "Match Date"]].rename(
    columns={"AwayTeam":"team", "AwayPts":"points"}
)

long = pd.concat([home, away], ignore_index=True)
long = long.sort_values(["team", "Match Date"])
long["matchday"] = long.groupby("team").cumcount() + 1
long["cum_points"] = long.groupby("team")["points"].cumsum()

wide = long.pivot(index="team", columns="matchday", values="cum_points")
wide = wide.ffill(axis=1).fillna(0)
wide.columns = [f"MD_{int(md)}" for md in wide.columns]
weekly_df = wide.reset_index()

md_cols = sorted([c for c in weekly_df.columns if c.startswith("MD_")],
                 key=lambda x: int(x.split("_")[1]))
max_md = len(md_cols)
print(f"Detected {max_md} matchdays.")


# =============================================================
# 2. FINAL RANKINGS TARGETS
# =============================================================

weekly_df["final_points"] = weekly_df[md_cols[-1]]
weekly_df = weekly_df.sort_values("final_points", ascending=False).reset_index(drop=True)

weekly_df["true_rank"] = weekly_df.index + 1
weekly_df["top4"] = (weekly_df["true_rank"] <= 4).astype(int)

true_rank = weekly_df["true_rank"].values
top4 = weekly_df["top4"].values


# =============================================================
# 3. SAFE TRAINING WRAPPERS
# =============================================================

def ranking_error(pred, true):
    return np.linalg.norm(pred - true) / np.linalg.norm(true)

def safe_train_and_predict(model, X, y, metric_fn):
    # Cannot train on identical labels
    if len(set(y)) < 2:
        return np.nan
    try:
        model.fit(X, y)
        pred = model.predict(X)
        return metric_fn(pred, y)
    except Exception:
        return np.nan


# =============================================================
# 4. MODEL DICTIONARIES
# =============================================================

ranking_models = {
    "RandomForest": lambda: RandomForestRegressor(n_estimators=500, random_state=42),
    "XGBoost": lambda: xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        objective="reg:squarederror", random_state=42
    ),
    "DecisionTree": lambda: DecisionTreeRegressor(random_state=42),
    "LinearRegression": lambda: LinearRegression(),
    "CatBoost": lambda: CatBoostRegressor(
        depth=6, learning_rate=0.05, iterations=400,
        verbose=False, random_seed=42
    ),
}

classification_models = {
    "RandomForest": lambda: RandomForestClassifier(n_estimators=500, random_state=42),
    "XGBoost": lambda: xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        use_label_encoder=False, eval_metric="logloss", random_state=42
    ),
    "DecisionTree": lambda: DecisionTreeClassifier(random_state=42),
    "LogisticRegression": lambda: LogisticRegression(max_iter=1000),
    "CatBoost": lambda: CatBoostClassifier(
        depth=6, learning_rate=0.05, iterations=400,
        verbose=False, random_seed=42
    ),
}


# =============================================================
# 5. TRAIN MODELS PROGRESSIVELY (MD_1 .. MD_k)
# =============================================================

ranking_results = {name: [] for name in ranking_models}
top4_results = {name: [] for name in classification_models}

for k in range(1, max_md + 1):

    features = [f"MD_{i}" for i in range(1, k+1)]
    X = weekly_df[features].values

    # Ranking (Regression)
    for model_name, model_fn in ranking_models.items():
        model = model_fn()
        score = safe_train_and_predict(model, X, true_rank, ranking_error)
        ranking_results[model_name].append(score)

    # Top-4 (Classification)
    for model_name, model_fn in classification_models.items():
        model = model_fn()
        score = safe_train_and_predict(model, X, top4,
                    lambda pred, true: accuracy_score(true, pred))
        top4_results[model_name].append(score)


# =============================================================
# 6. PLOTTING
# =============================================================

weeks = np.arange(1, max_md + 1)

# Ranking error plot
plt.figure(figsize=(13, 6))
for name, vals in ranking_results.items():
    plt.plot(weeks, vals, marker='o', label=name)
plt.title("Ranking Error vs Matchday (Multiple Models)")
plt.xlabel("Matchday")
plt.ylabel("Relative 2-Norm Error")
plt.grid()
plt.legend()
plt.show()

# Top-4 accuracy plot
plt.figure(figsize=(13, 6))
for name, vals in top4_results.items():
    plt.plot(weeks, vals, marker='o', label=name)
plt.title("Top-4 Accuracy vs Matchday (Multiple Models)")
plt.xlabel("Matchday")
plt.ylabel("Accuracy")
plt.grid()
plt.legend()
plt.show()

print("Finished training all models!")
