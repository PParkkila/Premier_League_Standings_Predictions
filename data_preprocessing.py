import pandas as pd
import numpy as np

df = pd.read_csv("./standings/Premier_1993-2022.csv")
df = df[df["Season_End_Year"] == 2020]
df["Match Date"] = pd.to_datetime(df["Date"], dayfirst=True)

def pts(goal_diff):
    if goal_diff > 0: return 3
    if goal_diff < 0: return 0
    return 1

df["HomePts"] = (df["HomeGoals"] - df["AwayGoals"]).apply(pts)
df["AwayPts"] = (df["AwayGoals"] - df["HomeGoals"]).apply(pts)

home = df[["HomeTeam", "HomePts", "Match Date"]].rename(
    columns={"HomeTeam":"team", "HomePts":"points"}
)
away = df[["AwayTeam", "AwayPts", "Match Date"]].rename(
    columns={"AwayTeam":"team", "AwayPts":"points"}
)

long = pd.concat([home, away], ignore_index=True)
long = long.sort_values(["team", "Match Date"])
long["matchday"] = long.groupby("team").cumcount() + 1
long["cum_points"] = long.groupby("team")["points"].cumsum()

wide = long.pivot(index="team", columns="matchday", values="cum_points").fillna(method="ffill", axis=1).fillna(0)
wide.columns = [f"MD_{md}" for md in wide.columns]
weekly_df = wide.reset_index()

weekly_df["final_points"] = weekly_df["MD_38"]
weekly_df = weekly_df.sort_values("final_points", ascending=False).reset_index(drop=True)
weekly_df["true_rank"] = weekly_df.index + 1
weekly_df["top4"] = (weekly_df["true_rank"] <= 4).astype(int)

true_rank = weekly_df["true_rank"].values
top4 = weekly_df["top4"].values

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score

ranking_errors = []
accuracies = []

for k in range(1, 39):
    feat = [f"MD_{i}" for i in range(1, k+1)]
    X = weekly_df[feat].values

    reg = RandomForestRegressor(n_estimators=500, random_state=42)
    reg.fit(X, true_rank)
    pred_r = reg.predict(X)
    ranking_errors.append(np.linalg.norm(pred_r - true_rank)/np.linalg.norm(true_rank))

    clf = RandomForestClassifier(n_estimators=500, random_state=42)
    clf.fit(X, top4)
    pred_c = clf.predict(X)
    accuracies.append(accuracy_score(top4, pred_c))


import matplotlib.pyplot as plt

weeks = range(1, 39)

plt.figure(figsize=(12,5))
plt.plot(weeks, ranking_errors, marker='o')
plt.title("Ranking Error vs Matchday (MD_1 → MD_38)")
plt.xlabel("Matchday")
plt.ylabel("Relative 2-Norm Error")
plt.grid()
plt.show()

plt.figure(figsize=(12,5))
plt.plot(weeks, accuracies, marker='o', color='green')
plt.title("Top-4 Accuracy vs Matchday")
plt.xlabel("Matchday")
plt.ylabel("Accuracy")
plt.grid()
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import xgboost as xgb
import tensorflow as tf


true_rank = weekly_df["true_rank"].values
top4 = weekly_df["top4"].values

ranking_results = {
    "RF": [],
    "XGBoost": [],
    "DecisionTree": [],
    "NN": []
}

top4_results = {
    "RF": [],
    "XGBoost": [],
    "DecisionTree": [],
    "NN": []
}

def ranking_error(pred, true):
    return np.linalg.norm(pred - true) / np.linalg.norm(true)


for k in range(1, 39):

    features = [f"MD_{i}" for i in range(1, k+1)]
    X = weekly_df[features].values
    input_dim = X.shape[1]

    rf_r = RandomForestRegressor(n_estimators=500, random_state=42)
    rf_r.fit(X, true_rank)
    pred_rf_r = rf_r.predict(X)
    ranking_results["RF"].append(ranking_error(pred_rf_r, true_rank))

    xgb_r = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05)
    xgb_r.fit(X, true_rank)
    pred_xgb_r = xgb_r.predict(X)
    ranking_results["XGBoost"].append(ranking_error(pred_xgb_r, true_rank))

    dt_r = DecisionTreeRegressor(random_state=42)
    dt_r.fit(X, true_rank)
    pred_dt_r = dt_r.predict(X)
    ranking_results["DecisionTree"].append(ranking_error(pred_dt_r, true_rank))

    rf_c = RandomForestClassifier(n_estimators=500, random_state=42)
    rf_c.fit(X, top4)
    pred_rf_c = rf_c.predict(X)
    top4_results["RF"].append(accuracy_score(top4, pred_rf_c))

    xgb_c = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05)
    xgb_c.fit(X, top4)
    pred_xgb_c = xgb_c.predict(X)
    top4_results["XGBoost"].append(accuracy_score(top4, pred_xgb_c))

    dt_c = DecisionTreeClassifier(random_state=42)
    dt_c.fit(X, top4)
    pred_dt_c = dt_c.predict(X)
    top4_results["DecisionTree"].append(accuracy_score(top4, pred_dt_c))

weeks = np.arange(1, 39)

plt.figure(figsize=(12,6))
for model in ranking_results:
    plt.plot(weeks, ranking_results[model], marker='o', label=model)

plt.title("Ranking Error vs Matchday (Multiple Models)")
plt.xlabel("Matchday")
plt.ylabel("Relative 2-Norm Error")
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
for model in top4_results:
    plt.plot(weeks, top4_results[model], marker='o', label=model)

plt.title("Top-4 Accuracy vs Matchday (Multiple Models)")
plt.xlabel("Matchday")
plt.ylabel("Accuracy")
plt.grid()
plt.legend()
plt.show()
