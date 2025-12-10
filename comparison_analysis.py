import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression

import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier

matches = pd.read_csv("./standings/Premier_1993-2022.csv")
matches = matches[matches["Season_End_Year"] == 2020].copy()

matches["Match Date"] = pd.to_datetime(matches["Date"], dayfirst=True)


def pts(diff: int) -> int:
    if diff > 0:
        return 3
    if diff < 0:
        return 0
    return 1


matches["HomePts"] = (matches["HomeGoals"] - matches["AwayGoals"]).apply(pts)
matches["AwayPts"] = (matches["AwayGoals"] - matches["HomeGoals"]).apply(pts)

home = matches[["HomeTeam", "HomePts", "Match Date"]].rename(
    columns={"HomeTeam": "team", "HomePts": "points"}
)
away = matches[["AwayTeam", "AwayPts", "Match Date"]].rename(
    columns={"AwayTeam": "team", "AwayPts": "points"}
)

long = pd.concat([home, away], ignore_index=True)
long = long.sort_values(["team", "Match Date"])

long["matchday"] = long.groupby("team").cumcount() + 1
long["cum_points"] = long.groupby("team")["points"].cumsum()

wide_pts = long.pivot(index="team", columns="matchday", values="cum_points")
wide_pts = wide_pts.ffill(axis=1).fillna(0)
wide_pts.columns = [f"MD_{int(md)}" for md in wide_pts.columns]
weekly_df = wide_pts.reset_index()

md_cols = sorted(
    [c for c in weekly_df.columns if c.startswith("MD_")],
    key=lambda x: int(x.split("_")[1])
)
max_md = len(md_cols)
print(f"Detected {max_md} matchdays: {md_cols[0]} .. {md_cols[-1]}")

sent_df = pd.read_csv("./statistics/social_media_sentiment_stats.csv")

if "team" not in sent_df.columns and "file_name" in sent_df.columns:
    sent_df = sent_df.rename(columns={"file_name": "team"})

sent_df = sent_df.copy()
sent_df.set_index("team", inplace=True)

name_fix = {
    "TottenhamHotspur": "Tottenham Hotspur",
    "ManchesterUnited": "Manchester United",
    "ManchesterCity": "Manchester City",
    "LeicesterCity": "Leicester City",
    "CrystalPalace": "Crystal Palace",
}
sent_df.rename(index=name_fix, inplace=True)

yw_re = re.compile(r"^\d{4}_\d+$")
sent_cols = [c for c in sent_df.columns if yw_re.match(str(c))]
sent_df = sent_df[sent_cols]

def parse_yearweek(col: str):
    year_str, week_str = col.split("_")
    return int(year_str), int(week_str)

sent_week_keys = {col: parse_yearweek(col) for col in sent_cols}
sent_weeks_sorted = sorted(
    [(y, w, col) for col, (y, w) in sent_week_keys.items()],
    key=lambda t: (t[0], t[1])
)

iso = long["Match Date"].dt.isocalendar()
long["year"] = iso.year.astype(int)
long["week"] = iso.week.astype(int)
long["year_week"] = long["year"].astype(str) + "_" + long["week"].astype(str)


def get_latest_sentiment_col(y: int, w: int):
    candidate = None
    for sy, sw, col in sent_weeks_sorted:
        if (sy < y) or (sy == y and sw <= w):
            candidate = col
        else:
            break
    return candidate


sent_values = []
for _, row in long.iterrows():
    y = int(row["year"])
    w = int(row["week"])
    t = row["team"]

    col = get_latest_sentiment_col(y, w)
    if col is None or t not in sent_df.index:
        sent_values.append(np.nan)
    else:
        sent_values.append(sent_df.loc[t, col])

long["sentiment"] = sent_values

sent_wide = long.pivot(index="team", columns="matchday", values="sentiment")
sent_wide = sent_wide.ffill(axis=1).fillna(0)  # carry forward sentiment
sent_wide.columns = [f"SMD_{int(md)}" for md in sent_wide.columns]

sent_wide = sent_wide.reindex(weekly_df["team"]).fillna(0)

weekly_with_sent = weekly_df.merge(sent_wide.reset_index(), on="team", how="left")

md_cols = sorted(
    [c for c in weekly_with_sent.columns if c.startswith("MD_")],
    key=lambda x: int(x.split("_")[1])
)
smd_cols = sorted(
    [c for c in weekly_with_sent.columns if c.startswith("SMD_")],
    key=lambda x: int(x.split("_")[1])
)
max_md = len(md_cols)
print(f"Sentiment matchday columns detected: {smd_cols[:5]} ...")

weekly_with_sent["final_points"] = weekly_with_sent[md_cols[-1]]
weekly_with_sent = weekly_with_sent.sort_values(
    "final_points", ascending=False
).reset_index(drop=True)

weekly_with_sent["true_rank"] = weekly_with_sent.index + 1
weekly_with_sent["top4"] = (weekly_with_sent["true_rank"] <= 4).astype(int)

true_rank = weekly_with_sent["true_rank"].values
top4 = weekly_with_sent["top4"].values

def ranking_error(pred, true):
    return np.linalg.norm(pred - true) / np.linalg.norm(true)


def safe_train_and_predict(model, X, y, metric_fn):
    # If labels are constant, some models can't train
    if len(set(y)) < 2:
        return np.nan
    try:
        model.fit(X, y)
        pred = model.predict(X)
        return metric_fn(pred, y)
    except Exception:
        return np.nan
    

ranking_models = {
    "RandomForest": lambda: RandomForestRegressor(
        n_estimators=500, random_state=42
    ),
    "XGBoost": lambda: xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42,
    ),
    "DecisionTree": lambda: DecisionTreeRegressor(random_state=42),
    "LinearRegression": lambda: LinearRegression(),
    "CatBoost": lambda: CatBoostRegressor(
        depth=6, learning_rate=0.05, iterations=400,
        verbose=False, random_seed=42,
    ),
}

classification_models = {
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=500, random_state=42
    ),
    "XGBoost": lambda: xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    ),
    "DecisionTree": lambda: DecisionTreeClassifier(random_state=42),
    "LogisticRegression": lambda: LogisticRegression(max_iter=1000),
    "CatBoost": lambda: CatBoostClassifier(
        depth=6,
        learning_rate=0.05,
        iterations=400,
        verbose=False,
        random_seed=42,
    ),
}

ranking_results_points = {name: [] for name in ranking_models}
ranking_results_both = {name: [] for name in ranking_models}

top4_results_points = {name: [] for name in classification_models}
top4_results_both = {name: [] for name in classification_models}

for k in range(1, max_md + 1):
    feats_pts = [f"MD_{i}" for i in range(1, k + 1)]
    X_pts = weekly_with_sent[feats_pts].values

    feats_smd = [f"SMD_{i}" for i in range(1, k + 1)]
    feats_smd = [f for f in feats_smd if f in weekly_with_sent.columns]
    X_both = weekly_with_sent[feats_pts + feats_smd].values

    for name, model_fn in ranking_models.items():
        model = model_fn()
        ranking_results_points[name].append(
            safe_train_and_predict(model, X_pts, true_rank, ranking_error)
        )

        model = model_fn()
        ranking_results_both[name].append(
            safe_train_and_predict(model, X_both, true_rank, ranking_error)
        )

    for name, model_fn in classification_models.items():
        model = model_fn()
        top4_results_points[name].append(
            safe_train_and_predict(
                model, X_pts, top4, lambda p, t: accuracy_score(t, p)
            )
        )

        model = model_fn()
        top4_results_both[name].append(
            safe_train_and_predict(
                model, X_both, top4, lambda p, t: accuracy_score(t, p)
            )
        )

weeks = np.arange(1, max_md + 1)

plt.figure(figsize=(13, 6))
for name, vals in ranking_results_points.items():
    plt.plot(weeks, vals, marker="o", label=name)
plt.title("Ranking Error vs Matchday — Points Only")
plt.xlabel("Matchday")
plt.ylabel("Relative 2-Norm Error")
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(13, 6))
for name, vals in ranking_results_both.items():
    plt.plot(weeks, vals, marker="o", label=name)
plt.title("Ranking Error vs Matchday — Points + Sentiment")
plt.xlabel("Matchday")
plt.ylabel("Relative 2-Norm Error")
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(13, 6))
for name, vals in top4_results_points.items():
    plt.plot(weeks, vals, marker="o", label=name)
plt.title("Top-4 Accuracy vs Matchday — Points Only")
plt.xlabel("Matchday")
plt.ylabel("Accuracy")
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(13, 6))
for name, vals in top4_results_both.items():
    plt.plot(weeks, vals, marker="o", label=name)
plt.title("Top-4 Accuracy vs Matchday — Points + Sentiment")
plt.xlabel("Matchday")
plt.ylabel("Accuracy")
plt.grid()
plt.legend()
plt.show()

rf_rank_pts = ranking_results_points["RandomForest"]
rf_rank_both = ranking_results_both["RandomForest"]

plt.figure(figsize=(13, 6))
plt.plot(weeks, rf_rank_pts, marker="o", label="RF — Points Only")
plt.plot(weeks, rf_rank_both, marker="o", label="RF — Points + Sentiment")
plt.title("RandomForest Ranking Error vs Matchday (With vs Without Sentiment)")
plt.xlabel("Matchday")
plt.ylabel("Relative 2-Norm Error")
plt.grid()
plt.legend()
plt.show()

rf_acc_pts = top4_results_points["RandomForest"]
rf_acc_both = top4_results_both["RandomForest"]

plt.figure(figsize=(13, 6))
plt.plot(weeks, rf_acc_pts, marker="o", label="RF — Points Only")
plt.plot(weeks, rf_acc_both, marker="o", label="RF — Points + Sentiment")
plt.title("RandomForest Top-4 Accuracy vs Matchday (With vs Without Sentiment)")
plt.xlabel("Matchday")
plt.ylabel("Accuracy")
plt.grid()
plt.legend()
plt.show()

print("Finished training points-only vs points+sentiment models.")
