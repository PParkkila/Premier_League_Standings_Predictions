import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression

import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier


matches_all = pd.read_csv("./standings/Premier_1993-2022.csv")
matches_all["Match Date"] = pd.to_datetime(matches_all["Date"], dayfirst=True)



def build_cumulative_table(df):
    df = df.copy()

    def pts(diff):
        return 3 if diff > 0 else (1 if diff == 0 else 0)

    df["HomePts"] = (df["HomeGoals"] - df["AwayGoals"]).apply(pts)
    df["AwayPts"] = (df["AwayGoals"] - df["HomeGoals"]).apply(pts)

    home = df[["HomeTeam", "HomePts", "Match Date"]].rename(
        columns={"HomeTeam": "team", "HomePts": "points"}
    )
    away = df[["AwayTeam", "AwayPts", "Match Date"]].rename(
        columns={"AwayTeam": "team", "AwayPts": "points"}
    )

    long = pd.concat([home, away], ignore_index=True).sort_values(["team", "Match Date"])
    long["matchday"] = long.groupby("team").cumcount() + 1
    long["cum_points"] = long.groupby("team")["points"].cumsum()

    wide = long.pivot(index="team", columns="matchday", values="cum_points")
    wide = wide.ffill(axis=1).fillna(0)
    wide.columns = [f"MD_{int(c)}" for c in wide.columns]

    return wide.reset_index()


train_matches = matches_all[matches_all["Season_End_Year"] == 2020]
train_df = build_cumulative_table(train_matches)

md_cols = sorted([c for c in train_df.columns if c.startswith("MD_")],
                 key=lambda x: int(x.split("_")[1]))
max_md = len(md_cols)
print(f"Training season — detected {max_md} matchdays.")

train_df["final_points"] = train_df[md_cols[-1]]
train_df = train_df.sort_values("final_points", ascending=False).reset_index(drop=True)
train_df["true_rank"] = train_df.index + 1
train_df["top4"] = (train_df["true_rank"] <= 4).astype(int)

training_teams = set(train_df["team"])
true_rank = train_df["true_rank"].values
top4 = train_df["top4"].values


def safe_train(model, X, y):
    try:
        if len(set(y)) < 2:
            return None
        model.fit(X, y)
        return model
    except:
        return None


ranking_models = {
    "RandomForest": RandomForestRegressor(n_estimators=500, random_state=42),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        objective="reg:squarederror", random_state=42
    ),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "LinearRegression": LinearRegression(),
    "CatBoost": CatBoostRegressor(
        depth=6, learning_rate=0.05, iterations=400,
        verbose=False, random_seed=42
    ),
}

classification_models = {
    "RandomForest": RandomForestClassifier(n_estimators=500, random_state=42),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        use_label_encoder=False, eval_metric="logloss", random_state=42
    ),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "CatBoost": CatBoostClassifier(
        depth=6, learning_rate=0.05, iterations=400,
        verbose=False, random_seed=42
    ),
}


X_train = train_df[md_cols].values

for name in ranking_models:
    ranking_models[name] = safe_train(ranking_models[name], X_train, true_rank)

for name in classification_models:
    classification_models[name] = safe_train(classification_models[name], X_train, top4)

print("Finished training models.")


prediction_results = {}

for season in range(2008, 2019):

    season_matches = matches_all[matches_all["Season_End_Year"] == season]
    season_df = build_cumulative_table(season_matches)

    season_df["is_in_training"] = season_df["team"].isin(training_teams)

    for col in season_df.columns:
        if col.startswith("MD_"):
            season_df.loc[~season_df["is_in_training"], col] = 0

    for md in md_cols:
        if md not in season_df.columns:
            season_df[md] = 0

    season_df = season_df[["team"] + md_cols]
    X_test = season_df[md_cols].values

    preds = {"teams": season_df["team"].tolist()}

    for name, model in ranking_models.items():
        preds[name + "_rank"] = None if model is None else model.predict(X_test)

    for name, model in classification_models.items():
        preds[name + "_top4"] = None if model is None else model.predict(X_test)

    prediction_results[season] = preds

print("Finished predictions 2008–2018.")


top4_accuracy_by_model = {m: [] for m in classification_models}

for season in range(2008, 2019):

    season_matches = matches_all[matches_all["Season_End_Year"] == season]
    true_df = build_cumulative_table(season_matches)

    mdc = sorted([c for c in true_df.columns if c.startswith("MD_")],
                 key=lambda x: int(x.split("_")[1]))

    true_df["final_points"] = true_df[mdc[-1]]
    true_df = true_df.sort_values("final_points", ascending=False).reset_index(drop=True)
    true_df["true_top4"] = (true_df.index < 4).astype(int)

    true_map = dict(zip(true_df["team"], true_df["true_top4"]))

    preds = prediction_results[season]
    teams = preds["teams"]

    for model_name in classification_models.keys():

        model_pred = preds[model_name + "_top4"]

        if model_pred is None:
            top4_accuracy_by_model[model_name].append(np.nan)
            continue

        y_true = [true_map.get(t, 0) for t in teams]
        y_pred = model_pred

        acc = accuracy_score(y_true, y_pred)
        top4_accuracy_by_model[model_name].append(acc)

print("Top-4 accuracy computed for all seasons.")

mean_accuracy = {m: np.nanmean(acc) for m, acc in top4_accuracy_by_model.items()}
print("\nMean Top-4 Accuracy (2008–2018):")
for m, v in mean_accuracy.items():
    print(f"{m}: {v:.4f}")

plt.figure(figsize=(14, 7))
seasons = np.arange(2008, 2019)

for model_name, acc_list in top4_accuracy_by_model.items():
    plt.plot(seasons, acc_list, marker='o', label=model_name)

plt.title("Top-4 Prediction Accuracy by Season (Training on 2019–2020 Only)")
plt.xlabel("Season End Year")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.show()

dtc = classification_models["DecisionTree"]
if dtc is not None:
    plt.figure(figsize=(22, 12))
    tree.plot_tree(dtc, feature_names=md_cols, class_names=["No", "Top4"],
                   filled=True, rounded=True)
    plt.title("Decision Tree Classifier — Top-4 Prediction")
    plt.show()

dtr = ranking_models["DecisionTree"]
if dtr is not None:
    plt.figure(figsize=(22, 12))
    tree.plot_tree(dtr, feature_names=md_cols, filled=True, rounded=True)
    plt.title("Decision Tree Regressor — Ranking Prediction")
    plt.show()

print("Decision Tree visualizations complete.")
