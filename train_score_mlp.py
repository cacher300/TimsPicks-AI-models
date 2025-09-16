import os, glob
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report, f1_score
)
from sklearn.inspection import permutation_importance

BOX_DIR = "boxscores_out"                 
PLAYER_STATS_CSV = "out/season_stats.csv"
STANDINGS_CSV = "out/standings.csv"       
CUTOFF_DATE = None                        
MODEL_OUT = "score_mlp_calibrated.joblib"
FEATURES_OUT = "score_mlp_features.csv"
PERM_IMPORTANCE_OUT = "mlp_perm_feature_importance.csv"

def atoi_to_seconds(s):
    if pd.isna(s): return np.nan
    parts = str(s).split(":")
    try:
        parts = [float(x) for x in parts]
    except:
        return np.nan
    if len(parts) == 2:
        m, sec = parts; return m*60 + sec
    if len(parts) == 3:
        h, m, sec = parts; return h*3600 + m*60 + sec
    return np.nan

def load_boxscores(box_dir):
    paths = sorted(glob.glob(os.path.join(box_dir, "**", "*.csv"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No CSVs found under {box_dir}")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        rename = {c.lower(): c for c in df.columns}
        def pick(name): return rename.get(name, name)
        df = df.rename(columns={
            pick("game_id"): "game_id",
            pick("opponent"): "opponent",
            pick("player_id"): "player_id",
            pick("player_name"): "player_name",
            pick("scored"): "scored"
        })
        frames.append(df[["game_id","opponent","player_id","player_name","scored"]])
    out = pd.concat(frames, ignore_index=True)
    out["scored"] = pd.to_numeric(out["scored"], errors="coerce").fillna(0).astype(int)
    out["opponent"] = out["opponent"].astype(str).str.strip().str.upper()
    out["player_id"] = out["player_id"].astype(str).str.strip().str.lower()
    return out

def load_player_stats(path):
    s = pd.read_csv(path)
    if "PlayerID" in s.columns: s = s.rename(columns={"PlayerID":"player_id"})
    if "player_id" not in s.columns:
        raise ValueError("Player stats must contain PlayerID or player_id")
    s["player_id"] = s["player_id"].astype(str).str.strip().str.lower()

    if "ATOI" in s.columns: s["ATOI_sec"] = s["ATOI"].apply(atoi_to_seconds)
    if "TOI" in s.columns:  s["TOI_sec"]  = s["TOI"].apply(atoi_to_seconds)

    drop = {"Player","PlayerURL","ATOI","TOI"}
    keep_cols = [c for c in s.columns if c == "player_id" or c not in drop]
    s = s[keep_cols]

    for c in s.columns:
        if c == "player_id": continue
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s = s.drop_duplicates(subset=["player_id"])
    return s

def load_standings(path):
    st = pd.read_csv(path)
    st = st.rename(columns={c: c.strip() for c in st.columns})
    if not {"Rk","Team"}.issubset(st.columns):
        raise ValueError("Standings must have columns Rk and Team")
    st["Team"] = st["Team"].astype(str).str.strip().str.upper()
    st["Rk"] = pd.to_numeric(st["Rk"], errors="coerce")
    maxrk = st["Rk"].max()
    st["opp_strength"] = 1.0 - (st["Rk"] - 1) / (maxrk - 1) 
    return st[["Team","Rk","opp_strength"]]

def engineer_opponent_features(df):
    df["opp_rank"] = pd.to_numeric(df["Rk"], errors="coerce")
    df["opp_strength"] = pd.to_numeric(df.get("opp_strength", 1.0), errors="coerce").fillna(1.0)
    df["opp_tier_top5"] = (df["opp_rank"] <= 5).astype(int)
    df["opp_tier_6_15"] = ((df["opp_rank"] > 5) & (df["opp_rank"] <= 15)).astype(int)
    df["opp_tier_bottom"] = (df["opp_rank"] > 15).astype(int)
    for col in ["SH%", "SOG"]:
        if col in df.columns:
            df[f"{col}_x_opp"] = pd.to_numeric(df[col], errors="coerce") * df["opp_strength"]
    return df

def build_dataset():
    box = load_boxscores(ABOX_DIR if (ABOX_DIR:=BOX_DIR) else BOX_DIR)  
    pstats = load_player_stats(PLAYER_STATS_CSV)
    stand = load_standings(STANDINGS_CSV)
    df = box.merge(pstats, on="player_id", how="left")
    df = df.merge(stand, left_on="opponent", right_on="Team", how="left").drop(columns=["Team"])
    df = engineer_opponent_features(df)
    df["game_dt"] = pd.to_datetime(df["game_id"].astype(str).str[:8], format="%Y%m%d", errors="coerce")
    return df

def time_split(df, cutoff_date=None):
    if cutoff_date is None:
        cutoff_date = df["game_dt"].quantile(0.8)
    else:
        cutoff_date = pd.Timestamp(cutoff_date)
    train = df[df["game_dt"] < cutoff_date].copy()
    test  = df[df["game_dt"] >= cutoff_date].copy()
    drop_cols = {"game_id","game_dt","opponent","player_id","player_name","scored","Rk"}
    num_cols = [c for c in df.columns if c not in drop_cols]
    X_train = train[num_cols].apply(pd.to_numeric, errors="coerce")
    X_test  = test[num_cols].apply(pd.to_numeric, errors="coerce")
    med = X_train.median(numeric_only=True)
    X_train = X_train.fillna(med)
    X_test  = X_test.fillna(med)
    y_train = train["scored"].astype(int).values
    y_test  = test["scored"].astype(int).values
    return X_train, X_test, y_train, y_test, num_cols, med, cutoff_date


def tune_mlp_params(X_train, y_train, pos_weight_multiplier=1.0, random_state=42):
    pos = int(y_train.sum()); neg = int(len(y_train) - pos)
    base_pos_w = (neg / max(pos, 1)) if pos > 0 else 1.0
    pos_w = base_pos_w * float(pos_weight_multiplier)
    sample_w = np.where(y_train == 1, pos_w, 1.0)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            batch_size=256,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
            shuffle=True,
            random_state=random_state
        ))
    ])

    param_dist = {
        "mlp__hidden_layer_sizes": [(64,32), (128,64), (64,64), (256,128,64)],
        "mlp__alpha": [1e-5, 1e-4, 5e-4, 1e-3, 3e-3, 1e-2],
        "mlp__learning_rate_init": [3e-4, 1e-3, 3e-3],
        "mlp__batch_size": [128, 256, 512],
        "mlp__activation": ["relu", "tanh"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=25,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=random_state,
        refit=True
    )
    search.fit(X_train, y_train, mlp__sample_weight=sample_w)
    return search.best_estimator_, search.best_params_, pos_w

def train_calibrated_mlp_tuned(X_train, y_train, pos_weight_multiplier=1.0, random_state=42):
    best_pipe, best_params, pos_w = tune_mlp_params(
        X_train, y_train, pos_weight_multiplier=pos_weight_multiplier, random_state=random_state
    )
    cv_cal = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cal = CalibratedClassifierCV(best_pipe, method="isotonic", cv=cv_cal)

    pos = int(y_train.sum()); neg = int(len(y_train) - pos)
    base_pos_w = (neg / max(pos, 1)) if pos > 0 else 1.0
    pos_w_full = base_pos_w * float(pos_weight_multiplier)
    sample_w_full = np.where(y_train == 1, pos_w_full, 1.0)

    cal.fit(X_train, y_train, mlp__sample_weight=sample_w_full)
    return cal, best_params, pos_w


def evaluate(cal, X_test, y_test):
    proba = cal.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else np.nan
    ap = average_precision_score(y_test, proba)
    ths = np.linspace(0.05, 0.8, 40)
    best_f1, best_t = max((f1_score(y_test, proba >= t), t) for t in ths)
    print(f"AUC: {auc:.3f}" if not np.isnan(auc) else "AUC: NA")
    print(f"Avg Precision (PR AUC): {ap:.3f}")
    print(f"Best threshold: {best_t:.2f}")
    print(classification_report(y_test, (proba >= best_t).astype(int), digits=3))
    return proba, best_t

def export_permutation_importance(cal, X_test, y_test, feature_names):
    perm = permutation_importance(cal, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    imp = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)
    imp.to_frame("perm_importance_mean").to_csv(PERM_IMPORTANCE_OUT)
    print(f"Saved permutation importances -> {PERM_IMPORTANCE_OUT}")
    return imp


def main():
    df = build_dataset()
    X_train, X_test, y_train, y_test, feat_names, medians, cutoff_used = time_split(df, CUTOFF_DATE)
    print(f"Train size: {len(y_train)}  Test size: {len(y_test)}  Cutoff: {cutoff_used.date()}")

    cal, best_params, pos_w = train_calibrated_mlp_tuned(X_train, y_train, pos_weight_multiplier=1.0)
    print("Best MLP params:", best_params)

    proba, best_t = evaluate(cal, X_test, y_test)

    dump({
        "model": cal,
        "feature_names": feat_names,
        "train_medians": medians,
        "best_threshold": best_t,
        "best_params": best_params
    }, MODEL_OUT)
    pd.Series(feat_names, name="feature").to_csv(FEATURES_OUT, index=False)
    print(f"Saved model -> {MODEL_OUT}")
    print(f"Saved features -> {FEATURES_OUT}")

    export_permutation_importance(cal, X_test, y_test, feat_names)

if __name__ == "__main__":
    main()
