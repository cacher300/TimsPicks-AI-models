import os, glob
import numpy as np
import pandas as pd
from joblib import dump
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report, f1_score
)
from sklearn.inspection import permutation_importance

BOX_DIR = "boxscores_out"               
PLAYER_STATS_CSV = "out/season_stats.csv"    
STANDINGS_CSV = "out/standings.csv"           
CUTOFF_DATE = None                        
MODEL_OUT = "score_xgb_calibrated.joblib"
FEATURES_OUT = "score_xgb_features.csv"
RAW_IMPORTANCE_OUT = "xgb_feature_importance.csv"
PERM_IMPORTANCE_OUT = "perm_feature_importance.csv"

def atoi_to_seconds(s):
    if pd.isna(s):
        return np.nan
    parts = str(s).split(":")
    try:
        parts = [float(x) for x in parts]
    except:
        return np.nan
    if len(parts) == 2:
        m, sec = parts
        return m*60 + sec
    if len(parts) == 3:
        h, m, sec = parts
        return h*3600 + m*60 + sec
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
    box = load_boxscores(BOX_DIR)
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

def train_calibrated_xgb(X_train, y_train):
    pos = y_train.sum()
    neg = len(y_train) - pos
    pos_w = (neg / max(pos, 1)) if pos > 0 else 1.0  # handle imbalance

    base_xgb = XGBClassifier(
        n_estimators=200,          
        max_depth=4,              
        learning_rate=0.05,        
        min_child_weight=3,        
        subsample=0.8,             
        colsample_bytree=0.8,      
        gamma=0.5,                
        reg_lambda=2.0,            
        reg_alpha=0.1,             
        scale_pos_weight=pos_w,    
        tree_method="hist",
        n_jobs=-1,
        random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cal = CalibratedClassifierCV(base_xgb, method="isotonic", cv=cv)
    cal.fit(X_train, y_train)
    return cal, pos_w


def evaluate(cal, X_test, y_test):
    proba = cal.predict_proba(X_test)[:, 1]
    # guard AUC for single-class y
    auc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else np.nan
    ap = average_precision_score(y_test, proba)
    ths = np.linspace(0.05, 0.8, 40)
    best_f1, best_t = max((f1_score(y_test, proba >= t), t) for t in ths)
    if np.isnan(auc):
        print(f"AUC: NA")
    else:
        print(f"AUC: {auc:.3f}")
    print(f"Avg Precision (PR AUC): {ap:.3f}")
    print(f"Best threshold: {best_t:.2f}")
    print(classification_report(y_test, (proba >= best_t).astype(int), digits=3))
    return proba, best_t

def export_importances(cal, feature_names):
    """
    Aggregate tree feature importances across calibrated folds.
    Works for XGBoost estimators accessible via calibrated_classifiers_.
    """
    trees = []
    if hasattr(cal, "calibrated_classifiers_"):
        for cc in cal.calibrated_classifiers_:
            est = getattr(cc, "estimator", None)
            if est is not None and hasattr(est, "feature_importances_"):
                trees.append(est.feature_importances_)
    if not trees:
        print("No fitted model importances available; skipping RAW importance export.")
        return pd.Series(dtype=float)

    arr = np.vstack(trees)
    mean_imp = arr.mean(axis=0)
    s = pd.Series(mean_imp, index=feature_names).sort_values(ascending=False)
    s.to_frame("importance").to_csv(RAW_IMPORTANCE_OUT)
    print(f"Saved raw importances -> {RAW_IMPORTANCE_OUT}")
    return s

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

    cal, pos_w = train_calibrated_xgb(X_train, y_train)
    proba, best_t = evaluate(cal, X_test, y_test)

    dump({
        "model": cal,
        "feature_names": feat_names,
        "train_medians": medians,
        "best_threshold": best_t
    }, MODEL_OUT)
    pd.Series(feat_names, name="feature").to_csv(FEATURES_OUT, index=False)
    print(f"Saved model -> {MODEL_OUT}")
    print(f"Saved features -> {FEATURES_OUT}")

    export_importances(cal, feat_names)
    export_permutation_importance(cal, X_test, y_test, feat_names)

if __name__ == "__main__":
    main()
