#WITH THE NEW FEATURES

"""
TWEAKED XGBoost baseline for the CSI500 stock-selection competition.

Pipeline
--------
1. Load data/prices.parquet
2. Build features + 5-day forward target (features.py)
3. Train XGBoost on all but the last `EMBARGO_DAYS` training rows
4. Validate on those held-out rows (reports rank IC as sanity check)
5. Predict on the most recent date
6. Build a portfolio: top-K names, score-weighted with the 10% cap

Usage
-----
  python baseline_xgboost.py                       # predict from latest data
  python baseline_xgboost.py --as-of 20260503      # predict as of a given date
  python baseline_xgboost.py --top-k 50 --out submissions/week1.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

from features import (
    FEATURE_COLUMNS, TARGET_COLUMN, FORWARD_HORIZON,
    build_features, training_frame, prediction_frame,
    load_sector_map, load_margin_features, load_fin_ind_feats,
    add_sector_features, add_margin_features, add_quality_value_features,
    add_market_regime_feature,
    get_market_regime, build_index_df
)

from stockLSTM import train_lstm, predict_lstm

DATA_DIR = Path(__file__).parent / "data"
VAL_DAYS = 10               # number of trading days in the validation window
EMBARGO_DAYS = 5            # gap between train end and val start (>= FORWARD_HORIZON
                            # so training targets don't reach into val dates)
MIN_STOCKS = 30             # rule: portfolio must hold >= 30 names
MAX_WEIGHT = 0.10           # rule: per-stock weight cap
DEFAULT_TOP_K = 75          # baseline picks top-50 by predicted score
include_LSTM = True


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame, args) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample,
        min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda,
        tree_method="hist",
        n_jobs=-1,
        early_stopping_rounds=30,
    )
    model.fit(
        train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN],
        eval_set=[(val_df[FEATURE_COLUMNS], val_df[TARGET_COLUMN])],
        verbose=False,
    )
    return model


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray) -> float:
    """Daily cross-sectional Spearman correlation, averaged over dates."""
    ics = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < 20:
            continue
        rho, _ = spearmanr(y_true[mask], y_pred[mask])
        if not np.isnan(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else float("nan")


def build_portfolio(scores, alpha, vols=None, top_k=DEFAULT_TOP_K):
    """
    alpha: weight on score ranks vs inverse vol ranks (0=pure low-vol, 1=pure score)
    """

    if top_k < MIN_STOCKS:
        raise ValueError(f"top_k must be >= {MIN_STOCKS} (rule)")
    chosen = scores.sort_values(ascending=False).head(top_k).copy()

    if vols is not None:
        chosen_vols = vols.reindex(chosen.index).fillna(vols.median())
        
        # rank scores within chosen (percentile, higher=better)
        score_ranks = chosen.rank(pct=True)
        
        # rank vols inversely within chosen (percentile, lower vol=higher rank)
        vol_ranks = 1 - chosen_vols.rank(pct=True)
        
        # blend
        combined = alpha * score_ranks + (1 - alpha) * vol_ranks
        w = combined / combined.sum()
    else:
        ranks = np.arange(top_k, 0, -1, dtype=float)
        w = pd.Series(ranks / ranks.sum(), index=chosen.index)

    # iteratively cap and redistribute
    for _ in range(500):
        over = w > MAX_WEIGHT
        if not over.any():
            break
        excess = (w[over] - MAX_WEIGHT).sum()
        w[over] = MAX_WEIGHT
        free = ~over
        if not free.any():
            w = pd.Series(1.0 / top_k, index=chosen.index)
            break
        w[free] += excess * w[free] / w[free].sum()

    w = w / w.sum()

    assert abs(w.sum() - 1.0) < 1e-4, f"weights sum to {w.sum()}"
    assert (w <= MAX_WEIGHT + 1e-6).all(), f"cap violated max={w.max():.8f}"
    assert (w > 0).sum() >= MIN_STOCKS, f"too few names: {(w>0).sum()}"
    return w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--as-of", default=None, help="YYYYMMDD; defaults to latest date in data")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample", type=float, default=0.6)
    p.add_argument("--min-child-weight", type=int, default=10)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    p.add_argument("--alpha", type=float,default=0.45)
    p.add_argument("--out", default="submission.csv")
    args = p.parse_args()


    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    print(f"   {len(prices):,} rows, {prices['stock_code'].nunique()} stocks, "
          f"dates {prices['date'].min().date()} to {prices['date'].max().date()}")

    print(">> Building features")
    panel = build_features(prices)

    """ ADDED sector + margin features -->"""
    sector_map = load_sector_map()
    margin = load_margin_features()
    fin_feats = load_fin_ind_feats()
    
    panel = add_sector_features(panel, sector_map)
    panel = add_margin_features(panel, margin)
    panel = add_market_regime_feature(panel)
    panel = add_quality_value_features(panel, fin_df=fin_feats)
    
    """excluding fin_features"""
    #fin_df = load_fin_ind_feats()
    #panel = add_quality_value_features(panel, fin_df)
    # Bound training data so backtesting with --as-of doesn't leak future rows.
    # Training uses features from date t with target = close(t+FORWARD_HORIZON),
    # so we cap training dates at as_of - FORWARD_HORIZON trading days.
    as_of_ts = pd.Timestamp(args.as_of) if args.as_of else panel["date"].max()
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts)))
    cutoff_idx = max(0, as_of_idx - FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])
    train_pool = training_frame(panel, max_date=train_cutoff)

    # Time-based split with embargo:
    #   [ ... train ... | embargo (discarded) | val (last VAL_DAYS) ]
    # The embargo prevents training labels (5-day forward) from reaching into
    # dates whose prices also feed the validation features.
    all_dates = np.sort(train_pool["date"].unique())
    if len(all_dates) < VAL_DAYS + EMBARGO_DAYS + 20:
        raise RuntimeError("Not enough dates to train; download more history.")
    val_start = pd.Timestamp(all_dates[-VAL_DAYS])
    train_end = pd.Timestamp(all_dates[-(VAL_DAYS + EMBARGO_DAYS + 1)])
    train_df = train_pool[train_pool["date"] <= train_end]
    val_df = train_pool[train_pool["date"] >= val_start]
    print(f"   train: {len(train_df):,} rows up to {train_end.date()}")
    print(f"   embargo: {EMBARGO_DAYS} trading days (discarded)")
    print(f"   val:   {len(val_df):,} rows from {val_start.date()}")

    print(">> Training XGBoost")
    model = train_model(train_df, val_df, args)

    val_pred = model.predict(val_df[FEATURE_COLUMNS])
    ic = rank_ic(val_df[TARGET_COLUMN].to_numpy(), val_pred, val_df["date"].to_numpy())
    print(f"   validation rank IC: {ic:.4f}")
    # adding this tagged version for shell parsing:
    print(f"RANK_IC={ic:.4f}")

    print(">> Predicting portfolio")
    pred_df = prediction_frame(panel, as_of=args.as_of)
    if pred_df.empty:
        raise RuntimeError(f"No rows available for as_of={args.as_of}. Check data.")
    pred_date = pred_df["date"].iloc[0]
    print(f"   as of {pred_date.date()}, scoring {len(pred_df)} stocks")

    pred_df = pred_df.assign(score=model.predict(pred_df[FEATURE_COLUMNS]))
    scores = pred_df.set_index("stock_code")["score"]
    vols = pred_df.set_index("stock_code")["vol_20d"]

     #if including LSTM
    if include_LSTM:
        model_LSTM = train_lstm(
            train_df, val_df, 
            FEATURE_COLUMNS, TARGET_COLUMN,
            seq_len=20, epochs=50
        )
        lstm_scores = predict_lstm(model_LSTM, panel, FEATURE_COLUMNS, as_of=args.as_of)

        common_stocks = lstm_scores.index.intersection(scores.index) #scores = xgb_scores
        print("common_stocks:",common_stocks)
        print("count common_stocks:", len(common_stocks))



    weights = build_portfolio(scores, args.alpha, vols=vols, top_k=args.top_k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    out.to_csv(out_path, index=False)
    print(f">> Wrote {len(out)} names to {out_path}")
    print(f"   weight summary: min={out['weight'].min():.4f} "
          f"max={out['weight'].max():.4f} sum={out['weight'].sum():.4f}")


if __name__ == "__main__":
    main()
