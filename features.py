"""
Feature engineering for the CSI500 stock-selection baseline.

A small set of classic technical features + cross-sectional ranks.  Students are
encouraged to extend this (add fundamentals, industry dummies, alternative data,
better cross-sectional normalization, etc.).

The target is the 5-trading-day forward return on the forward-adjusted close,
i.e. what the portfolio earns if you hold a $1 position from close(t) to close(t+5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import akshare as ak
from pathlib import Path

# columns used downstream by the baseline
BASE_FEATURE_COLUMNS = [
    "ret_5d", "ret_1d", "ret_10d", "ret_20d", "ret_60d", 
    "vol_20d", "volume_z_20d", "turnover_ma_20d", 
    "close_over_ma20", "close_over_ma60", "rsi_14",
    "ret_5d_rank", "ret_20d_rank", "vol_20d_rank",
    "reversal_1w", "low_vol_rank", #LAUREN ADDED THESE TWO <---
    "risk_adj_momen" #ADDED 20260504
]


SECTOR_FEATURE_COLUMNS = [
    # sector features -- lauren added!
    #drop -> "ret_5d_vs_sector", 
    "ret_20d_vs_sector", "ret_60d_vs_sector",
    "volume_z_vs_sector",
    "sector_l1_encoded", "sector_l2_encoded"
]
MARGIN_FEATURE_COLUMNS = [
    # margin features -- lauren added!
    "margin_balance_chg_5d",
    #drop -> "margin_balance_chg_20d", 
    "margin_turnover_ratio",
    "short_vol_chg_5d",
    #drop -> "net_sentiment_chg_5d",
    "net_sentiment_chg_20d",
]
FIN_FEATURE_COLUMNS = [ 
    #"pb_vs_sector", NOT helpful
    "roe_chg_qoq"
]
BASE_SECTOR_COLUMNS = BASE_FEATURE_COLUMNS + SECTOR_FEATURE_COLUMNS
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS +SECTOR_FEATURE_COLUMNS + MARGIN_FEATURE_COLUMNS  + FIN_FEATURE_COLUMNS# remove fin_feature_cols
TARGET_COLUMN = "target_5d"
FORWARD_HORIZON = 5


def _per_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features that only depend on a single stock's time series."""
    df = df.sort_values("date").copy()
    close = df["close"]

    df["ret_1d"] = close.pct_change(1)
    df["ret_5d"] = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)
    df["ret_20d"] = close.pct_change(20)
    df["ret_60d"] = close.pct_change(60)

    df["vol_20d"] = df["ret_1d"].rolling(20).std()

    vol = df["volume"].astype(float)
    vol_mean = vol.rolling(20).mean()
    vol_std = vol.rolling(20).std().replace(0, np.nan)
    df["volume_z_20d"] = (vol - vol_mean) / vol_std

    if "turnover" in df.columns:
        df["turnover_ma_20d"] = df["turnover"].astype(float).rolling(20).mean()
    else:
        df["turnover_ma_20d"] = np.nan

    df["close_over_ma20"] = close / close.rolling(20).mean() - 1.0
    df["close_over_ma60"] = close / close.rolling(60).mean() - 1.0

    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    rs = up / down
    df["rsi_14"] = 100 - 100 / (1 + rs)

    """ LAUREN ADDING THIS"""
    df["reversal_1w"] = -df["ret_5d"]  # negative of 5-day return
    df["low_vol_rank"] = 1 - df.groupby("date")["vol_20d"].rank(pct=True)

    df[TARGET_COLUMN] = close.shift(-FORWARD_HORIZON) / close - 1.0
    return df

""" -------- Functions I (Lauren) added: --------"""
def get_sector_map_ak(constituents_path="data/constituents.csv",clf=None): # don't call many times, just used for one-time-build
    """Please use get_sector_map_csv to retrieve data from CSV file instead.
    
    Dataframe mapping CSI-500 stocks to a sector.

    Columns: stock_code, sector_code, sector_l1, sector_l2.
    """
    if clf is None:
        clf = ak.stock_industry_clf_hist_sw()
        # most recent classification per stock
        clf = (clf
            .sort_values("start_date", ascending=False)
            .drop_duplicates(subset=["symbol"], keep="first")
            .rename(columns={"symbol": "stock_code", "industry_code": "sector_code"})
        )
    
    clf["sector_code"] = clf["sector_code"].astype(str)
    clf["sector_l1"] = clf["sector_code"].str[:2]
    clf["sector_l2"] = clf["sector_code"].str[:4]
    
    # filter to CSI500 universe
    constituents = pd.read_csv(constituents_path, dtype={"stock_code": str})
    constituents["stock_code"] = constituents["stock_code"].str.zfill(6)
    csi500_codes = set(constituents["stock_code"]) #now same format as in clf
    
    clf = clf[clf["stock_code"].isin(csi500_codes)]
    
    print(f"Sector map: {len(clf)} stocks matched out of {len(csi500_codes)} CSI500 constituents")
    missing = csi500_codes - set(clf["stock_code"]) #csi500 codes not found in clf
    if missing:
        print(f"  {len(missing)} stocks have no sector mapping: {sorted(missing)[:5]}...")
    
    return clf[["stock_code", "sector_code", "sector_l1", "sector_l2"]]

# -----downloading to and from csv file-----
def save_sector_map(): #--one time save to csv file
    sector_map_df = get_sector_map_ak()
    sector_map_df.to_csv("data/sector_map.csv",index=False)

def load_sector_map(path="data/sector_map.csv"):
    """Dataframe mapping CSI-500 stocks to a sector.

    Columns: stock_code, sector_code, sector_l1, sector_l2.
    """
    print("Attempting to load sector map from csv file...")
    if Path(path).exists():
        return pd.read_csv(path, dtype={"stock_code": str})
    print("sector_map.csv not found, downloading...")
    sector_map = get_sector_map_ak()
    sector_map.to_csv(path, index=False)
    return sector_map

def load_margin_features():
    df = pd.read_csv("data/margin.csv",dtype={"Stockcode": str})
    df = df.rename(columns={"Mtdate": "date", 
                            "Stockcode": "stock_code",
                            "Financebalance": "margin_balance",
                            "Financeamount": "margin_turnover",
                            "Secshortbalance": "short_balance_volume",
                            "Shortbalance": "short_balance"})
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])

    #keep only A-shares
    df = df[df["SecurityTypeID"] == "S0101"]

    #manipulate features
    #rate of change features
    df["margin_balance_chg_5d"] = df.groupby("stock_code")["margin_balance"].pct_change(5) #shorter-term change
    df["margin_balance_chg_20d"] = df.groupby("stock_code")["margin_balance"].pct_change(20) #longer-term change

    # margin turnover ratio — new buying relative to outstanding balance
    df["margin_turnover_ratio"] = df["margin_turnover"] / (df["margin_balance"] + 1)

    # short ratio — how much of float is sold short (need to normalize somehow)
    df["short_vol_chg_5d"] = df.groupby("stock_code")["short_balance_volume"].pct_change(5)

    # net pressure in same units
    df["net_leverage_sentiment"] = df["margin_balance"] - df["short_balance"]

    #net sentiment change
    df["net_sentiment_chg_5d"] = df.groupby("stock_code")["net_leverage_sentiment"].pct_change(5)
    df["net_sentiment_chg_20d"] = df.groupby("stock_code")["net_leverage_sentiment"].pct_change(20)

    #handle inf values caused by division of 0 for calculation of features
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def load_fin_ind_feats():
    """Data on quality (ROE) and value (P/B ratio)
    """
    df_quality = pd.read_csv("data/roe.csv",dtype={"Stkcd": str})
    df_quality = df_quality.rename(columns={"F050501B": "roe", "Stkcd": "stock_code", "Accper":"end_date"})
    df_quality = df_quality[df_quality["Typrep"] == "A"] #We want consolidated statements

    df_value = pd.read_csv("data/pb_ratio.csv",dtype={"Stkcd": str})
    df_value = df_value.rename(columns={"F100401A": "pb_ratio", "Stkcd": "stock_code","Accper":"end_date"})

    df = df_quality.merge(df_value,on=["stock_code","end_date"],how="left")

    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    df["end_date"] = pd.to_datetime(df["end_date"])
    #add 45 days to prevent lookahead leakage -- we will merge panel on available_date and date
    df["available_date"] = df["end_date"]+ pd.Timedelta(days=45) 

    return df


#-----------------------------------------

def add_sector_features(panel: pd.DataFrame, sector_map: pd.DataFrame) -> pd.DataFrame:
    """Add sector features to dataframe with CSI-500 stocks.

    Added features include: sector, sector_mean, sector_vol_mean.
    """
    panel = panel.merge(sector_map,on="stock_code",how="left")

    #now get sector means
    for col in ["ret_5d", "ret_20d", "ret_60d"]:
        sector_mean = panel.groupby(["date", "sector_l1"])[col].transform("mean")
        panel[f"{col}_vs_sector"] = panel[col] - sector_mean

    sector_vol_mean = panel.groupby(["date","sector_l1"])["volume_z_20d"].transform("mean")
    panel["volume_z_vs_sector"] = panel["volume_z_20d"] - sector_vol_mean
    #categorical -->
    panel["sector_l1_encoded"] = panel["sector_l1"].astype("category").cat.codes
    panel["sector_l2_encoded"] = panel["sector_l2"].astype("category").cat.codes

    return panel

def add_margin_features(panel: pd.DataFrame, margin: pd.DataFrame) -> pd.DataFrame:
    """ Add margin features to dataframe with CSI-500 stocks.

    Added features include: Balance of Margin Trading (margin_balance), Turnover of Margin Trading (margin_turnover), 
    Balance Volume of Securities Lending (short_balance_volume), Balance of Securities Lending (short_balance).
    """
    margin_cols = ["stock_code", "date", 
                   "margin_balance_chg_5d", "margin_balance_chg_20d",
                   "margin_turnover_ratio", "short_vol_chg_5d",
                   "net_sentiment_chg_5d"]
    panel = panel.merge(margin,on=["stock_code","date"],how="left")

    print("sneak peek of panel w/ margin features")
    print(panel.head(6))
    return panel

def add_quality_value_features(panel: pd.Dataframe, fin_df: pd.Dataframe) -> pd.DataFrame:
    """ Add quality and value features to dataframe with CSI-500 stocks.

    Added features include: ROE and P/B ratio.
    """
    fin_df_cols = ["stock_code", "end_date", "available_date","pb_ratio", "roe"]
    
    #have to sort by merge key. also, make same exact datetime type (nanoseconds)
    panel = panel.sort_values("date")
    panel["date"] = panel["date"].astype("datetime64[ns]") 
    fin_df = fin_df.sort_values("available_date")
    fin_df["available_date"] = fin_df["available_date"].astype("datetime64[ns]")

    # quarter-over-quarter change in ROE — earnings momentum
    fin_df["roe_chg_qoq"] = fin_df.groupby("stock_code")["roe"].pct_change(1)

    # is P/B getting cheaper or more expensive recently
    fin_df["pb_chg_qoq"] = fin_df.groupby("stock_code")["pb_ratio"].pct_change(1)

    #using merge_asof since the available date should be <= panel date, not daily vals for fin data
    panel = pd.merge_asof(
        panel,
        fin_df[["stock_code", "available_date", "roe_chg_qoq", "pb_ratio"]],
        left_on="date",
        right_on="available_date",
        by="stock_code",
        direction="backward" #find the most recent available_date <= panel date
    )
    #adding pb_vs_sector
    panel["pb_vs_sector"] = panel["pb_ratio"] - panel.groupby(["date", "sector_l1"])["pb_ratio"].transform("mean")
    panel = panel.drop(columns=["pb_ratio"])

    print("sneak peek of panel w/ roe + p/b ratio")
    print(panel.head(6))
    return panel

def add_market_regime_feature(panel: pd.DataFrame, e=0.0001):
    """ Sharpe-like feature
    """
    panel["risk_adj_momen"] = panel["ret_20d"] / (panel["vol_20d"] + e)
    return panel
    

## ---------------- tweaking portfolio construction based on whether CSI500 is in a downtrend --------- WONT USE
def build_index_df(): #won't use, 
    path="data/prices.parquet"
    index_df = pd.read_parquet(path)
    print(index_df)
    return index_df


def get_market_regime(index_df, as_of_date, lookback=20, threshold=-0.02): #won't use, didn't help
    """Returns 1 (uptrend) or -1 (downtrend) based on index momentum."""
    as_of = pd.Timestamp(as_of_date)
    recent = index_df[index_df["date"] <= as_of].sort_values("date").tail(lookback)
    if len(recent) < lookback:
        return 1  # default to uptrend if not enough data
    
    # simple regime: is index above its 20-day MA?
    ma20 = recent["close"].mean()
    current = recent["close"].iloc[-1]
    pct_diff = (current - ma20) / ma20
    return 1 if pct_diff > threshold else -1  
""" -------- end of functions lauren added -------- """

def _cross_sectional_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily cross-sectional rank of selected features (values in [0, 1])."""
    for base in ["ret_5d", "ret_20d", "vol_20d"]:
        panel[f"{base}_rank"] = (
            panel.groupby("date")[base].rank(method="average", pct=True)
        )
    return panel


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Build a (date, stock_code) panel of features + target.

    Parameters
    ----------
    prices : DataFrame with columns [date, stock_code, open, close, high, low,
             volume, amount, turnover?]

    Returns
    -------
    DataFrame with FEATURE_COLUMNS and TARGET_COLUMN populated.  Rows where any
    feature is NaN (typically the first ~60 days per stock) are kept so callers
    can decide how to handle them.
    """
    required = {"date", "stock_code", "close", "volume"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices is missing required columns: {missing}")

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    panel = (
        prices.groupby("stock_code", group_keys=True)
        .apply(_per_stock_features)
        .reset_index(level="stock_code") 
        .reset_index(drop=True) 
        # removing this line bc of issue w/ newer pandas promoting grouping key to index, then dropping it--> .reset_index(drop=True)
    )
    #Added this line (Lauren) (to give market adjusted return)->
    panel[TARGET_COLUMN] = panel.groupby("date")[TARGET_COLUMN].transform(
        lambda x: x - x.mean()
    )

    panel = _cross_sectional_ranks(panel)
    assert "stock_code" in panel.columns, f"stock_code missing! columns: {panel.columns.tolist()}"
    return panel


def training_frame(panel: pd.DataFrame, min_date=None, max_date=None) -> pd.DataFrame:
    """Rows usable for supervised training: all features present AND target present.

    The target for date t uses close(t+5), so rows within the last 5 trading
    days of the panel are dropped automatically (target is NaN there).
    """
    df = panel.dropna(subset=BASE_SECTOR_COLUMNS + [TARGET_COLUMN]).copy()
    if min_date is not None:
        df = df[df["date"] >= pd.Timestamp(min_date)]
    if max_date is not None:
        df = df[df["date"] <= pd.Timestamp(max_date)]
    return df


def prediction_frame(panel: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """Rows for a single prediction date (defaults to the latest date)."""

    if as_of is None:
        as_of = panel["date"].max()
    as_of = pd.Timestamp(as_of)
    df = panel[panel["date"] == as_of].dropna(subset=BASE_SECTOR_COLUMNS).copy()
    return df
