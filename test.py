#%%
import akshare as ak
import pandas as pd
import inspect
from features import load_fin_ind_feats, load_sector_map, build_index_df


"""
df = pd.read_csv("tweaking_params/topk_results.csv")
df["IR"] = df["er_mean"] / df["er_std"]
print(df.sort_values("IR", ascending=False))
"""


"""
df = build_index_df()
"""

""" testing march13.csv 
march13_portfolio = pd.read_csv("submissions/march13.csv",dtype={"stock_code": str})
sector_map = load_sector_map()
march13_portfolio = march13_portfolio.merge(sector_map,on="stock_code",how="left")
print(march13_portfolio.value_counts("sector_l1"))
print(march13_portfolio.value_counts("sector_l2"))
"""


#%%
df = pd.read_csv("logs3/updated/summary.csv")
df["excess_return"] = pd.to_numeric(df["excess_return"])
df["rank_ic"] = pd.to_numeric(df["rank_ic"])

print("instances of negative excess_return")
df_neg = df[df["excess_return"] < 0]
print(df_neg[["as_of","test_start","test_end","excess_return"]])
print("instances of negative rank_ic")
df_neg = df[df["rank_ic"] < 0]
print(df_neg[["as_of","test_start","test_end","rank_ic"]])

#%%
summary_ER = (df
    .groupby(["subsample", "colsample", "min_child_weight", "k", "alpha"])
    .agg(
        mean=("excess_return", "mean"),
        std=("excess_return", "std"),
        min=("excess_return", "min"),
        max=("excess_return", "max"),
        median=("excess_return", "median"),
        n_negative=("excess_return", lambda x: (x < 0).sum()),
        n_windows=("excess_return", "count"),
    )
    .sort_values("mean", ascending=False)
)
summary_ER["win_rate"] = 1 - summary_ER["n_negative"] / summary_ER["n_windows"]

summary_IC = df.groupby(["subsample","colsample","min_child_weight","k", "alpha"])["rank_ic"].agg(["mean","std","min"]).sort_values("mean", ascending=False)
summary_ER["IR"] = summary_ER["mean"] / summary_ER["std"]
print(" ---------summary stats for ER (sorted by mean ER)--------- ")
print(summary_ER)
print("\n---------summary stats for rank_IC---------")
print(summary_IC)

print(" ---------summary stats for ER (sorted by IR)--------- ")
print(summary_ER.sort_values("IR",ascending=False))
# %%
"""
filter_layer = "subsample == 0.5 & colsample == 0.8 & min_child_weight == 10 &  k == 38"
filter_df = df.query(filter_layer)
print(filter_df)
"""

