"""
Module 1: 数据获取和预处理
从FRED拉取美国国债收益率历史数据（免费，无需API key）
输出: data/yield_curves.csv
"""

import pandas as pd
import numpy as np
import pandas_datareader.data as web
import os
from datetime import datetime

# ── 配置 ──────────────────────────────────────────────────────────────────────

START_DATE = "2024-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")

# FRED ticker → 期限标签
FRED_TICKERS = {
    "DGS1MO":  "1M",
    "DGS3MO":  "3M",
    "DGS6MO":  "6M",
    "DGS1":    "1Y",
    "DGS2":    "2Y",
    "DGS5":    "5Y",
    "DGS7":    "7Y",
    "DGS10":   "10Y",
    "DGS20":   "20Y",
    "DGS30":   "30Y",
}

import pandas as pd

def load_treasury_csv(path="data/treasury_raw.csv"):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date")
    
    # Treasury的列名 → 我们的标签
    rename_map = {
        "1 Mo": "1M", "3 Mo": "3M", "6 Mo": "6M",
        "1 Yr": "1Y", "2 Yr": "2Y", "5 Yr": "5Y",
        "7 Yr": "7Y", "10 Yr": "10Y", "20 Yr": "20Y", "30 Yr": "30Y"
    }
    df = df.rename(columns=rename_map)
    df = df[list(rename_map.values())].dropna()
    df["spread_2y10y"] = df["10Y"] - df["2Y"]
    return df

# ── 预处理 ────────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    清洗 + 标准化
    返回: (df_clean, df_scaled)
    df_clean  → 原始收益率，保留用于可视化颜色编码
    df_scaled → 标准化后的数据，输入VAE
    """
    # 1. 只保留所有期限都有数据的交易日
    df = df.dropna()
    print(f"\n有效交易日: {len(df)} 天 ({df.index[0].date()} → {df.index[-1].date()})")

    # 2. 计算2Y-10Y spread（用于后续颜色编码）
    df["spread_2y10y"] = df["10Y"] - df["2Y"]

    # 3. 标准化（zero mean, unit variance）
    features = [col for col in df.columns if col != "spread_2y10y"]
    df_scaled = df[features].copy()
    means = df_scaled.mean()
    stds  = df_scaled.std()
    df_scaled = (df_scaled - means) / stds

    return df, df_scaled

# ── 主流程 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    df_raw = load_treasury_csv()
    df_clean, df_scaled = preprocess(df_raw)

    # 保存
    df_clean.to_csv("data/yield_curves.csv")
    df_scaled.to_csv("data/yield_curves_scaled.csv")
    print("\n✓ 数据已保存至 data/")
    print(df_clean.tail(3))
