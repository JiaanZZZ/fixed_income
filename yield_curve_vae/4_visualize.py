"""
Module 4: Latent Space 3D可视化
输出一个交互式HTML文件，可以在浏览器里旋转查看
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_plot_data():
    # latent坐标
    df_latent = pd.read_csv("data/latent_coords.csv", index_col=0, parse_dates=True)

    # 原始收益率（用于颜色编码和hover信息）
    df_raw = pd.read_csv("data/yield_curves.csv", index_col=0, parse_dates=True)

    # 对齐日期
    df = df_latent.join(df_raw, how="inner")
    return df

# ── 颜色编码逻辑 ──────────────────────────────────────────────────────────────

def assign_regime(spread: pd.Series) -> pd.Series:
    """
    根据2Y-10Y spread判断曲线形态
    > +50bp  : Normal (正常陡峭)
    -50~50bp : Flat   (平坦)
    < -50bp  : Inverted (倒挂)
    """
    conditions = [
        spread >  0.5,
        spread < -0.5,
    ]
    choices = ["Normal", "Inverted"]
    return pd.Series(
        np.select(conditions, choices, default="Flat"),
        index=spread.index
    )

# ── 主图 ─────────────────────────────────────────────────────────────────────

def plot_latent_space(df: pd.DataFrame):
    df["regime"]       = assign_regime(df["spread_2y10y"])
    df["date_str"]     = df.index.strftime("%Y-%m-%d")
    df["spread_abs"]   = df["spread_2y10y"].abs()

    # 配色
    color_map = {
        "Normal":   "#2563eb",   # 蓝色
        "Flat":     "#f59e0b",   # 黄色
        "Inverted": "#dc2626",   # 红色
    }

    fig = go.Figure()

    for regime, color in color_map.items():
        mask = df["regime"] == regime
        sub  = df[mask]

        fig.add_trace(go.Scatter3d(
            x=sub["z0"],
            y=sub["z1"],
            z=sub["z2"],
            mode="markers",
            name=regime,
            marker=dict(
                size=3,
                color=color,
                opacity=0.7,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Regime: " + regime + "<br>"
                "2Y-10Y spread: %{customdata[1]:.2f}%<br>"
                "10Y yield: %{customdata[2]:.2f}%<br>"
                "<extra></extra>"
            ),
            customdata=sub[["date_str", "spread_2y10y", "10Y"]].values,
        ))

    fig.update_layout(
        title=dict(
            text="Yield Curve Latent Space — VAE 3D Projection<br>"
                 "<sup>每个点 = 一个交易日的收益率曲线 | 颜色 = 曲线形态</sup>",
            font=dict(size=16),
        ),
        scene=dict(
            xaxis_title="Latent Dim 0 (z₀)",
            yaxis_title="Latent Dim 1 (z₁)",
            zaxis_title="Latent Dim 2 (z₂)",
            bgcolor="rgb(10, 10, 20)",
            xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            zaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        ),
        paper_bgcolor="rgb(10, 10, 20)",
        plot_bgcolor ="rgb(10, 10, 20)",
        font=dict(color="white"),
        legend=dict(
            title="曲线形态",
            itemsizing="constant",
        ),
        width=1000,
        height=750,
    )

    return fig

# ── PCA对比图（用来说明非线性优势）────────────────────────────────────────────

def plot_pca_comparison(df: pd.DataFrame):
    from sklearn.decomposition import PCA

    raw_cols = ["1M","3M","6M","1Y","2Y","5Y","7Y","10Y","20Y","30Y"]
    X = df[raw_cols].values

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    df["pca0"] = X_pca[:, 0]
    df["pca1"] = X_pca[:, 1]
    df["pca2"] = X_pca[:, 2]

    color_map = {"Normal": "#2563eb", "Flat": "#f59e0b", "Inverted": "#dc2626"}
    fig = go.Figure()

    for regime, color in color_map.items():
        mask = df["regime"] == regime
        sub  = df[mask]
        fig.add_trace(go.Scatter3d(
            x=sub["pca0"], y=sub["pca1"], z=sub["pca2"],
            mode="markers",
            name=regime,
            marker=dict(size=3, color=color, opacity=0.7),
        ))

    explained = pca.explained_variance_ratio_
    fig.update_layout(
        title=dict(
            text=f"PCA 3D Projection（对比用）<br>"
                 f"<sup>解释方差: PC1={explained[0]:.1%}, PC2={explained[1]:.1%}, PC3={explained[2]:.1%}</sup>",
            font=dict(size=16),
        ),
        scene=dict(
            xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3",
            bgcolor="rgb(10, 10, 20)",
        ),
        paper_bgcolor="rgb(10, 10, 20)",
        font=dict(color="white"),
        width=1000, height=750,
    )

    return fig

# ── 主流程 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_plot_data()
    print(f"可视化数据: {len(df)} 天")

    # VAE latent space图
    fig_vae = plot_latent_space(df)
    fig_vae.write_html("latent_space_vae.html")
    print("✓ VAE图已保存: latent_space_vae.html")

    # PCA对比图
    fig_pca = plot_pca_comparison(df)
    fig_pca.write_html("latent_space_pca.html")
    print("✓ PCA图已保存: latent_space_pca.html")

    print("\n在浏览器打开HTML文件，可以旋转3D图查看manifold结构")
