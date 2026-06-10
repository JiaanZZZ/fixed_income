"""
Module 3: 训练循环
读取预处理后的数据，训练VAE，保存模型权重
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
from model import YieldCurveVAE, vae_loss

# ── 配置 ──────────────────────────────────────────────────────────────────────

EPOCHS     = 200
BATCH_SIZE = 64
LR         = 1e-3
BETA       = 1.0    # KL weight（可以试试 0.5 或 2.0）
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv("data/yield_curves_scaled.csv", index_col=0, parse_dates=True)
    # 只取收益率列（去掉spread_2y10y如果有）
    feature_cols = [c for c in df.columns if c != "spread_2y10y"]
    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    return X, df.index

# ── 训练 ─────────────────────────────────────────────────────────────────────

def train():
    os.makedirs("checkpoints", exist_ok=True)

    X, dates = load_data()
    dataset  = TensorDataset(X)
    loader   = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model     = YieldCurveVAE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"设备: {DEVICE}")
    print(f"训练样本: {len(X)} 天")
    print(f"开始训练 ({EPOCHS} epochs)...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for (batch,) in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()

            x_recon, mu, log_var = model(batch)
            loss = vae_loss(batch, x_recon, mu, log_var, beta=BETA)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(X)

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{EPOCHS}  |  avg loss: {avg_loss:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "checkpoints/vae.pt")
    print("\n✓ 模型已保存至 checkpoints/vae.pt")

    # 顺便保存latent坐标（用于可视化）
    model.eval()
    with torch.no_grad():
        X_device = X.to(DEVICE)
        latent   = model.encode_to_latent(X_device).cpu().numpy()

    df_latent = pd.DataFrame(
        latent,
        index=dates,
        columns=["z0", "z1", "z2"]
    )
    df_latent.to_csv("data/latent_coords.csv")
    print("✓ Latent坐标已保存至 data/latent_coords.csv")


if __name__ == "__main__":
    train()
