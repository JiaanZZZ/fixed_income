"""
Module 2: VAE 模型定义（PyTorch）

架构:
  Encoder: 10维 → 隐藏层 → mu(3维) + log_var(3维)
  Decoder: 3维 → 隐藏层 → 10维重建

关键概念:
  - mu        : latent space里这个点的"中心位置"
  - log_var   : 这个点的"不确定性范围"（取log保证数值稳定）
  - z         : 从 N(mu, sigma) 采样得到的实际latent vector
  - KL loss   : 约束latent space不要太"分散"，保持结构
  - Recon loss: 重建误差，保证信息不丢失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 配置 ──────────────────────────────────────────────────────────────────────

INPUT_DIM  = 10   # 收益率曲线的期限数量
HIDDEN_DIM = 32   # 隐藏层大小（可调）
LATENT_DIM = 3    # 压缩到3维，用于3D可视化


# ── Encoder ───────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    把10维收益率曲线压缩成latent space里的一个分布
    输出: mu (均值), log_var (方差的log)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
        )
        self.fc_mu      = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.fc_log_var = nn.Linear(HIDDEN_DIM, LATENT_DIM)

    def forward(self, x):
        h = self.net(x)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var


# ── Decoder ───────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    把latent vector (3维) 还原回10维收益率曲线
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, INPUT_DIM),
            # 不加activation：收益率是unbounded的实数
        )

    def forward(self, z):
        return self.net(z)


# ── VAE (完整模型) ────────────────────────────────────────────────────────────

class YieldCurveVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick:
        直接从分布采样无法反向传播，所以改写成:
            z = mu + epsilon * sigma
        其中 epsilon ~ N(0,1) 是随机噪声
        这样梯度可以通过mu和sigma传回去
        """
        std     = torch.exp(0.5 * log_var)   # sigma = exp(0.5 * log_var)
        epsilon = torch.randn_like(std)       # epsilon ~ N(0,1)
        z       = mu + epsilon * std
        return z

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z           = self.reparameterize(mu, log_var)
        x_recon     = self.decoder(z)
        return x_recon, mu, log_var

    def encode_to_latent(self, x):
        """推断时用：直接返回mu，不采样（确定性）"""
        mu, _ = self.encoder(x)
        return mu


# ── Loss函数 ──────────────────────────────────────────────────────────────────

def vae_loss(x, x_recon, mu, log_var, beta=1.0):
    """
    VAE的总loss = 重建误差 + KL散度

    reconstruction_loss: 模型重建的曲线和原曲线差多少
    kl_loss:             latent分布和标准正态N(0,1)差多少
                         → 防止latent space"塌缩"成一堆孤立的点

    beta: 控制两项的权重（beta-VAE）
          beta > 1 → 更紧凑的latent space，但重建精度稍低
    """
    reconstruction_loss = F.mse_loss(x_recon, x, reduction="sum")

    # KL散度的解析解（对角高斯分布）
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return reconstruction_loss + beta * kl_loss


# ── 快速测试 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = YieldCurveVAE()
    print(model)

    # 模拟一个batch：32个交易日，每天10个期限
    dummy_input = torch.randn(32, INPUT_DIM)
    x_recon, mu, log_var = model(dummy_input)

    loss = vae_loss(dummy_input, x_recon, mu, log_var)

    print(f"\n输入shape:    {dummy_input.shape}")
    print(f"重建shape:    {x_recon.shape}")
    print(f"Latent mu:    {mu.shape}   ← 这就是3D坐标")
    print(f"Loss:         {loss.item():.2f}")
    print("\n✓ 模型结构正常")
