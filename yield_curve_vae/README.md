# Yield Curve VAE — Project Framework

## 结构总览

```
yield_curve_vae/
├── 1_data.py          # 数据获取和预处理
├── 2_model.py         # VAE模型定义
├── 3_train.py         # 训练循环
├── 4_visualize.py     # Latent space 3D可视化
└── README.md
```

## 运行顺序
```bash
python 1_data.py       # 生成 data/yield_curves.csv
python 2_model.py      # 定义模型（无需单独运行，被import）
python 3_train.py      # 训练，保存 checkpoints/vae.pt
python 4_visualize.py  # 输出 latent_space_3d.html
```

## 依赖
```
pip install torch pandas numpy plotly pandas-datareader scikit-learn
```
