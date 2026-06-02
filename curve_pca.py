import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from data_pipeline import YieldCurvePipeline # 复用你第一步写的管道

def run_curve_pca():
    # 1. 初始化并获取清洗后的数据
    pipeline = YieldCurvePipeline()
    df = pipeline.fetch_data()
    
    # 我们只对 2Y, 5Y, 10Y 这三列原始收益率进行 PCA 分析
    features = ['2Y', '5Y', '10Y']
    X = df[features]
    
    # 2. 我们将收益率做“一阶差分”（即每日变动值），因为金融时间序列的绝对值往往不平稳
    X_diff = X.diff().dropna()
    
    # 3. 实例化 PCA 模型，我们要提取 3 个主成分
    pca = PCA(n_components=3)
    pca.fit(X_diff)
    
    # 4. 打印每个主成分解释的方差比例 (Explained Variance Ratio)
    print("\n" + "="*50)
    print("【工业级 PCA 曲线解构结果】")
    print(f"第一主成分 (PC1) 解释力度: {pca.explained_variance_ratio_[0]*100:.2f}%")
    print(f"第二主成分 (PC2) 解释力度: {pca.explained_variance_ratio_[1]*100:.2f}%")
    print(f"第三主成分 (PC3) 解释力度: {pca.explained_variance_ratio_[2]*100:.2f}%")
    print(f"总解释度 (Total): {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
    print("="*50)
    
    # 5. 打印因子载荷 (Component Loadings) —— 这就是区分内行外行的关键
    loadings = pd.DataFrame(
        pca.components_.T, 
        columns=['PC1_Shift', 'PC2_Twist', 'PC3_Curvature'], 
        index=features
    )
    print("\n--- 因子载荷矩阵 (Loadings Matrix) ---")
    print(loadings)
    print("-"*50)
    
    return pca, X_diff

if __name__ == "__main__":
    run_curve_pca()