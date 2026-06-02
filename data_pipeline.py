import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

class YieldCurvePipeline:
    def __init__(self):
        # 使用 FRED API 免费公开接口获取美债收益率数据
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        # 门外汉用复杂的鉴权，门里人做项目先用公共镜像源，或者你也可以去FRED官网秒申一个免费API Key
        # 这里我们先用一种极其稳健的、绕过Key的直接下载CSV方式，确保你100%能跑通
        self.series_ids = {
            '2Y': 'DGS2',   # 2-Year Treasury Constant Maturity Rate
            '5Y': 'DGS5',   # 5-Year Treasury Constant Maturity Rate
            '10Y': 'DGS10'  # 10-Year Treasury Constant Maturity Rate
        }

    def fetch_data(self) -> pd.DataFrame:
        """从FRED异步/循环获取国债数据并进行外键拼接"""
        dfs = []
        for name, series_id in self.series_ids.items():
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            df = pd.read_csv(url)
            # FRED 近期把日期列从 'DATE' 改成了 'observation_date'，这里自动识别第一列作为时间戳
            date_col = df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
            # 过滤掉非数字的脏数据（比如市场休市时的 "."）
            df[name] = pd.to_numeric(df[series_id], errors='coerce')
            dfs.append(df[[name]])
        
        # 纵向合并，按时间戳对齐
        curve_df = pd.concat(dfs, axis=1)
        # 前向填充市场休市导致的空值，然后剔除无法对齐的残缺行
        curve_df = curve_df.ffill().dropna()
        return curve_df

    def calculate_butterfly_spread(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算经典的蝶式价差 (Butterfly Spread)
        Formula: Spread = 2Y_Yield - 2 * 5Y_Yield + 10Y_Yield
        """
        df['Butterfly_Spread'] = df['2Y'] - 2 * df['5Y'] + df['10Y']
        
        # 计算实现波动率 (Realized Volatility) - 滚动30天
        # 金融城交易台看这个指标来判断市场是否开始发生结构性变异
        df['Spread_Vol'] = df['Butterfly_Spread'].rolling(window=30).std()
        
        return df

# 测试管道运行
if __name__ == "__main__":
    print("正在启动数据管道，正在从美国联储数据库提取美债曲线...")
    pipeline = YieldCurvePipeline()
    curve_data = pipeline.fetch_data()
    curve_data = pipeline.calculate_butterfly_spread(curve_data)
    
    print("\n--- 数据清洗完毕，最新交易日曲线切片 ---")
    print(curve_data.tail())
    
    # 画出蝶式价差的历史走势，看看它在危机时是怎么非线性跳跃的
    plt.figure(figsize=(12, 6))
    plt.plot(curve_data.index[-252*5:], curve_data['Butterfly_Spread'].iloc[-252*5:], label='Butterfly Spread (2Y-2*5Y+10Y)')
    plt.title('Yield Curve Curvature (Last 5 Years)')
    plt.grid(True)
    plt.legend()
    plt.savefig('butterfly_spread.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图表已保存至 butterfly_spread.png")