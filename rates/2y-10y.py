# 你的从零开始：第一行宏观破局代码
import yfinance as yf
import pandas as pd

# 1. 自动拉取2年期和10年期美债的数据（在门里，我们用^IRX, ^FVX, ^TNX等代表收益率）
# 这里我们直接拉取真实的10年期美债和2Y美债的相关ETF或收益率代号
print("正在拉取宏观核心数据...")
tickers = ['^TNX', '^IRX'] # ^TNX是10年期利率，^IRX是13周/短端利率
data = yf.download(tickers, start="2020-01-01")['Close']

# 2. 计算利差 (Spread)
data['Spread'] = data['^TNX'] - data['^IRX']

# 3. 打印出历史上利差最极端的日子
print("\n【历史最极端宏观扭曲时刻】")
print(data.sort_values(by='Spread').head(5)) # 倒挂最严重的几天