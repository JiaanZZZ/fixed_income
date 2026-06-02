import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

print("正在从 FRED 拉取 2Y / 10Y 美债收益率...")

end_date = datetime.today()
start_date = end_date - timedelta(days=365 * 5)

def fetch_fred(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    return pd.to_numeric(df[series_id], errors='coerce').rename(series_id)

y2  = fetch_fred('DGS2')
y10 = fetch_fred('DGS10')

data = pd.concat([y2, y10], axis=1).ffill().dropna()
data = data[data.index >= pd.Timestamp(start_date)]
data['Spread'] = data['DGS10'] - data['DGS2']

print(f"\n数据范围: {data.index[0].date()} → {data.index[-1].date()}  ({len(data)} 个交易日)")
print(f"当前利差 (10Y - 2Y): {data['Spread'].iloc[-1]:+.3f}%")
print(f"5年利差最低 (最深倒挂): {data['Spread'].min():+.3f}%  ({data['Spread'].idxmin().date()})")
print(f"5年利差最高:           {data['Spread'].max():+.3f}%  ({data['Spread'].idxmax().date()})")

# 画图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
fig.suptitle('US Treasury Yield Curve: 2Y vs 10Y (Past 5 Years)', fontsize=14, fontweight='bold')

ax1.plot(data.index, data['DGS10'], label='10Y', color='#1f77b4', linewidth=1.5)
ax1.plot(data.index, data['DGS2'],  label='2Y',  color='#ff7f0e', linewidth=1.5)
ax1.set_ylabel('Yield (%)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.fill_between(data.index, data['Spread'], 0,
                 where=(data['Spread'] >= 0), color='#2ca02c', alpha=0.4, label='Normal (10Y > 2Y)')
ax2.fill_between(data.index, data['Spread'], 0,
                 where=(data['Spread'] < 0),  color='#d62728', alpha=0.4, label='Inverted (10Y < 2Y)')
ax2.plot(data.index, data['Spread'], color='black', linewidth=1)
ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax2.set_ylabel('Spread (10Y - 2Y, %)')
ax2.legend(loc='lower left')
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.autofmt_xdate()

# 关键事件标注
events = [
    ('2022-03-16', 'Fed 首次\n加息',       '#e67e22', 'bottom'),
    ('2022-11-30', 'ChatGPT\n问世',         '#8e44ad', 'top'),
    ('2023-05-24', 'NVDA 财报\n股价起飞',   '#27ae60', 'bottom'),
]

for date_str, label, color, valign in events:
    dt = pd.Timestamp(date_str)
    if dt < data.index[0] or dt > data.index[-1]:
        continue
    for ax in (ax1, ax2):
        ax.axvline(dt, color=color, linewidth=1.2, linestyle='--', alpha=0.85)
    y_pos = ax1.get_ylim()[1] * 0.95 if valign == 'top' else ax1.get_ylim()[0] + 0.05
    ax1.annotate(label, xy=(dt, y_pos),
                 xytext=(6, 0), textcoords='offset points',
                 fontsize=8, color=color, fontweight='bold',
                 va='top' if valign == 'top' else 'bottom')

plt.tight_layout()
plt.savefig('2y_10y_spread.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n图表已保存至 2y_10y_spread.png")
