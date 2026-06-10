🌐 Global Macro Dislocation & Fixed Income Portfolio Positioning Engine
基于全球央行非结构化文本的固定收益投资组合（Duration & Curve）动态对冲模型
📌 项目核心痛点 (The Real-World Problem)
在固定收益（Rates）市场中，中央银行（如美联储 Fed / 英国央行 BOE）的公开 Forward Guidance（前瞻指引）经常与内部真实的宏观焦虑存在非线性偏差（即言行不一）。传统投资经理依赖人工阅读长篇报告，反应存在严重滞后。本系统旨在利用 Python 数据工程 + LLM 语义识别，量化央行的“隐性焦虑”，捕捉市场对未来利率路径的错误定价（Dislocation），并将其自动转化为固定收益投资组合的久期（Duration）与曲线（Curve）交易策略。
🛠️ 项目系统架构大纲 (Project Outline) wor
阶段一：自动化宏观数据管线 (Data Engineering Pipeline)
目标：白手起家，构建稳健的、零人工介入的一手宏观数据源。
•	[ ] 1.1 央行文本抓取模块 (scrapers/)
•	利用 requests / BeautifulSoup 自动监控美联储（Federal Reserve）和英国央行（BOE）官网。
•	实时抓取最新的主席演讲稿（Speeches）、**会议纪要（FOMC Minutes）**及金融稳定报告。
•	[ ] 1.2 市场利率数据同步 (market_data/)
•	通过 yfinance 接口或圣路易斯联储 FRED API 自动同步核心 Rates 资产：
•	美国2年期国债收益率（短端政策利率风向标）
•	美国10年期国债收益率（长端宏观与通胀溢价风向标）
•	自动计算并清洗出每日的 10Y-2Y 收益率利差（Yield Curve Spread）。
阶段二：AI 智能体情绪解构引擎 (LLM Sentiment Engine)
目标：把你的历史、人性、社会博弈直觉，内化为 Agent 的判定规则。
•	[ ] 2.1 文本结构化与 Embedding 化
•	将几万字的会议纪要长文本进行智能分块（Chunking），保留时间戳与上下文。
•	[ ] 2.2 “央行隐性焦虑” Agent 设定 (agents/cb_analyst.py)
•	调用大模型 API（如 OpenAI 或 Anthropic），设计工业级 System Prompt：
•	维度 A：鹰鸽非一致性（Policy Inconsistency） —— 捕捉措辞相较于历史报告的微妙转变。
•	维度 B：历史危机相似度（Historical Parallels） —— 评估当前文本与历史上（如1970年代滞胀、2008年流动性危机）的文本距离。
•	维度 C：语义逃避度（Uncertainty Index） —— 统计“难以评估”、“有待观察”等恐慌词汇的密度。
•	强迫 Agent 规避“小作文”，严格输出包含上述得分的结构化 JSON。
阶段三：宏观扭曲量化与交易信号生成 (Dislocation & Signal Generation)
目标：连接分析与交易，生成让 Trader 兴奋的信号。
•	[ ] 3.1 偏离度指数计算 (analytics/quant_engine.py)
•	将 AI 算出的“央行真实焦虑指数”与市场通过国债利差交易出来的“隐含恐慌度”进行对齐。
•	计算两者的差值 ‭‬‭‬‭‬。
•	[ ] 3.2 信号触发器 (signals/)
•	运用动态标准差滚动窗口：当 ‭‬ 偏离过去 90 天均值 1.5 个标准差 以上时，判定市场发生错误定价，触发系统警报。
阶段四：投资组合映射与对冲方案 (Portfolio Construction - 大摩最看重这步)
目标：完美契合摩根士丹利 Portfolio Analyst 岗位的职责核心。
•	[ ] 4.1 投资组合策略自动化生成 (portfolio/positioning.py)
•	情境 A：AI 焦虑暴增，市场曲线依旧平坦 ‭‬ 自动生成 Curve Steepener（曲线走陡策略） 报告：建议 BUY 2Y Treasury / SELL 10Y Treasury，并计算 DV01 风险对冲比例。
•	情境 B：AI 显示央行彻底嘴硬，但数据已经崩塌 ‭‬ 自动生成 Duration Hedging（久期对冲策略）：提示基金经理削减长端 IG Credit 敞口，防范信用踩踏。