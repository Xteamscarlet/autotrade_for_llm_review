```bash```

1. 克隆项目
git clone <your-repo-url>
cd autotrade

2. 创建虚拟环境（推荐）
conda create -n autotrade python=3.10
conda activate autotrade

3. 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas ta-lib efinance akshare optuna scikit-learn joblib
pip install tqdm matplotlib python-dotenv requests

ta-lib 如果 pip 安装失败，参考：https://github.com/ta-lib/ta-lib-python

## 旧模型迁移

如果你之前已经用 V2 训练过模型，**不需要重新训练**，直接复制文件到项目根目录：

你的旧项目目录/
├── model_weights.pth → 复制到 autotrade/
├── swa_model_weights.pth → 复制到 autotrade/
├── per_stock_scalers.pkl → 复制到 autotrade/
├── global_scaler.pkl → 复制到 autotarde/
├── checkpoints/ → 整个文件夹复制到 autotrade/
│ └── *.pth
└── stock_data_cleaned.feather → 复制到 autotrade/


复制后的目录结构：
autotrade/
├── model_weights.pth ✅ EMA最佳模型（实盘推荐）
├── swa_model_weights.pth ✅ SWA模型（回测对比用）
├── per_stock_scalers.pkl ✅ 股票独立scaler
├── global_scaler.pkl ✅ 全局scaler（ unseen股票fallback）
├── checkpoints/ ✅ TopK集成模型
│ └── model_epoch*_loss*.pth
├── stock_data_cleaned.feather ✅ 训练数据
├── .env
└── …


> **为什么直接放根目录就行？** `.env` 中的路径配置默认值就是这些文件名，零配置即可识别。
> 如果你想放到其他位置，修改 `.env` 中对应的路径即可。

## 配置说明

所有参数通过 `.env` 文件管理，修改参数**不需要改任何代码**。

bash

.env 文件关键配置项
---- 代理（国内需要） ----
HTTP_PROXY=http://127.0.0.1:6789
HTTPS_PROXY=http://127.0.0.1:6789

---- 模型超参 ----
LOOKBACK_DAYS=120 # 回看天数（改这个必须重新训练）
BATCH_SIZE=256
EPOCHS=7
LEARNING_RATE=3e-5

---- 风控阈值 ----
MAX_DRAWDOWN_LIMIT=-20.0 # 最大回撤限制（%）
MIN_PROFIT_FACTOR=1.5 # 最小利润因子
MAX_POSITION_RATIO=0.3 # 单只最大仓位
MIN_WIN_RATE=40.0 # 最小胜率（%）
MIN_TRADES=15 # 最小交易次数

---- 回测参数 ----
N_OPTUNA_TRIALS=150 # Optuna试验次数（越大越慢越准）
N_SPLITS=5 # Walk-Forward折数

---- 文件路径 ----
MODEL_PATH=model_weights.pth
SCALER_PATH=per_stock_scalers.pkl
STRATEGY_FILE=optimized_strategies.json
PORTFOLIO_FILE=my_portfolio.json


## 完整使用流程

### 第一步：数据下载

bash

方式A：只下载配置中的24只股票（快速，约5分钟）
python run_data_download.py

方式B：下载全量A股（慢，约数小时，用于重新训练）
python run_data_download.py --all --batch-size 50

方式C：指定股票代码
python run_data_download.py --codes 601127 002241 600519

方式D：已有旧数据文件，跳过此步

输出文件：`stock_data_cleaned.feather`

### 第二步：训练模型

bash

方式A：直接训练（使用 .env 中的参数）
python run_train.py

方式B：先检查环境再训练
python run_train.py --dry-run # 检查数据、GPU等是否就绪
python run_train.py # 确认无误后正式训练

方式C：训练完发送企业微信通知
python run_train.py --notify

方式D：已有旧模型，跳过此步

输出文件：
- `model_weights.pth` — EMA最佳模型（**实盘用这个**）
- `swa_model_weights.pth` — SWA模型（泛化更好，可对比）
- `per_stock_scalers.pkl` — 每只股票独立的标准化器
- `global_scaler.pkl` — 全局标准化器（用于未见过的新股票）
- `checkpoints/` — TopK集成模型（3个）

### 第三步：回测优化

bash

执行完整回测流程：
1. Walk-Forward划分（5折，gap=20天防泄露）
2. Optuna多目标优化（每个市场状态150次试验）
3. 双层风控过滤（硬限制+软目标9项指标）
4. 生成可视化图表
python run_backtest.py


输出文件：
- `optimized_strategies.json` — 每只股票的最优参数和因子权重
- `stock_cache/optimized_strategy_results/backtest_charts/` — 可视化图表

终端输出示例：
测试集汇总报告 (含9项指标)
──────────────────────────────────────────────────────────────
名称 收益% 胜率% 交易数 夏普 最大回撤% 利润因子 Sortino Calmar
──────────────────────────────────────────────────────────────

贵州茅台 12.35 55.00 20 1.82 -5.20 2.15 2.45 2.38

赛力斯 8.72 50.00 18 1.45 -8.10 1.85 1.92 1.08
…
──────────────────────────────────────────────────────────────
组合总收益: 15.23%
组合夏普: 1.65 | 最大回撤: -6.80% | 利润因子: 2.05


> **注意**：如果某只股票的风控软目标未通过（如最大回撤>20%、利润因子<1.5），
> 会被自动过滤，不会出现在 `optimized_strategies.json` 中。

### 第四步：预测评分

bash

方式A：预测配置中的股票池
python run_predict.py --pool

方式B：预测指定股票
python run_predict.py --codes 601127 002241 600519 000725

方式C：只看TOP 10
python run_predict.py --pool --top 10

方式D：并发送企业微信
python run_predict.py --pool --notify


输出示例：
排名 代码 趋势 概率 预测收益 综合得分 分歧度 模型数 Scaler
──────────────────────────────────────────────────────────────────
1 601127 上涨 68.3% +3.25% +0.0221 0.0312 5 专用
2 002241 上涨 62.1% +2.10% +0.0130 0.0285 5 专用
3 600519 上涨 58.7% +1.85% +0.0109 0.0350 4 全局
…


输出文件：`stock_predictions.json`

### 第五步：实盘决策

bash

首次运行会自动生成 my_portfolio.json 模板
填入你的实际持仓信息后运行：
方式A：标准运行
python run_advisor.py

方式B：详细日志（调试用）
python run_advisor.py -v

方式C：只检查环境不执行
python run_advisor.py --dry-run


输出示例：
📢 当前市场环境: 【neutral】 (日期: 2026-03-31)

--------------------【卖出监控】--------------------
🚨 赛力斯 (601127)
现价: 85.20 | 收益: 12.35%
原因: 🛡️ 移动止损 (Level2)

--------------------【买入机会】--------------------

🟢 强信号 科大讯飞 (002230)
现价: 52.80 | 综合得分: 0.85 (阈值: 0.60)
AI观点: 0.72
📊 建议仓位: 28.5% | 建议买入 5,400 股
🟡 中信号 彩虹集团 (003023)
现价: 18.50 | 综合得分: 0.72 (阈值: 0.60)
AI观点: 0.65
📊 建议仓位: 20.0% | 建议买入 10,800 股

### 日常使用流程（简化版）

如果你已经完成了数据下载、模型训练和回测优化，日常只需要两步：

bash

每天收盘后运行（15:30之后）
python run_advisor.py


如果需要更新模型预测（比如模型很久没跑、想看最新AI观点）：

bash

先预测评分
python run_predict.py --pool --notify

再看决策建议
python run_advisor.py


## 目录结构

autotrade/
├── .env # 🔧 集中配置（改参数只改这里）
├── config.py # 配置加载逻辑
├── exceptions.py # 6种自定义异常
├── risk_manager.py # 双层风控管理器
│
├── data/ # 📊 数据层
│ ├── init.py
│ ├── types.py # 常量定义（FEATURES列名等）
│ ├── loader.py # efinance/akshare 数据下载
│ ├── normalize.py # 列名映射、OHLCV校验
│ ├── cache.py # 缓存管理（原子写入）
│ └── indicators.py # 技术指标计算（唯一来源）
│
├── strategies/ # 📈 策略层
│ ├── init.py
│ ├── base.py # BaseStrategy 抽象基类
│ ├── compound_signal.py # 复合信号策略
│ └── loader.py # 策略动态加载器
│
├── model/ # 🧠 深度学习模型层
│ ├── init.py
│ ├── transformer.py # StockTransformer 模型定义
│ ├── trainer.py # 训练循环（EMA/SWA/Scheduler）
│ ├── predictor.py # 集成推理+因子序列计算
│ └── stock_pool.json # 股票池配置
│
├── backtest/ # 🔬 回测引擎层
│ ├── init.py
│ ├── engine.py # 回测主循环
│ ├── optimizer.py # Optuna参数优化
│ ├── evaluator.py # 9项统计指标
│ └── visualizer.py # 可视化图表
│
├── live/ # 💰 实盘决策层
│ ├── init.py
│ ├── advisor.py # 决策辅助主逻辑
│ ├── signal_filter.py # 信号置信度分级
│ └── portfolio_risk.py # 组合层面风控
│
├── run_data_download.py # 🚀 入口：数据下载
├── run_train.py # 🚀 入口：模型训练
├── run_backtest.py # 🚀 入口：回测优化
├── run_predict.py # 🚀 入口：集成预测
├── run_advisor.py # 🚀 入口：实盘决策
│
├── model_weights.pth # 模型权重（EMA）
├── swa_model_weights.pth # 模型权重（SWA）
├── per_stock_scalers.pkl # 股票独立scaler
├── global_scaler.pkl # 全局scaler
├── checkpoints/ # TopK集成模型
├── stock_data_cleaned.feather # 训练数据
├── optimized_strategies.json # 回测输出的策略参数
├── my_portfolio.json # 持仓信息
└── stock_predictions.json # 预测结果


## 风控体系说明

### 硬限制（回测前阻断）

在 `risk_manager.py` 的 `check_hard_limits()` 中实现，违反直接抛异常：

| 检查项 | 规则 | 目的 |
|--------|------|------|
| 止损宽度 | stop_loss < -15% | 防止参数过于宽松 |
| 买入阈值 | buy_threshold < 0.4 | 防止过度交易 |
| 阈值关系 | buy > sell | 逻辑一致性 |
| 移动止损 | 利润/回撤参数为正且递增 | 参数合理性 |

### 软目标（回测后评估）

在 `risk_manager.py` 的 `evaluate_soft_targets()` 中实现，不达标标记为 `discard`：

| 指标 | 默认阈值 | 未通过后果 |
|------|---------|-----------|
| 最大回撤 | > 20% | **丢弃**（核心指标） |
| 利润因子 | < 1.5 | **丢弃**（核心指标） |
| 夏普比率 | < 0.5 | 警告 |
| 胜率 | < 40% | 警告 |
| 交易次数 | < 15 或 > 120 | 警告 |

### 组合风控（实盘前过滤）

在 `live/portfolio_risk.py` 中实现：

| 检查项 | 规则 |
|--------|------|
| 总仓位 | 不超过 80% |
| 单只仓位 | 不超过 30% |
| 板块集中度 | 同板块不超过 40% |

## 常见问题

### Q: 改了 LOOKBACK_DAYS 需要重新训练吗？
**需要。** `LOOKBACK_DAYS` 是模型输入维度，改了之后旧权重无法加载。其他超参（学习率、dropout等）改了不需要重新训练，只需重新跑回测优化参数即可。

### Q: 只想用回测+实盘，不想训练模型？
把旧模型文件放到根目录，直接从第三步开始：
bash
python run_backtest.py
python run_advisor.py


### Q: 回测太慢怎么优化？
1. 减少 Optuna 试验次数：`.env` 中 `N_OPTUNA_TRIALS=50`
2. 减少 Walk-Forward 折数：`.env` 中 `N_SPLITS=3`
3. 关掉 Transformer 因子：删除 `model_weights.pth`，系统会自动 fallback 到 0.5

### Q: 实盘决策和回测结果不一致怎么办？
检查以下几点：
1. `my_portfolio.json` 中的 `buy_price` 和 `buy_date` 是否正确
2. `optimized_strategies.json` 是否是最新的回测结果
3. 大盘数据缓存是否过期（删除 `stock_cache/market_data.pkl` 重新下载）

### Q: 如何添加新股票？
编辑 `config.py` 中的 `STOCK_CODES` 字典，或者在 `model/stock_pool.json` 的 `default_pool` 中添加。然后重新运行回测和预测。

### Q: 如何添加新因子？
1. 在 `data/types.py` 的 `TRADITIONAL_FACTOR_COLS` 中添加列名
2. 在 `data/indicators.py` 的 `calculate_orthogonal_factors()` 中计算该因子
3. 回测和实盘会自动识别新因子（因为因子列是动态检测的）

### Q: 企业微信通知怎么配？
在 `.env` 中填入你的 Webhook URL：
WECHAT_WEBHOOK=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=你的key
WECHAT_UPLOAD_URL=https://qyapi.weixin.qq.com/cgi-bin/webhook/upload_media?key=你的key

然后运行时加 `--notify` 参数。