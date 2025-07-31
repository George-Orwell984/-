# XGBoost+LightGBM 股票预测系统

## 项目概述

这是一个基于深度学习和机器学习技术的多股票预测系统，结合了LSTM、注意力机制、XGBoost和LightGBM等多种算法，用于预测股票价格走势。项目支持多行业、多股票的训练和预测，并提供了完整的数据处理、模型训练和推理流程。

## 主要特性

- **多模型融合**: 结合LSTM、注意力机制、XGBoost和LightGBM
- **多行业支持**: 支持银行、券商、互联网、房地产等20个行业
- **时序特征工程**: 丰富的技术指标和时序特征
- **实时预测**: 支持单股票和多股票批量预测
- **高性能**: 优化的模型架构和训练流程
- **完整评估**: 提供详细的模型性能评估指标

## 项目结构

```
XGBoost+LightGBM/
├── dataset/                          # 数据集目录
│   ├── 2007-2025-no_news/           # 无新闻数据
│   ├── 2017-2023-news/              # 包含新闻数据
│   ├── process_notext_tom/          # 无文本处理数据
│   ├── process_text_add_tom/        # 添加文本处理数据
│   └── test/                        # 测试数据
├── model.py                         # 模型定义文件
├── utils.py                         # 工具函数
├── Tushare.py                       # 数据获取脚本
├── multi_stock_training_all.py      # 多股票训练主程序
├── multi_stock_inference_all.py     # 多股票推理程序
├── single_stock_inference.py        # 单股票推理程序
├── process_text_all.py              # 文本数据处理
├── diagnose_training.py             # 训练诊断工具
└── README.md                        # 项目说明文档
```

## 核心模型架构

### 1. PyramidConvLSTMAttentionModel
- **卷积层**: 多尺度卷积核(3,7,15)提取局部特征
- **LSTM层**: 双向LSTM捕获时序依赖
- **注意力机制**: 自适应权重分配
- **全连接层**: 最终预测输出

### 2. 特征工程
- **技术指标**: RSI、MACD、布林带、随机指标等
- **时序特征**: 移动平均、价格变化率、波动率
- **多尺度特征**: 不同时间窗口的特征组合

## 安装依赖

```bash
pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn
pip install xgboost lightgbm
pip install matplotlib seaborn tqdm
pip install tushare rqdatac
pip install joblib
```

## 使用方法

### 1. 数据获取

```python
# 使用Tushare获取股票数据
python Tushare.py
```

### 2. 模型训练

```python
# 训练所有行业的模型
python multi_stock_training_all.py
```
# 训练所有行业的模型
python multi_stock_training_lstm.py

### 3. 模型推理

```python
# 多股票批量预测
python multi_stock_inference_all.py

# 单股票预测
python single_stock_inference.py
```

## 模型性能

根据最新测试结果，各行业模型性能如下：

| 行业 | AUC | F1-Score | 测试样本数 |
|------|-----|----------|------------|
| 银行 | 0.808 | 0.673 | 4569 |
| 券商 | 0.787 | 0.646 | 3980 |
| 酿酒 | 0.633 | 0.575 | 3268 |
| 航天 | 0.637 | 0.567 | 1315 |
| 煤炭 | 0.585 | 0.524 | 2330 |
| 家电 | 0.600 | 0.502 | 1933 |
| 影视动漫 | 0.567 | 0.525 | 1480 |
| 贸易 | 0.552 | 0.507 | 1366 |
| 机械机床 | 0.542 | 0.491 | 1141 |
| 轻工 | 0.550 | 0.468 | 1232 |

## 主要功能模块

### 1. 数据处理模块 (`Tushare.py`)
- 从Tushare API获取股票数据
- 支持多数据源整合
- 自动数据清洗和格式化

### 2. 特征工程模块 (`utils.py`)
- 技术指标计算
- 数据标准化
- 时序特征创建

### 3. 模型定义模块 (`model.py`)
- 注意力机制实现
- LSTM模型定义
- XGBoost集成

### 4. 训练模块 (`multi_stock_training_all.py`)
- 多行业并行训练
- 模型验证和评估
- 超参数优化

### 5. 推理模块 (`multi_stock_inference_all.py`)
- 批量预测
- 性能评估
- 结果可视化

## 配置说明

### 数据配置
- 支持多种数据源：Tushare、RQData等
- 可配置数据时间范围
- 支持实时数据更新

### 模型配置
- 可调整LSTM层数和隐藏单元数
- 支持不同的注意力机制
- 可配置训练参数

### 训练配置
- 支持GPU/CPU训练
- 可配置批次大小和学习率
- 支持早停和模型保存

## 注意事项

1. **数据依赖**: 需要配置Tushare和RQData的API密钥
2. **硬件要求**: 建议使用GPU进行训练
3. **内存要求**: 大数据集需要足够的内存
4. **时间要求**: 完整训练可能需要数小时

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。

## 许可证

本项目采用MIT许可证。

## 联系方式

如有问题，请通过GitHub Issues联系。

---

**免责声明**: 本项目仅供学习和研究使用，不构成投资建议。股票投资存在风险，请谨慎决策。 