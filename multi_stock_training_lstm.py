import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import glob
from tqdm import tqdm
import joblib

# 设置环境变量，避免并行处理问题
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# --------- 技术指标函数 ---------
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-6)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fastperiod=12, slowperiod=26, signalperiod=9):
    ema_fast = prices.ewm(span=fastperiod).mean()
    ema_slow = prices.ewm(span=slowperiod).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signalperiod).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def calculate_bollinger_bands(prices, period=20, nbdev=2):
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = middle + (std * nbdev)
    lower = middle - (std * nbdev)
    return upper, middle, lower

def calculate_stochastic(df, k_period=14, d_period=3):
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    k = 100 * ((df['close'] - low_min) / (high_max - low_min + 1e-6))
    d = k.rolling(window=d_period).mean()
    return k, d

def calculate_williams_r(df, period=14):
    high_max = df['high'].rolling(window=period).max()
    low_min = df['low'].rolling(window=period).min()
    wr = -100 * ((high_max - df['close']) / (high_max - low_min + 1e-6))
    return wr

# --------- 时序特征工程 ---------
def create_time_series_features(df, lookback_window=30):
    """创建时序特征，考虑历史窗口"""
    print(f"创建时序特征，历史窗口: {lookback_window}")
    
    # 基础价格特征
    if 'close' in df.columns:
        # 价格变化率
        df['price_change'] = df['close'].pct_change()
        df['price_change_2d'] = df['close'].pct_change(2)
        df['price_change_5d'] = df['close'].pct_change(5)
        
        # 移动平均
        for window in [5, 10, 20, 30]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            df[f'price_sma_{window}_ratio'] = df['close'] / df[f'sma_{window}']
        
        # 技术指标
        df['rsi_14'] = calculate_rsi(df['close'], period=14)
        df['rsi_21'] = calculate_rsi(df['close'], period=21)
        
        macd, macd_signal, macd_hist = calculate_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['close'])
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-6)
    
    # 成交量特征
    if 'vol' in df.columns:
        df['vol_ma_5'] = df['vol'].rolling(window=5).mean()
        df['vol_ma_10'] = df['vol'].rolling(window=10).mean()
        df['vol_ratio'] = df['vol'] / df['vol_ma_5']
        df['vol_change'] = df['vol'].pct_change()
    
    # 时序统计特征
    for window in [5, 10, 20]:
        if 'close' in df.columns:
            df[f'price_volatility_{window}'] = df['close'].rolling(window=window).std()
            df[f'price_momentum_{window}'] = df['close'] - df['close'].shift(window)
            df[f'price_acceleration_{window}'] = df['price_change'].rolling(window=window).mean()
        
        if 'vol' in df.columns:
            df[f'vol_volatility_{window}'] = df['vol'].rolling(window=window).std()
            df[f'vol_momentum_{window}'] = df['vol'] - df['vol'].shift(window)
    
    # 趋势特征
    if 'close' in df.columns:
        # 趋势强度
        df['trend_strength_5'] = (df['close'] > df['close'].shift(1)).rolling(5).sum()
        df['trend_strength_10'] = (df['close'] > df['close'].shift(1)).rolling(10).sum()
        
        # 价格位置
        df['price_position_20'] = (df['close'] - df['close'].rolling(20).min()) / (df['close'].rolling(20).max() - df['close'].rolling(20).min() + 1e-6)
        df['price_position_50'] = (df['close'] - df['close'].rolling(50).min()) / (df['close'].rolling(50).max() - df['close'].rolling(50).min() + 1e-6)
    
    # 交互特征
    if 'rsi_14' in df.columns and 'macd' in df.columns:
        df['rsi_macd_interaction'] = df['rsi_14'] * df['macd'] / 100
    
    if 'bb_position' in df.columns and 'rsi_14' in df.columns:
        df['bb_rsi_interaction'] = df['bb_position'] * df['rsi_14'] / 100
    
    return df

# --------- 时序数据集类 ---------
class TimeSeriesDataset(Dataset):
    def __init__(self, data, lookback_window=30, target_col='y', feature_cols=None):
        self.lookback_window = lookback_window
        self.target_col = target_col
        
        if feature_cols is None:
            self.feature_cols = [col for col in data.columns if col not in [target_col, 'stock_code', 'trade_date']]
        else:
            self.feature_cols = feature_cols
        
        # 按股票分组创建序列
        self.sequences = []
        self.targets = []
        
        for stock_code in data['stock_code'].unique():
            stock_data = data[data['stock_code'] == stock_code].sort_values('trade_date' if 'trade_date' in data.columns else 'index')
            
            if len(stock_data) < lookback_window + 1:
                continue
            
            # 提取特征和目标
            features = stock_data[self.feature_cols].values
            targets = stock_data[target_col].values
            
            # 创建滑动窗口序列
            for i in range(lookback_window, len(features)):
                sequence = features[i-lookback_window:i]
                target = targets[i]
                
                # 检查序列中是否有NaN
                if not np.isnan(sequence).any() and not np.isnan(target):
                    self.sequences.append(sequence)
                    self.targets.append(target)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        target = torch.FloatTensor([self.targets[idx]])
        return sequence, target

# --------- LSTM + Attention 模型 ---------
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, hidden_states):
        # hidden_states: (batch_size, seq_len, hidden_size)
        attention_weights = self.attention(hidden_states)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # 加权平均
        context = torch.sum(attention_weights * hidden_states, dim=1)  # (batch_size, hidden_size)
        return context, attention_weights

class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3, transformer_layers=1, nhead=4):
        super(LSTMAttentionModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        if num_layers > 1:
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=True
            )
        else:
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True
            )
        
        # LayerNorm层（加在LSTM和Transformer之间）
        self.ln_before_transformer = nn.LayerNorm(hidden_size * 2)
        
        # Transformer层（插入在LSTM和Attention之间）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size * 2,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # Attention层
        self.attention = AttentionLayer(hidden_size * 2)  # 双向LSTM
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() < 2:
                    # 1维权重（如LayerNorm、BatchNorm），用1初始化
                    nn.init.constant_(param, 1)
                elif 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size * 2)
        
        # LayerNorm
        lstm_out = self.ln_before_transformer(lstm_out)
        
        # Transformer前向传播
        transformer_out = self.transformer(lstm_out)  # (batch_size, seq_len, hidden_size * 2)
        
        # Attention机制
        context, attention_weights = self.attention(transformer_out)
        
        # 分类
        output = self.classifier(context)
        
        return output, attention_weights

# --------- 训练函数 ---------
def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.0005):
    """训练模型"""
    print("开始训练模型...")
    
    # 使用更适合二分类的损失函数
    criterion = nn.BCEWithLogitsLoss()
    
    # 降低学习率，增加权重衰减
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 更激进的学习率调度
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.3, patience=3, min_lr=1e-6
    )
    
    train_losses = []
    val_losses = []
    val_aucs = []
    
    best_val_auc = 0
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 15
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []
        
        for sequences, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(sequences)
            loss = criterion(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy())
            train_targets.extend(targets.detach().cpu().numpy())
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                
                outputs, _ = model(sequences)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                val_preds.extend(outputs.detach().cpu().numpy())
                val_targets.extend(targets.detach().cpu().numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # 计算AUC
        val_auc = roc_auc_score(val_targets, val_preds)
        val_aucs.append(val_auc)
        
        # 学习率调度
        scheduler.step(val_auc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Val AUC: {val_auc:.4f}')
        
        # 早停 - 同时考虑AUC和Loss
        improved = False
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            improved = True
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            improved = True
        
        if improved:
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'  ✅ 模型改进 - 最佳AUC: {best_val_auc:.4f}, 最佳Loss: {best_val_loss:.4f}')
        else:
            patience_counter += 1
            print(f'  ⏳ 等待改进 ({patience_counter}/{early_stopping_patience})')
            
        if patience_counter >= early_stopping_patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    
    return model, train_losses, val_losses, val_aucs

# --------- 数据加载和预处理 ---------
def load_and_preprocess_data(sector_name):
    """加载并预处理指定板块的数据"""
    print(f"开始加载 {sector_name} 板块数据...")
    
    # 查找多模态特征文件
    multimodal_root_path = './dataset/process_text_add_tom/'
    multimodal_path = os.path.join(multimodal_root_path, sector_name, 'multimodal.csv')
    
    if not os.path.exists(multimodal_path):
        print(f"错误：找不到 {sector_name} 板块的数据文件")
        return None
    
    # 加载数据
    df = pd.read_csv(multimodal_path)
    
    # 确保必要的列存在
    if 'y' not in df.columns or 'stock_code' not in df.columns:
        print(f"错误：{sector_name} 板块缺少必要列")
        return None
    
    # 添加板块信息
    df['sector'] = sector_name
    
    # 确保数据按时间排序
    if 'trade_date' in df.columns:
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values(['stock_code', 'trade_date'])
    else:
        # 如果没有日期列，按索引排序
        df = df.sort_values(['stock_code']).reset_index(drop=True)
    
    print(f"  {sector_name}: {len(df)} 行，{df['stock_code'].nunique()} 只股票")
    print(f"  标签分布: {df['y'].value_counts().to_dict()}")
    
    return df

def get_all_sectors():
    """获取所有可用的板块"""
    multimodal_root_path = './dataset/process_text_add_tom/'
    sectors = []
    
    for sector in os.listdir(multimodal_root_path):
        multimodal_path = os.path.join(multimodal_root_path, sector, 'multimodal.csv')
        if os.path.exists(multimodal_path):
            sectors.append(sector)
    
    return sectors

def check_data_leakage(df, feature_cols):
    """检查数据泄露"""
    print("检查数据泄露...")
    
    # 检查特征与标签的相关性
    high_corr_features = []
    for col in feature_cols:
        corr = abs(df[col].corr(df['y']))
        if corr > 0.8:  # 高相关性阈值
            high_corr_features.append((col, corr))
    
    if high_corr_features:
        print(f"⚠️  发现 {len(high_corr_features)} 个高相关特征，可能存在数据泄露:")
        for col, corr in high_corr_features[:5]:  # 只显示前5个
            print(f"    {col}: {corr:.4f}")
        return True
    else:
        print("✅ 未发现明显的数据泄露")
        return False

# --------- 主流程 ---------

def get_nhead(hidden_size):
    # 取能整除hidden_size*2的最大不超过8的nhead
    for nh in [8, 4, 2, 1]:
        if (hidden_size * 2) % nh == 0:
            return nh
    return 1

def train_sector_model(sector_name):
    """训练单个板块的模型"""
    print("="*60)
    print(f"训练 {sector_name} 板块时序股票预测模型")
    print("="*60)
    
    # 1. 加载数据
    df = load_and_preprocess_data(sector_name)
    if df is None:
        return None
    
    # 2. 时序特征工程
    print("\n开始时序特征工程...")
    df = create_time_series_features(df, lookback_window=30)
    
    # 3. 数据清理
    print("数据清理...")
    # 删除非数值列
    non_numeric_cols = ['stock_code', 'sector', 'y']
    if 'trade_date' in df.columns:
        non_numeric_cols.append('trade_date')
    
    feature_cols = [col for col in df.columns if col not in non_numeric_cols]
    
    # 处理缺失值
    for col in feature_cols:
        df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # 删除仍有缺失值的行
    df = df.dropna()
    print(f"清理后数据形状: {df.shape}")
    
    # 4. 特征质量检查和选择
    print("特征质量检查...")
    
    # 检查特征与标签的相关性
    correlations = []
    for col in feature_cols:
        corr = abs(df[col].corr(df['y']))
        correlations.append((col, corr))
    
    # 按相关性排序
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    # 选择相关性较高的特征（前100个）
    top_features = [col for col, corr in correlations[:100]]
    print(f"选择前100个最相关特征，相关性范围: {correlations[0][1]:.4f} - {correlations[99][1]:.4f}")
    
    # 更新特征列
    feature_cols = top_features
    
    # 检查数据泄露
    has_leakage = check_data_leakage(df, feature_cols)
    if has_leakage:
        print("⚠️  警告：检测到可能的数据泄露，建议检查特征工程过程")
    
    # 特征标准化
    print("特征标准化...")
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # 5. 创建时序数据集
    print("创建时序数据集...")
    lookback_window = 30
    
    # 按股票分组创建数据集
    train_sequences = []
    train_targets = []
    val_sequences = []
    val_targets = []
    test_sequences = []
    test_targets = []
    test_stock_codes = []
    
    for stock_code in df['stock_code'].unique():
        stock_data = df[df['stock_code'] == stock_code].copy()
        
        if len(stock_data) < lookback_window + 50:  # 确保有足够的数据
            continue
        
        # 提取特征和目标
        features = stock_data[feature_cols].values
        targets = stock_data['y'].values
        
        # 按时间分割：70% 训练，15% 验证，15% 测试
        n = len(features)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        
        # 训练集
        for i in range(lookback_window, train_end):
            if not np.isnan(features[i-lookback_window:i]).any() and not np.isnan(targets[i]):
                train_sequences.append(features[i-lookback_window:i])
                train_targets.append(targets[i])
        
        # 验证集
        for i in range(train_end, val_end):
            if not np.isnan(features[i-lookback_window:i]).any() and not np.isnan(targets[i]):
                val_sequences.append(features[i-lookback_window:i])
                val_targets.append(targets[i])
        
        # 测试集
        for i in range(val_end, n):
            if not np.isnan(features[i-lookback_window:i]).any() and not np.isnan(targets[i]):
                test_sequences.append(features[i-lookback_window:i])
                test_targets.append(targets[i])
                test_stock_codes.append(stock_code)
    
    print(f"训练集: {len(train_sequences)} 样本")
    print(f"验证集: {len(val_sequences)} 样本")
    print(f"测试集: {len(test_sequences)} 样本")
    
    if len(train_sequences) < 100:  # 数据太少，跳过
        print(f"警告：{sector_name} 板块训练样本太少，跳过训练")
        return None
    
    # 检查和修正标签与特征
    def clean_targets(targets, name):
        targets = np.array(targets).astype(np.float32)
        # 清理inf/nan
        targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
        print(f'{name} unique:', np.unique(targets))
        print(f'{name} dtype:', targets.dtype)
        if not set(np.unique(targets)).issubset({0.0, 1.0}):
            raise ValueError(f"{name}标签异常: {np.unique(targets)}")
        return targets
    def clean_sequences(sequences, name):
        arr = np.array(sequences)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    train_targets = clean_targets(train_targets, 'train_targets')
    val_targets = clean_targets(val_targets, 'val_targets')
    test_targets = clean_targets(test_targets, 'test_targets')
    train_sequences = clean_sequences(train_sequences, 'train_sequences')
    val_sequences = clean_sequences(val_sequences, 'val_sequences')
    test_sequences = clean_sequences(test_sequences, 'test_sequences')
    
    # 6. 创建数据加载器
    batch_size = 64
    
    # 直接使用序列和目标，而不是通过TimeSeriesDataset
    train_sequences_tensor = torch.FloatTensor(train_sequences)
    train_targets_tensor = torch.FloatTensor(train_targets).unsqueeze(1)
    val_sequences_tensor = torch.FloatTensor(val_sequences)
    val_targets_tensor = torch.FloatTensor(val_targets).unsqueeze(1)
    
    # 创建简单的数据集类
    class SimpleDataset(Dataset):
        def __init__(self, sequences, targets):
            self.sequences = sequences
            self.targets = targets
        
        def __len__(self):
            return len(self.sequences)
        
        def __getitem__(self, idx):
            return self.sequences[idx], self.targets[idx]
    
    train_dataset = SimpleDataset(train_sequences_tensor, train_targets_tensor)
    val_dataset = SimpleDataset(val_sequences_tensor, val_targets_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 7. 创建模型
    input_size = len(feature_cols)
    
    # 根据数据量调整模型复杂度
    if len(train_sequences) < 5000:
        # 小数据集，使用简单模型
        hidden_size = 64
        num_layers = 1
        dropout = 0.2
        transformer_layers = 1
        nhead = get_nhead(hidden_size)
    elif len(train_sequences) < 15000:
        # 中等数据集
        hidden_size = 96
        num_layers = 2
        dropout = 0.3
        transformer_layers = 1
        nhead = get_nhead(hidden_size)
    else:
        # 大数据集
        hidden_size = 128
        num_layers = 2
        dropout = 0.3
        transformer_layers = 1
        nhead = get_nhead(hidden_size)
    
    model = LSTMAttentionModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        transformer_layers=transformer_layers,
        nhead=nhead
    ).to(device)
    
    print(f"模型配置: hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout}, transformer_layers={transformer_layers}, nhead={nhead}")
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 8. 训练模型
    model, train_losses, val_losses, val_aucs = train_model(
        model, train_loader, val_loader, num_epochs=50, learning_rate=0.001
    )
    
    # 9. 测试模型
    print("\n测试模型性能...")
    model.eval()
    test_preds = []
    
    with torch.no_grad():
        for i in range(0, len(test_sequences), batch_size):
            batch_sequences = test_sequences[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch_sequences).to(device)
            
            outputs, _ = model(batch_tensor)
            probs = torch.sigmoid(outputs)
            test_preds.extend(probs.detach().cpu().numpy())
    
    test_preds = np.array(test_preds).flatten()
    test_targets = np.array(test_targets)
    
    # 计算指标
    test_auc = roc_auc_score(test_targets, test_preds)
    test_f1 = f1_score(test_targets, test_preds > 0.5)
    
    print(f"测试集性能:")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  F1: {test_f1:.4f}")
    
    # 10. 可视化结果
    print("\n生成可视化图表...")
    
    plt.figure(figsize=(15, 10))
    
    # 训练曲线
    plt.subplot(2, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'{sector_name} - 训练损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # AUC曲线
    plt.subplot(2, 3, 2)
    plt.plot(val_aucs, label='Val AUC')
    plt.title(f'{sector_name} - 验证AUC曲线')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    
    # ROC曲线
    plt.subplot(2, 3, 3)
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(test_targets, test_preds)
    plt.plot(fpr, tpr, label=f'LSTM+Attention (AUC={test_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{sector_name} - ROC曲线')
    plt.legend()
    plt.grid(True)
    
    # 预测概率分布
    plt.subplot(2, 3, 4)
    plt.hist(test_preds[test_targets == 0], bins=30, alpha=0.7, label='y=0', color='red')
    plt.hist(test_preds[test_targets == 1], bins=30, alpha=0.7, label='y=1', color='blue')
    plt.xlabel('预测概率')
    plt.ylabel('样本数')
    plt.title(f'{sector_name} - 预测概率分布')
    plt.legend()
    
    # 混淆矩阵
    plt.subplot(2, 3, 5)
    from sklearn.metrics import confusion_matrix
    y_pred = (test_preds > 0.5).astype(int)
    cm = confusion_matrix(test_targets, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{sector_name} - 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    
    # 按股票分析
    plt.subplot(2, 3, 6)
    stock_performance = {}
    for i, stock_code in enumerate(test_stock_codes):
        if stock_code not in stock_performance:
            stock_performance[stock_code] = {'preds': [], 'targets': []}
        stock_performance[stock_code]['preds'].append(test_preds[i])
        stock_performance[stock_code]['targets'].append(test_targets[i])
    
    stock_aucs = []
    for stock_code, data in stock_performance.items():
        if len(data['targets']) >= 10:  # 至少10个样本
            try:
                auc = roc_auc_score(data['targets'], data['preds'])
                stock_aucs.append(auc)
            except:
                continue
    
    plt.hist(stock_aucs, bins=20, alpha=0.7, color='green')
    plt.xlabel('股票AUC')
    plt.ylabel('股票数量')
    plt.title(f'{sector_name} - 各股票AUC分布')
    
    plt.tight_layout()
    
    # 保存到板块特定目录
    multimodal_root_path = './dataset/process_text_add_tom/'
    sector_path = os.path.join(multimodal_root_path, sector_name)
    plt_save_path = os.path.join(sector_path, f'{sector_name}_lstm_attention_results.png')
    plt.savefig(plt_save_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    # 11. 保存结果
    print("\n保存结果...")
    
    results_df = pd.DataFrame({
        'stock_code': test_stock_codes,
        'true_label': test_targets,
        'predicted_probability': test_preds,
        'predicted_label': (test_preds > 0.5).astype(int)
    })
    
    # 保存到板块特定目录
    results_save_path = os.path.join(sector_path, f'{sector_name}_lstm_attention_predictions.csv')
    results_df.to_csv(results_save_path, index=False)
    
    # 保存模型和预处理器
    model_save_path = os.path.join(sector_path, f'{sector_name}_lstm_attention_model.pth')
    scaler_save_path = os.path.join(sector_path, f'{sector_name}_lstm_attention_scaler.pkl')
    features_save_path = os.path.join(sector_path, f'{sector_name}_lstm_attention_features.pkl')
    
    torch.save(model.state_dict(), model_save_path)
    joblib.dump(scaler, scaler_save_path)
    joblib.dump(feature_cols, features_save_path)
    
    print("结果已保存:")
    print(f"- {sector_name}_lstm_attention_predictions.csv: 预测结果")
    print(f"- {sector_name}_lstm_attention_model.pth: 模型权重")
    print(f"- {sector_name}_lstm_attention_scaler.pkl: 特征标准化器")
    print(f"- {sector_name}_lstm_attention_features.pkl: 特征列名")
    print(f"- {sector_name}_lstm_attention_results.png: 可视化图表")
    
    print("\n" + "="*60)
    print(f"{sector_name} 板块时序股票预测模型训练完成！")
    print("="*60)
    
    return {
        'sector': sector_name,
        'auc': test_auc,
        'f1': test_f1,
        'test_samples': len(test_sequences)
    }

def main():
    """主函数：按板块分别训练"""
    print("="*60)
    print("时序股票预测模型训练 - 按板块分别训练")
    print("="*60)
    
    # 获取所有可用板块
    sectors = get_all_sectors()
    print(f"找到 {len(sectors)} 个可用板块: {sectors}")
    
    # 存储所有板块的结果
    all_results = []
    
    # 按板块分别训练
    for i, sector in enumerate(sectors):
        print(f"\n{'='*20} 训练第 {i+1}/{len(sectors)} 个板块: {sector} {'='*20}")
        
        try:
            result = train_sector_model(sector)
            if result:
                all_results.append(result)
                print(f"✅ {sector} 板块训练成功")
            else:
                print(f"❌ {sector} 板块训练失败")
        except Exception as e:
            print(f"❌ {sector} 板块训练出错: {e}")
            continue
        
        # 清理内存
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 汇总结果
    print("\n" + "="*60)
    print("所有板块训练完成！汇总结果:")
    print("="*60)
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        print(results_df.sort_values('auc', ascending=False))
        
        # 保存汇总结果
        results_df.to_csv('all_sectors_results.csv', index=False)
        print(f"\n汇总结果已保存到: all_sectors_results.csv")
        
        # 计算平均性能
        avg_auc = results_df['auc'].mean()
        avg_f1 = results_df['f1'].mean()
        print(f"\n平均性能:")
        print(f"  平均AUC: {avg_auc:.4f}")
        print(f"  平均F1: {avg_f1:.4f}")
    else:
        print("没有成功训练的板块")
    
    print("\n" + "="*60)
    print("所有板块训练完成！")
    print("="*60)

if __name__ == '__main__':
    main() 