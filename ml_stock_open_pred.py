import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve

# --------- 技术指标函数（与前文一致） ---------
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

# --------- 特征工程 ---------
def feature_engineering(df):
    lag_features = ['pct_chg', 'vol', 'amount', 'turnover_rate', 'volume_ratio', 'pe', 'pb']
    for lag in [2, 3, 5, 10]:
        for col in lag_features:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    rolling_features = ['close', 'vol', 'pct_chg', 'turnover_rate']
    for window in [5, 10, 20]:
        for col in rolling_features:
            df[f'{col}_roll{window}_mean'] = df[col].shift(1).rolling(window).mean()
            df[f'{col}_roll{window}_std'] = df[col].shift(1).rolling(window).std()
            df[f'{col}_roll{window}_max'] = df[col].shift(1).rolling(window).max()
            df[f'{col}_roll{window}_min'] = df[col].shift(1).rolling(window).min()
    df['rsi_14'] = calculate_rsi(df['close'].shift(1), period=14)
    macd, macd_signal, macd_hist = calculate_macd(df['close'].shift(1))
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['close'].shift(1))
    df['bb_width'] = (bb_upper - bb_lower) / (bb_middle + 1e-6)
    df['circ_mv_ratio'] = df['circ_mv'].shift(1) / (df['total_mv'].shift(1) + 1e-6)
    df['volprice_div'] = df['volume_ratio'].shift(1) * df['pct_chg'].shift(1)
    return df

# --------- 数据集构造 ---------
class StockSequenceDataset(Dataset):
    def __init__(self, df, seq_len=30):
        self.seq_len = seq_len
        self.features = [c for c in df.columns if c not in ['y']]
        self.X = []
        self.y = []
        arr = df[self.features].values
        labels = df['y'].values
        for i in range(seq_len, len(df)):
            self.X.append(arr[i-seq_len:i])
            self.y.append(labels[i])
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --------- LSTM/GRU模型 ---------
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时刻
        out = self.fc(out)
        return out.squeeze(-1)

# --------- 主流程 ---------
def main():
    df = pd.read_csv('601988.SH.csv')
    df['y'] = (df['close'] > df['pre_close']).astype(int)
    df = feature_engineering(df)
    drop_cols = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']
    df = df.drop(columns=drop_cols)
    for col in ['pe', 'pb', 'ps']:
        df[col] = df[col].fillna(method='ffill')
    df = df.dropna()
    scaler = StandardScaler()
    feature_cols = [c for c in df.columns if c not in ['y']]
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    # 时间序列分割
    n = len(df)
    split = int(n * 0.8)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    # 构造序列数据
    seq_len = 30
    train_set = StockSequenceDataset(train_df, seq_len=seq_len)
    test_set = StockSequenceDataset(test_df, seq_len=seq_len)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    # 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMClassifier(input_dim=len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    # 训练
    best_auc = 0
    patience, patience_limit = 0, 10
    for epoch in range(50):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        # 验证
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                logits = model(X)
                prob = torch.sigmoid(logits).cpu().numpy()
                y_pred.extend(prob)
                y_true.extend(y.numpy())
        auc = roc_auc_score(y_true, y_pred)
        print(f'Epoch {epoch+1}, AUC={auc:.3f}')
        if auc > best_auc:
            best_auc = auc
            patience = 0
        else:
            patience += 1
        if patience >= patience_limit:
            print('Early stopping.')
            break
    # 评估
    y_pred_bin = (np.array(y_pred) > 0.5).astype(int)
    f1 = f1_score(y_true, y_pred_bin)
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    idx = np.where(recall >= 0.8)[0][0] if np.any(recall >= 0.8) else -1
    print(f'Final AUC: {best_auc:.3f}, F1: {f1:.3f}')
    if idx != -1:
        print(f'Precision@Recall=80%: {precision[idx]:.3f}')
    else:
        print('Recall未达到80%')

if __name__ == '__main__':
    main()