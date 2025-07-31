import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, classification_report
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, f_classif
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
# import gc0
import glob
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_ind

# 设置环境变量，避免并行处理问题
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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

# --------- 数据加载和合并 ---------
def load_and_merge_stock_data():
    """加载并合并多个股票数据"""
    print("开始加载多个股票数据...")
    
    # 查找所有股票数据文件
    stock_files = glob.glob('./dataset/2007-2025-no_news/生物医药/*.SH.csv') + glob.glob('./dataset/2007-2025-no_news/生物医药/*.SZ.csv')
    print(f"找到 {len(stock_files)} 个股票数据文件: {stock_files}")
    
    all_data = []
    
    for file in stock_files:
        try:
            print(f"正在加载 {file}...")
            df = pd.read_csv(file)
            
            # 检查数据质量
            if len(df) < 100:  # 数据太少，跳过
                print(f"跳过 {file}：数据量太少 ({len(df)} 行)")
                continue
                
            # 检查必要的列是否存在
            required_cols = ['close', 'pre_close', 'pct_chg', 'vol', 'amount']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"跳过 {file}：缺少必要列 {missing_cols}")
                continue
            
            # 添加股票代码标识
            stock_code = file.split('.')[0]
            df['stock_code'] = stock_code
            
            # df['y'] = (df['close'] > df['pre_close']).astype(int)
            df['next_close'] = df['close'].shift(-1)
            df['y'] = (df['next_close'] > df['close']).astype(int)
            df = df.dropna(subset=['next_close'])
            df = df.iloc[:-1]  # 去掉最后一行（没有next_close，标签无效）
            
            # 检查标签分布
            label_dist = df['y'].value_counts()
            if label_dist.min() < 50:  # 某个类别样本太少
                print(f"跳过 {file}：标签分布不平衡 {label_dist.to_dict()}")
                continue
            
            all_data.append(df)
            print(f"成功加载 {file}：{len(df)} 行，标签分布 {label_dist.to_dict()}")
            
        except Exception as e:
            print(f"加载 {file} 时出错: {e}")
            continue
    
    if not all_data:
        print("错误：没有成功加载任何数据文件！")
        return None
    
    # 合并所有数据
    print("合并所有股票数据...")
    merged_df = pd.concat(all_data, ignore_index=True)
    
    print(f"合并后数据形状: {merged_df.shape}")
    print(f"包含的股票: {merged_df['stock_code'].unique()}")
    print(f"总体标签分布: {merged_df['y'].value_counts().to_dict()}")
    
    return merged_df

# --------- 增强特征工程 ---------
def enhanced_feature_engineering(df):
    print("开始增强特征工程...")
    
    # 检查必要的列是否存在
    required_cols = ['close', 'vol', 'amount', 'pct_chg']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"警告：缺少必要列: {missing_cols}")
        # 为缺失的列创建默认值
        for col in missing_cols:
            if col in ['close', 'vol', 'amount']:
                df[col] = 0
            else:
                df[col] = 0.0
    
    # 1. 基础滞后特征
    lag_features = ['pct_chg', 'vol', 'amount']
    if 'turnover_rate' in df.columns:
        lag_features.append('turnover_rate')
    if 'volume_ratio' in df.columns:
        lag_features.append('volume_ratio')
    if 'pe' in df.columns:
        lag_features.append('pe')
    if 'pb' in df.columns:
        lag_features.append('pb')
    
    for lag in [1, 2, 3, 5, 10]:
        for col in lag_features:
            if col in df.columns:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    # 2. 滚动统计特征
    rolling_features = ['close', 'vol', 'pct_chg', 'amount']
    for window in [3, 5, 7, 10, 15]:
        for col in rolling_features:
            if col in df.columns:
                df[f'{col}_roll{window}_mean'] = df[col].shift(1).rolling(window, min_periods=1).mean()
                df[f'{col}_roll{window}_std'] = df[col].shift(1).rolling(window, min_periods=1).std()
                df[f'{col}_roll{window}_max'] = df[col].shift(1).rolling(window, min_periods=1).max()
                df[f'{col}_roll{window}_min'] = df[col].shift(1).rolling(window, min_periods=1).min()
    
    # 3. 趋势相关指标
    if 'close' in df.columns:
        # Simple Moving Average (SMA)
        for window in [5, 10, 20, 30, 60]:
            df[f'sma_{window}'] = df['close'].shift(1).rolling(window, min_periods=1).mean()
        # Exponential Moving Average (EMA)
        for span in [5, 10, 20, 30, 60]:
            df[f'ema_{span}'] = df['close'].shift(1).ewm(span=span, adjust=False).mean()
        # MACD
        macd, macd_signal, macd_hist = calculate_macd(df['close'].shift(1))
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        df['macd_ratio'] = macd / (df['close'].shift(1) + 1e-6)
        # ADX
        if all(col in df.columns for col in ['high', 'low', 'close']):
            high = df['high'].shift(1)
            low = df['low'].shift(1)
            close = df['close'].shift(1)
            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14, min_periods=1).mean()
            plus_di = 100 * (plus_dm.rolling(14, min_periods=1).sum() / (atr + 1e-6))
            minus_di = 100 * (minus_dm.rolling(14, min_periods=1).sum() / (atr + 1e-6))
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-6)
            df['adx'] = dx.rolling(14, min_periods=1).mean()
        # Ichimoku Cloud
        if all(col in df.columns for col in ['high', 'low', 'close']):
            high = df['high']
            low = df['low']
            # Tenkan-sen (Conversion Line)
            period9_high = high.rolling(window=9, min_periods=1).max()
            period9_low = low.rolling(window=9, min_periods=1).min()
            df['ichimoku_tenkan_sen'] = (period9_high + period9_low) / 2
            # Kijun-sen (Base Line)
            period26_high = high.rolling(window=26, min_periods=1).max()
            period26_low = low.rolling(window=26, min_periods=1).min()
            df['ichimoku_kijun_sen'] = (period26_high + period26_low) / 2
            # Senkou Span A (Leading Span A)
            df['ichimoku_senkou_span_a'] = ((df['ichimoku_tenkan_sen'] + df['ichimoku_kijun_sen']) / 2).shift(26)
            # Senkou Span B (Leading Span B)
            period52_high = high.rolling(window=52, min_periods=1).max()
            period52_low = low.rolling(window=52, min_periods=1).min()
            df['ichimoku_senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)
            # Chikou Span (Lagging Span)
            df['ichimoku_chikou_span'] = df['close'].shift(-26)

    # 4. 技术指标
    if 'close' in df.columns:
        df['rsi_14'] = calculate_rsi(df['close'].shift(1), period=14)
        df['rsi_21'] = calculate_rsi(df['close'].shift(1), period=21)
        
        macd, macd_signal, macd_hist = calculate_macd(df['close'].shift(1))
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        df['macd_ratio'] = macd / (df['close'].shift(1) + 1e-6)
        
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['close'].shift(1))
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / (bb_middle + 1e-6)
        df['bb_position'] = (df['close'].shift(1) - bb_lower) / (bb_upper - bb_lower + 1e-6)
    
    # 随机指标和威廉指标需要high/low列
    if all(col in df.columns for col in ['high', 'low', 'close']):
        k, d = calculate_stochastic(df.shift(1))
        df['stoch_k'] = k
        df['stoch_d'] = d
        df['williams_r'] = calculate_williams_r(df.shift(1))
    
    # 5. 基本面衍生特征
    if 'circ_mv' in df.columns and 'total_mv' in df.columns:
        df['circ_mv_ratio'] = df['circ_mv'].shift(1) / (df['total_mv'].shift(1) + 1e-6)
    if 'volume_ratio' in df.columns and 'pct_chg' in df.columns:
        df['volprice_div'] = df['volume_ratio'].shift(1) * df['pct_chg'].shift(1)
    if 'pe' in df.columns and 'pb' in df.columns:
        df['pe_pb_ratio'] = df['pe'].shift(1) / (df['pb'].shift(1) + 1e-6)
    
    # 6. 序列特有特征
    if 'pct_chg' in df.columns:
        # 趋势连续性
        df['trend_continuity'] = (df['pct_chg'].shift(1) > 0).rolling(5, min_periods=1).sum()
        df['trend_strength'] = df['pct_chg'].shift(1).rolling(5, min_periods=1).apply(lambda x: np.sum(x > 0) - np.sum(x < 0))
        
        # 动量特征
        df['momentum_decay'] = df['pct_chg'].rolling(3, min_periods=1).mean().shift(1) / (df['pct_chg'].rolling(10, min_periods=1).mean().shift(1) + 1e-6)
        df['momentum_acceleration'] = df['pct_chg'].rolling(3, min_periods=1).mean().shift(1) - df['pct_chg'].rolling(7, min_periods=1).mean().shift(1)
    
    if 'vol' in df.columns:
        # 波动率特征
        df['volatility_change'] = df['vol'].rolling(5, min_periods=1).std().shift(1) / (df['vol'].rolling(10, min_periods=1).std().shift(1) + 1e-6)
        df['volume_surge'] = df['vol'].shift(1) / (df['vol'].rolling(10, min_periods=1).mean().shift(1) + 1e-6)
    
    if 'close' in df.columns:
        # 价格位置
        df['price_position'] = (df['close'].shift(1) - df['close'].rolling(20, min_periods=1).min().shift(1)) / (df['close'].rolling(20, min_periods=1).max().shift(1) - df['close'].rolling(20, min_periods=1).min().shift(1) + 1e-6)
        df['price_ma_ratio'] = df['close'].shift(1) / (df['close'].rolling(20, min_periods=1).mean().shift(1) + 1e-6)
        df['price_volatility'] = df['close'].rolling(10, min_periods=1).std().shift(1) / (df['close'].rolling(10, min_periods=1).mean().shift(1) + 1e-6)
    
    # 7. 特征交互
    if 'rsi_14' in df.columns and 'macd' in df.columns:
        df['rsi_macd_ratio'] = df['rsi_14'] * df['macd'] / 100
    if 'bb_position' in df.columns and 'rsi_14' in df.columns:
        df['bb_rsi_position'] = df['bb_position'] * df['rsi_14'] / 100
    if 'volume_surge' in df.columns and 'rsi_14' in df.columns:
        df['vol_rsi_interaction'] = df['volume_surge'] * df['rsi_14'] / 100
    
    # 8. 统计特征
    if 'pct_chg' in df.columns:
        df['pct_chg_zscore'] = (df['pct_chg'].shift(1) - df['pct_chg'].rolling(20, min_periods=1).mean().shift(1)) / (df['pct_chg'].rolling(20, min_periods=1).std().shift(1) + 1e-6)
    if 'vol' in df.columns:
        df['vol_zscore'] = (df['vol'].shift(1) - df['vol'].rolling(20, min_periods=1).mean().shift(1)) / (df['vol'].rolling(20, min_periods=1).std().shift(1) + 1e-6)
    
    print(f"特征工程完成，特征数量: {len([c for c in df.columns if c not in ['y', 'stock_code']])}")
    return df

# --------- 特征选择 ---------
def feature_selection(X_train, y_train, X_test, method='correlation', k=100):
    print(f"开始特征选择，方法: {method}")
    
    if method == 'correlation':
        # 相关性选择
        correlations = np.abs(np.corrcoef(X_train.T, y_train))[:-1, -1]
        top_indices = np.argsort(correlations)[-k:]
        
    elif method == 'mutual_info':
        # 互信息选择
        from sklearn.feature_selection import mutual_info_classif
        mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
        top_indices = np.argsort(mi_scores)[-k:]
        
    elif method == 'f_classif':
        # F统计量选择
        selector = SelectKBest(f_classif, k=k)
        selector.fit(X_train, y_train)
        top_indices = selector.get_support(indices=True)
    
    X_train_selected = X_train[:, top_indices]
    X_test_selected = X_test[:, top_indices]
    
    print(f"特征选择完成，从 {X_train.shape[1]} 个特征中选择 {X_train_selected.shape[1]} 个")
    return X_train_selected, X_test_selected, top_indices

# --------- 模型训练 ---------
def train_models(X_train, y_train, X_test, y_test):
    """训练多个模型"""
    print("开始训练模型...")
    
    models = {}
    results = {}
    
    # LightGBM
    print("训练LightGBM...")
    lgbm = LGBMClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )
    lgbm.fit(X_train, y_train)
    lgbm_proba = lgbm.predict_proba(X_test)[:, 1]
    lgbm_auc = roc_auc_score(y_test, lgbm_proba)
    lgbm_f1 = f1_score(y_test, lgbm_proba > 0.5)
    
    models['LightGBM'] = lgbm
    results['LightGBM'] = (lgbm_proba, lgbm_auc, lgbm_f1)
    print(f"LightGBM - AUC: {lgbm_auc:.3f}, F1: {lgbm_f1:.3f}")
    
    # XGBoost
    print("训练XGBoost...")
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    xgb.fit(X_train, y_train)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_proba)
    xgb_f1 = f1_score(y_test, xgb_proba > 0.5)
    
    models['XGBoost'] = xgb
    results['XGBoost'] = (xgb_proba, xgb_auc, xgb_f1)
    print(f"XGBoost - AUC: {xgb_auc:.3f}, F1: {xgb_f1:.3f}")
    
    # CatBoost
    print("训练CatBoost...")
    cat = CatBoostClassifier(
        iterations=200,
        depth=7,
        learning_rate=0.05,
        l2_leaf_reg=3,
        random_state=42,
        verbose=False
    )
    cat.fit(X_train, y_train)
    cat_proba = cat.predict_proba(X_test)[:, 1]
    cat_auc = roc_auc_score(y_test, cat_proba)
    cat_f1 = f1_score(y_test, cat_proba > 0.5)
    
    models['CatBoost'] = cat
    results['CatBoost'] = (cat_proba, cat_auc, cat_f1)
    print(f"CatBoost - AUC: {cat_auc:.3f}, F1: {cat_f1:.3f}")
    
    # 集成模型
    print("训练集成模型...")
    ensemble = VotingClassifier(
        estimators=[
            ('lgbm', lgbm),
            ('xgb', xgb),
            ('cat', cat)
        ],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    ensemble_proba = ensemble.predict_proba(X_test)[:, 1]
    ensemble_auc = roc_auc_score(y_test, ensemble_proba)
    ensemble_f1 = f1_score(y_test, ensemble_proba > 0.5)
    
    models['Ensemble'] = ensemble
    results['Ensemble'] = (ensemble_proba, ensemble_auc, ensemble_f1)
    print(f"Ensemble - AUC: {ensemble_auc:.3f}, F1: {ensemble_f1:.3f}")
    
    return models, results

# --------- 主流程 ---------
def main():
    print("="*60)
    print("多股票数据集训练")
    print("="*60)

    # 优先加载多模态特征
    multimodal_root_path = 'F:\djy\Attention-CLX-stock-prediction\dataset\process_text_add_tom\\互联网'
    # multimodal_path = os.path.join(multimodal_root_path,'multimodal.csv') 
    multimodal_path = f'F:\djy\Attention-CLX-stock-prediction\dataset\process_text_add_tom\互联网\multimodal.csv'
    # multimodal_path = 'F:\djy\Attention-CLX-stock-prediction\dataset\process_text\\金属制品\multimodal.csv'
    if os.path.exists(multimodal_path):
        print("检测到多模态特征文件，直接加载...")
        df = pd.read_csv(multimodal_path)
        # 保证y和stock_code存在
        assert 'y' in df.columns and 'stock_code' in df.columns
        
        # 先提取y和stock_code，避免参与特征工程
        y = df['y'].values
        stock_code = df['stock_code'].values
        
        # 删除y和stock_code列，对剩余数据进行特征工程
        df_features = df.drop(columns=['y', 'stock_code'])
        
        # 对剩余数据进行特征工程
        # print("对多模态特征进行特征工程...")
        # df_features = enhanced_feature_engineering(df_features)
        
        # 选择数值型特征列
        feature_cols = [c for c in df_features.columns if pd.api.types.is_numeric_dtype(df_features[c])]
        X = df_features[feature_cols].fillna(0).values
        
        # ========== 分析趋势特征与y的关系 ==========
        print("\n==== 分析趋势特征与y的关系 ====")
        trend_features = [
            'sma_5', 'sma_10', 'sma_20', 'sma_30', 'sma_60',
            'ema_5', 'ema_10', 'ema_20', 'ema_30', 'ema_60',
            'macd', 'macd_signal', 'macd_hist', 'adx',
            'ichimoku_tenkan_sen', 'ichimoku_kijun_sen', 'ichimoku_senkou_span_a', 'ichimoku_senkou_span_b', 'ichimoku_chikou_span'
        ]
        trend_features = [f for f in trend_features if f in df_features.columns]
        results = []
        for f in trend_features:
            mean0, mean1 = df[df['y']==0][f].mean(), df[df['y']==1][f].mean()
            median0, median1 = df[df['y']==0][f].median(), df[df['y']==1][f].median()
            try:
                auc = roc_auc_score(df['y'], df[f].fillna(0))
            except:
                auc = np.nan
            try:
                t_stat, p_val = ttest_ind(df[df['y']==0][f].dropna(), df[df['y']==1][f].dropna(), equal_var=False)
            except:
                t_stat, p_val = np.nan, np.nan
            results.append({
                'feature': f,
                'mean_y0': mean0,
                'mean_y1': mean1,
                'median_y0': median0,
                'median_y1': median1,
                'auc': auc,
                't_stat': t_stat,
                'p_val': p_val,
                'mean_diff': abs(mean1-mean0)
            })
        results_df = pd.DataFrame(results)
        strong_signals = results_df[((results_df['auc'] > 0.6) | (results_df['auc'] < 0.4)) & (results_df['p_val'] < 0.05)]
        print('分布差异显著且AUC较高的强信号特征：')
        print(strong_signals.sort_values('mean_diff', ascending=False)[['feature','auc','p_val','mean_y0','mean_y1','median_y0','median_y1']])
        # 可视化前几个强信号特征
        for f in strong_signals['feature'].head(5):
            plt.figure(figsize=(8,4))
            sns.kdeplot(df.loc[df['y']==1, f].dropna(), label='y=1', color='r')
            sns.kdeplot(df.loc[df['y']==0, f].dropna(), label='y=0', color='b')
            plt.title(f"{f} 分布对比")
            plt.legend()
            plt.show()
        # ========== 分析结束 ==========

        # 新增：定义并fit scaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # 按股票分组的时间序列分割
        train_idx, test_idx = [], []
        for code in np.unique(stock_code):
            idx = np.where(stock_code == code)[0]
            n = len(idx)
            if n < 100:
                continue
            split = int(n * 0.8)
            train_idx.extend(idx[:split])
            test_idx.extend(idx[split:])
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # 新增：定义test_stock_code
        test_stock_code = stock_code[test_idx]
        
        # 调试：检查特征与y的相关性，找出可能的泄露
        print("="*50)
        print("检查特征泄露...")
        high_corr_features = []
        low_corr_indices = []  # 保存低相关特征的索引
        
        # 使用更严格的阈值：0.4
        for i, col in enumerate(feature_cols):
            corr = np.corrcoef(X_train[:, i], y_train)[0, 1]
            if abs(corr) > 0.4:  # 更严格的阈值
                high_corr_features.append((col, corr))
                print(f"高相关特征: {col}, 相关系数: {corr:.4f}")
            else:
                low_corr_indices.append(i)  # 保存低相关特征的索引
        
        if high_corr_features:
            print(f"发现 {len(high_corr_features)} 个高相关特征，自动过滤...")
            
            # 详细分析高相关特征
            print("高相关特征详细分析:")
            for col, corr in high_corr_features:
                print(f"  特征 {col}: 相关系数 {corr:.4f}")
                # 检查这个特征是否与y完全一致
                if abs(corr) > 0.99:
                    print(f"    ⚠️  警告: 与y几乎完全相关，可能存在直接泄露")
                elif abs(corr) > 0.8:
                    print(f"    ⚠️  警告: 与y高度相关，可能存在间接泄露")
                elif abs(corr) > 0.6:
                    print(f"    ⚠️  警告: 与y中等相关，可能存在泄露")
                else:
                    print(f"    ℹ️  信息: 与y低相关，可能是有效特征")
            
            # 过滤掉高相关特征
            X_train = X_train[:, low_corr_indices]
            X_test = X_test[:, low_corr_indices]
            feature_cols = [feature_cols[i] for i in low_corr_indices]
            print(f"过滤后特征数量: {len(feature_cols)}")
            
            # 检查过滤后的特征组合是否会导致完美预测
            print("检查特征组合泄露...")
            if len(feature_cols) > 0:
                # 使用前10个特征训练一个简单模型，检查是否完美预测
                test_features = min(10, len(feature_cols))
                X_test_subset = X_train[:, :test_features]
                from sklearn.linear_model import LogisticRegression
                test_model = LogisticRegression(random_state=42, max_iter=1000)
                test_model.fit(X_test_subset, y_train)
                test_pred = test_model.predict(X_test_subset)
                test_accuracy = np.mean(test_pred == y_train)
                print(f"前{test_features}个特征的训练准确率: {test_accuracy:.4f}")
                if test_accuracy > 0.95:
                    print("⚠️  警告: 特征组合可能导致过拟合，建议进一步过滤")
            # 过滤后再fit scaler
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        else:
            print("未发现明显的高相关特征")
            # 未过滤时也要fit scaler
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        print("="*50)
    else:
        # 1. 加载和合并数据
        df = load_and_merge_stock_data()
        if df is None:
            return
        # 2. 增强特征工程
        df = enhanced_feature_engineering(df)
        # 3. 数据清理
        print("数据清理...")
        drop_cols = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])
        # 处理缺失值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['y', 'stock_code']:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
        df = df.dropna()
        print(f"清理后数据形状: {df.shape}")
        print(f"标签分布: {df['y'].value_counts().to_dict()}")
        # 4. 特征标准化
        scaler = StandardScaler()
        feature_cols = [c for c in df.columns if c not in ['y', 'stock_code']]
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        # 5. 按股票分组的时间序列分割
        print("按股票分组的时间序列分割...")
        train_data = []
        test_data = []
        for stock_code in df['stock_code'].unique():
            stock_df = df[df['stock_code'] == stock_code].copy()
            stock_df = stock_df.sort_values('trade_date') if 'trade_date' in stock_df.columns else stock_df
            n = len(stock_df)
            if n < 100:  # 数据太少，跳过
                continue
            split = int(n * 0.8)
            train_data.append(stock_df.iloc[:split])
            test_data.append(stock_df.iloc[split:])
        train_df = pd.concat(train_data, ignore_index=True)
        test_df = pd.concat(test_data, ignore_index=True)
        X_train = train_df.drop(columns=['y', 'stock_code']).values
        y_train = train_df['y'].values
        X_test = test_df.drop(columns=['y', 'stock_code']).values
        y_test = test_df['y'].values
        print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
        test_stock_code = test_df['stock_code'].values

    # 6. 特征选择
    k_features = min(50, X_train.shape[1])
    X_train_selected, X_test_selected, selected_indices = feature_selection(
        X_train, y_train, X_test, method='f_classif', k=k_features
    )

    # 7. 模型训练
    models, results = train_models(X_train_selected, y_train, X_test_selected, y_test)

    # 8. 结果对比
    print("\n" + "="*60)
    print("模型性能对比")
    print("="*60)
    
    for model_name, (proba, auc, f1) in results.items():
        print(f"{model_name:12} - AUC: {auc:.3f}, F1: {f1:.3f}")
    
    # 9. 可视化
    print("\n生成可视化图表...")
    
    plt.figure(figsize=(15, 10))
    
    # AUC对比
    plt.subplot(2, 3, 1)
    models_list = list(results.keys())
    aucs = [results[m][1] for m in models_list]
    plt.bar(models_list, aucs, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.title('模型AUC对比')
    plt.ylabel('AUC')
    plt.ylim(0.5, 1.0)
    for i, v in enumerate(aucs):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # F1对比
    plt.subplot(2, 3, 2)
    f1s = [results[m][2] for m in models_list]
    plt.bar(models_list, f1s, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.title('模型F1对比')
    plt.ylabel('F1')
    plt.ylim(0.3, 0.8)
    for i, v in enumerate(f1s):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # ROC曲线
    plt.subplot(2, 3, 3)
    from sklearn.metrics import roc_curve
    for model_name, (proba, auc, f1) in results.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC曲线对比')
    plt.legend()
    plt.grid(True)
    
    # 特征重要性（LightGBM）
    plt.subplot(2, 3, 4)
    lgbm = models['LightGBM']
    importances = lgbm.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    feature_names = [feature_cols[i] for i in selected_indices]
    plt.barh(range(min(20, len(indices))), importances[indices[:20]])
    plt.yticks(range(min(20, len(indices))), [feature_names[i] for i in indices[:20]])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 特征重要性')
    
    # 预测概率分布
    plt.subplot(2, 3, 5)
    best_model_name = max(results.keys(), key=lambda x: results[x][1])
    best_proba = results[best_model_name][0]
    plt.hist(best_proba, bins=30, alpha=0.7, color='skyblue')
    plt.xlabel('预测概率')
    plt.ylabel('样本数')
    plt.title('预测概率分布')
    
    # 混淆矩阵
    plt.subplot(2, 3, 6)
    from sklearn.metrics import confusion_matrix
    y_pred = (best_proba > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    
    plt.tight_layout()
    plt.savefig('multi_stock_model_results.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    # 10. 保存结果
    print("\n保存结果...")
    
    results_df = pd.DataFrame({
        'true_label': y_test,
        'stock_code': test_stock_code
    })
    
    for model_name, (proba, auc, f1) in results.items():
        results_df[f'{model_name}_proba'] = proba
    
    results_df.to_csv('688271_multi_stock_predictions.csv', index=False)
    
    # 保存最佳模型
    import joblib

    best_model = models[best_model_name]
    best_multi_stock_model_path = os.path.join(multimodal_root_path,'best_multi_stock_model.pkl')
    multi_stock_scaler_path = os.path.join(multimodal_root_path, 'multi_stock_scaler.pkl')
    multi_stock_features_path = os.path.join(multimodal_root_path, 'multi_stock_features.pkl')
    multi_stock_feature_cols_path = os.path.join(multimodal_root_path, 'multi_stock_feature_cols.pkl')

    joblib.dump(best_model, best_multi_stock_model_path)
    joblib.dump(scaler, multi_stock_scaler_path)
    joblib.dump(selected_indices, multi_stock_features_path)
    joblib.dump(feature_cols, multi_stock_feature_cols_path)
    
    print("结果已保存:")
    print("- multi_stock_predictions.csv: 预测结果")
    print("- best_multi_stock_model.pkl: 最佳模型")
    print("- multi_stock_scaler.pkl: 特征标准化器")
    print("- multi_stock_features.pkl: 选择的特征索引")
    print("- multi_stock_feature_cols.pkl: 特征列名")
    print("- multi_stock_model_results.png: 可视化图表")
    
    print("\n" + "="*60)
    print("多股票训练完成！")
    print("="*60)

if __name__ == '__main__':
    main() 