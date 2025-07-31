#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 单只股票推理预测代码
# 专门针对688271SH进行下一天预测
# 基于历史数据生成特征，然后输入训练好的模型进行预测

import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime, timedelta

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def load_stock_data(stock_file):
    # 加载股票数据    
    print(f"加载股票数据: {stock_file}")
    if not os.path.exists(stock_file):
        raise FileNotFoundError(f"股票文件不存在: {stock_file}")
    
    df = pd.read_csv(stock_file)
    print(f"股票数据形状: {df.shape}")
    
    # 检查必要的列
    required_cols = ['close', 'pre_close', 'vol', 'amount']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"股票数据缺少必要列: {missing_cols}")
    
    # 按时间排序
    if 'trade_date' in df.columns:
        df = df.sort_values('trade_date').reset_index(drop=True)
    
    # 生成标签
    df['y'] = (df['close'] > df['pre_close']).astype(int)
    
    print(f"标签分布: {df['y'].value_counts().to_dict()}")    
    return df

def calculate_technical_indicators(df):
    print("计算技术指标...")    
    # RSI
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss +1e-6)
        rsi =100 * (100 / (1 + rs))
        return rsi
    
    # MACD
    def calculate_macd(prices, fastperiod=12, slowperiod=26, signalperiod=9):
        ema_fast = prices.ewm(span=fastperiod).mean()
        ema_slow = prices.ewm(span=slowperiod).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signalperiod).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    # 布林带
    def calculate_bollinger_bands(prices, period=20, nbdev=2):
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * nbdev)
        lower = middle - (std * nbdev)
        return upper, middle, lower
    
    # 计算技术指标
    if 'close' in df.columns:
        df['rsi_14'] = calculate_rsi(df['close'], period=14)
        df['rsi_21'] = calculate_rsi(df['close'], period=21)
        
        macd, macd_signal, macd_hist = calculate_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        df['macd_ratio'] = macd / (df['close'] + 1e-6)
        
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['close'])
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / (bb_middle + 1e-6)
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-6)
    return df

def generate_lag_features(df):
    print("生成滞后特征...")
    
    # 基础滞后特征
    lag_features = ['pct_chg', 'vol', 'amount']
    for lag in [1, 2, 10]:
        for col in lag_features:
            if col in df.columns:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    # 滚动统计特征
    rolling_features = ['close', 'vol', 'pct_chg', 'amount']
    for window in [3, 10, 15]:
        for col in rolling_features:
            if col in df.columns:
                df[f'{col}_roll{window}_mean'] = df[col].shift(1).rolling(window, min_periods=1).mean()
                df[f'{col}_roll{window}_std'] = df[col].shift(1).rolling(window, min_periods=1).std()
                df[f'{col}_roll{window}_max'] = df[col].shift(1).rolling(window, min_periods=1).max()
                df[f'{col}_roll{window}_min'] = df[col].shift(1).rolling(window, min_periods=1).min()
    
    return df

def generate_next_day_features_from_history(df, feature_columns):
    # 基于历史数据生成下一天的特征
    print("基于历史数据生成下一天特征...")
    
    # 获取最后一天的数据作为基准
    last_day_data = df.iloc[-1:].copy()
    
    # 获取历史数据（最近30天）来构造下一天的特征
    history_data = df.tail(30).copy()
    
    if len(history_data) < 20:
        print("历史数据不足，无法生成下一天特征")
        return None
    
    # 基于历史数据构造下一天的特征
    next_day_features_dict = {}
    
    #1. 价格相关特征（基于历史收盘价）
    if 'close' in history_data.columns:
        recent_prices = history_data['close'].tail(20).values  # 最近20天
        next_day_features_dict.update({
          'price_mean': np.mean(recent_prices),
         'price_std': np.std(recent_prices),
            'price_trend': (recent_prices[-1] - recent_prices[0]) / (recent_prices[0] + 1e-6),
            'price_momentum': recent_prices[-1] / (recent_prices[-5] + 1e-6) if len(recent_prices) >= 5 else 1
        })
    
    #2. 成交量相关特征
    if 'vol' in history_data.columns:
        recent_volumes = history_data['vol'].tail(20).values
        next_day_features_dict.update({
           'volume_mean': np.mean(recent_volumes),
          'volume_std': np.std(recent_volumes),
           'volume_trend': (recent_volumes[-1] - recent_volumes[0]) / (recent_volumes[0] + 0.000001)
        })
    
    #3. 涨跌幅相关特征
    if 'pct_chg' in history_data.columns:
        recent_changes = history_data['pct_chg'].tail(20).values
        next_day_features_dict.update({
           'change_mean': np.mean(recent_changes),
          'change_std': np.std(recent_changes),
          'positive_days': np.sum(recent_changes > 0),
          'negative_days': np.sum(recent_changes < 0)
        })
    
    # 4 技术指标特征（基于历史数据计算）
    if 'close' in history_data.columns:
        close_prices = history_data['close'].values
        # RSI
        if len(close_prices) >= 14:
            delta = np.diff(close_prices)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = np.mean(gain[-14:])
            avg_loss = np.mean(loss[-14:])
            rsi = 100 - (10 / (1 + avg_gain / (avg_loss + 1e-6)))
            next_day_features_dict['rsi'] = rsi
        
        # 移动平均
        if len(close_prices) >= 20:
            next_day_features_dict['ma_20'] = np.mean(close_prices[-20:])
            next_day_features_dict['ma_5'] = np.mean(close_prices[-5:])
            next_day_features_dict['ma_ratio'] = next_day_features_dict['ma_5'] / (next_day_features_dict['ma_20'] + 1e-6)
    
    # 5. 时间特征
    if 'trade_date' in last_day_data.columns:
        last_date = str(last_day_data['trade_date'].iloc[0])
        try:
            # 假设trade_date是YYYYMMDD格式
            date_obj = datetime.strptime(last_date, '%Y%m%d')
            next_date = date_obj + timedelta(days=1)
            next_day_features_dict['weekday'] = next_date.weekday()
            next_day_date = next_date.strftime('%Y%m%d')
        except:
            next_day_features_dict['weekday'] = 0
            next_day_date = 'unknown'
    else:
        next_day_date = 'unknown'    
    
    #6 基于历史数据的技术指标特征
    # 使用历史数据的最后一天的技术指标作为下一天的预测特征
    tech_features = ['rsi_14', 'rsi_21', 'macd', 'macd_signal', 'macd_hist', 'bb_position']
    for feature in tech_features:
        if feature in history_data.columns:
            next_day_features_dict[feature] = history_data[feature].iloc[-1]
    
    #7. 滞后特征（使用历史数据的最后一天）
    lag_features = [col for col in history_data.columns if '_lag' in col or '_roll' in col]
    for feature in lag_features:
        if feature in history_data.columns:
            next_day_features_dict[feature] = history_data[feature].iloc[-1]
    
    # 构造特征向量
    feature_vector = []
    for col in feature_columns:
        if col in next_day_features_dict:
            feature_vector.append(next_day_features_dict[col])
        else:
            feature_vector.append(0) # 使用0作为缺失值的默认填充
    
    return {
    'features': np.array(feature_vector).reshape(1, -1),
      'date': next_day_date
    }

def main():
    print("=" * 60)
    print("单只股票推理预测 - 688271SH")
    print("=" * 60)
    
    # 配置路径
    stock_file = r'F:\djy\Attention-CLX-stock-prediction\dataset\test\688271.SH.csv'
    config_path = './dataset/test/互联网'
    
    model_path = os.path.join(config_path, 'best_multi_stock_model.pkl')
    scaler_path = os.path.join(config_path, 'multi_stock_scaler.pkl')
    feature_columns_path = os.path.join(config_path, 'multi_stock_feature_cols.pkl')
    selected_indices_path = os.path.join(config_path, 'multi_stock_features.pkl')   
    
    # 检查文件是否存在
    print("检查必要文件...")
    required_files = [model_path, scaler_path, feature_columns_path, selected_indices_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            return
        else:
            print(f"✅ 文件存在: {file_path}")
    
    # 加载股票数据
    df = load_stock_data(stock_file)
    
    # 计算技术指标
    df = calculate_technical_indicators(df)
    
    # 生成滞后特征
    df = generate_lag_features(df)
    
    # 加载模型和工具
    print("\n加载模型和工具...")    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_columns = joblib.load(feature_columns_path)
    selected_indices = joblib.load(selected_indices_path)
    
    print(f"模型类型: {type(model)}")
    print(f"特征列数量: {len(feature_columns)}")
    print(f"选择特征数量: {len(selected_indices)}")
    
    # 准备测试集数据（后20%）
    total_len = len(df)
    test_start_idx = int(total_len * 0.8)
    test_df = df.iloc[test_start_idx:].copy()
    
    print(f"\n测试集数据: {len(test_df)} 行")
    
    # 准备测试特征
    print("准备测试特征...")
    missing_features = [col for col in feature_columns if col not in test_df.columns]
    # for col in missing_features:
    #     test_df[col] = 0
    if missing_features:
        zeros_df = pd.DataFrame(0, index=test_df.index, columns=missing_features)
        test_df = pd.concat([test_df, zeros_df], axis=1)
        test_df = test_df.copy()  # 去碎片化
    
    # 确保特征顺序一致
    X_test = test_df[feature_columns].fillna(0).values
    y_test = test_df['y'].values
    
    # 标准化
    X_test = scaler.transform(X_test)
    
    # 特征选择
    X_test_selected = X_test[:, selected_indices]
    
    # 进行测试集预测
    print("\n" + "=" * 60)
    print("测试集预测结果:")
    print("=" * 60)
    
    y_pred = model.predict(X_test_selected)
    y_pred_proba = model.predict_proba(X_test_selected)[:, 1] if hasattr(model, 'predict_proba') else None
    
    from sklearn.metrics import accuracy_score, classification_report
    accuracy = accuracy_score(y_test, y_pred)
    print(f"测试集准确率: {accuracy:.4f}")
    
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    if y_pred_proba is not None:
        print(f"\n预测概率统计:")
        print(f"平均预测概率: {np.mean(y_pred_proba):.4f}")
        print(f"预测概率标准差: {np.std(y_pred_proba):.4f}")
    
    # 生成下一天特征并预测
    print("\n" + "=" * 60)
    print("生成下一天特征并预测...")
    print("=" * 60)
    
    next_day_features = generate_next_day_features_from_history(df, feature_columns)
    
    if next_day_features:
        print("✅ 成功生成下一天特征")
        
        # 获取下一天特征
        next_day_X = next_day_features['features']
        next_day_date = next_day_features['date']
        
        # 标准化下一天特征
        next_day_X = scaler.transform(next_day_X)
        
        # 特征选择
        next_day_X_selected = next_day_X[:, selected_indices]
        
        # 进行下一天预测
        y_pred_next = model.predict(next_day_X_selected)
        y_pred_proba_next = model.predict_proba(next_day_X_selected)[:, 1] if hasattr(model, 'predict_proba') else None
        
        print(f"\n下一天预测结果:")
        print("=" * 60)
        print(f"股票代码: 688271SH")
        print(f"预测日期: {next_day_date}")
        print(f"预测结果: {'上涨' if y_pred_next[0] == 1 else '下跌'}")
        if y_pred_proba_next is not None:
            print(f"预测概率: {y_pred_proba_next[0]:.4f}")
        print("=" * 60)
        
        # 输出预测详情
        print(f"\n预测详情:")
        print(f"- 股票: 688271SH")
        print(f"- 日期: {next_day_date}")
        print(f"- 预测: {y_pred_next[0]} ({'上涨' if y_pred_next[0] == 1 else '下跌'})")
        if y_pred_proba_next is not None:
            print(f"- 概率: {y_pred_proba_next[0]:.4f}")
            if y_pred_proba_next[0] > 0.7:
                print(f"- 置信度: 高")
            elif y_pred_proba_next[0] > 0.6:
                print(f"- 置信度: 中等")
            else:
                print(f"- 置信度: 低")
    else:
        print("❌ 未能生成下一天特征")
    
    print("\n" + "=" * 60)
    print("预测完成")
    print(f"测试集准确率: {accuracy:.4f}")
    print("=" * 60)

if __name__ == '__main__':
    main() 