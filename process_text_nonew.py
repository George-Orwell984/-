import pandas as pd
import numpy as np
import sys
import os
import glob
import multi_stock_training

# 1. 遍历所有板块子文件夹
data_root = './dataset/2007-2025-no_news'
output_root = './dataset/process_notext_tom'

for sector in os.listdir(data_root):
    sector_path = os.path.join(data_root, sector)
    if not os.path.isdir(sector_path):
        continue
    file_list = glob.glob(os.path.join(sector_path, '*.csv'))
    if not file_list:
        continue
    print(f"处理板块: {sector}, 文件数: {len(file_list)}")
    all_struct_features = []
    all_y = []
    all_stock_code = []
    for csv_path in file_list:
        df = pd.read_csv(csv_path)
        stock_code = os.path.basename(csv_path).replace('.csv', '')
        # 先生成y（基于原始数据）
        df['next_close'] = df['close'].shift(-1)
        df['y'] = (df['next_close'] > df['close']).astype(int)
        df = df.iloc[:-1]  # 去掉最后一行（没有next_close，标签无效）
        # 跳过标签单一的股票
        if df['y'].nunique() < 2:
            print(f"{csv_path} 标签单一，跳过")
            continue
        # 按时间排序（假设有trade_date列，否则按原顺序）
        if 'trade_date' in df.columns:
            df = df.sort_values('trade_date').reset_index(drop=True)
        # 结构化特征
        df_struct = df.copy()
        df_struct = multi_stock_training.enhanced_feature_engineering(df_struct)
        exclude_cols = ['title', 'y', 'stock_code', 'trade_date']
        struct_cols = [col for col in df_struct.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_struct[col])]
        struct_features = df_struct[struct_cols].fillna(0).values
        y = df['y'].values
        stock_codes = np.array([stock_code] * len(df_struct))
        min_len = min(len(struct_features), len(y))
        all_struct_features.append(struct_features[:min_len])
        all_y.append(y[:min_len])
        all_stock_code.append(stock_codes[:min_len])
    if not all_struct_features:
        print(f"板块 {sector} 没有有效数据，跳过")
        continue
    # 合并所有股票
    all_struct_features = np.vstack(all_struct_features)
    all_y = np.concatenate(all_y)
    all_stock_code = np.concatenate(all_stock_code)
    # 保存为csv，包含y和stock_code，便于后续训练
    output_dir = os.path.join(output_root, sector)
    os.makedirs(output_dir, exist_ok=True)
    multi_modal_df = pd.DataFrame(all_struct_features, columns = struct_cols)
    multi_modal_df['y'] = all_y
    multi_modal_df['stock_code'] = all_stock_code
    output_path = os.path.join(output_dir, 'multimodal.csv')
    multi_modal_df.to_csv(output_path, index=False)
    print(f'板块 {sector} 多模态特征 shape: {all_struct_features.shape}, 标签 shape: {all_y.shape}, 已保存到: {output_path}') 