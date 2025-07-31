import pandas as pd
import numpy as np
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import sys
import os
import glob
import multi_stock_training

# 1. 遍历所有板块子文件夹
data_root = './dataset/2017-2023-news'
output_root = './dataset/process_text_add_tom'

embd = SentenceTransformerEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
window_size = 3  # 用过去3天新闻

for sector in os.listdir(data_root):
    sector_path = os.path.join(data_root, sector)
    if not os.path.isdir(sector_path):
        continue
    file_list = glob.glob(os.path.join(sector_path, '*.csv'))
    if not file_list:
        continue
    print(f"处理板块: {sector}, 文件数: {len(file_list)}")
    all_struct_features = []
    all_text_vecs = []
    all_y = []
    all_stock_code = []
    for csv_path in file_list:
        df = pd.read_csv(csv_path)
        if 'title' not in df.columns:
            continue
        stock_code = os.path.basename(csv_path).replace('.csv', '')
        # 先生成y（基于原始数据）
        df['next_close'] = df['close'].shift(-1)
        df['y'] = (df['next_close'] > df['close']).astype(int)
        df = df.dropna(subset=['next_close'])
        df = df.iloc[:-1]  # 去掉最后一行（没有next_close，标签无效）
        # if 'close' in df.columns and 'pre_close' in df.columns:
        #     df['y'] = (df['close'] > df['pre_close']).astype(int)
        # else:
        #     continue  # 跳过无标签的股票
        # 跳过标签单一的股票
        if df['y'].nunique() < 2:
            print(f"{csv_path} 标签单一，跳过")
            continue
        # 按时间排序（假设有trade_date列，否则按原顺序）
        if 'trade_date' in df.columns:
            df = df.sort_values('trade_date').reset_index(drop=True)
        # 文本embedding（用过去3天新闻的embedding均值或拼接作为T日特征）
        titles = df['title'].fillna('').astype(str).tolist()
        text_vecs = embd.embed_documents(titles)
        text_vecs = np.array(text_vecs)
        # 构造滑窗embedding（均值实现）
        text_vecs_window = []
        for i in range(len(text_vecs)):
            start = max(0, i - window_size)
            if i == 0:
                text_vecs_window.append(np.zeros_like(text_vecs[0]))
            else:
                window_embeds = text_vecs[start:i]
                if len(window_embeds) == 0:
                    text_vecs_window.append(np.zeros_like(text_vecs[0]))
                else:
                    text_vecs_window.append(np.mean(window_embeds, axis=0))
        text_vecs_window = np.array(text_vecs_window)
        # 结构化特征
        df_struct = df.copy()
        df_struct = multi_stock_training.enhanced_feature_engineering(df_struct)
        exclude_cols = ['title', 'y', 'stock_code', 'trade_date', 'next_close']
        struct_cols = [col for col in df_struct.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_struct[col])]
        struct_features = df_struct[struct_cols].fillna(0).values
        y = df['y'].values
        stock_codes = np.array([stock_code] * len(df_struct))
        min_len = min(len(struct_features), len(text_vecs_window), len(y))
        all_struct_features.append(struct_features[:min_len])
        all_text_vecs.append(text_vecs_window[:min_len])
        all_y.append(y[:min_len])
        all_stock_code.append(stock_codes[:min_len])
    if not all_struct_features:
        print(f"板块 {sector} 没有有效数据，跳过")
        continue
    # 合并所有股票
    all_struct_features = np.vstack(all_struct_features)
    all_text_vecs = np.vstack(all_text_vecs)
    all_y = np.concatenate(all_y)
    all_stock_code = np.concatenate(all_stock_code)
    multi_modal_features = np.concatenate([all_struct_features, all_text_vecs], axis=1)
    # 构建特征列名
    struct_feature_names = struct_cols
    text_feature_names = [f'text_emb_{i}' for i in range(all_text_vecs.shape[1])]
    all_feature_names = struct_feature_names + text_feature_names
    # 保存为csv，包含y和stock_code，便于后续训练
    output_dir = os.path.join(output_root, sector)
    os.makedirs(output_dir, exist_ok=True)
    multi_modal_df = pd.DataFrame(multi_modal_features, columns=all_feature_names)
    multi_modal_df['y'] = all_y
    multi_modal_df['stock_code'] = all_stock_code
    output_path = os.path.join(output_dir, 'multimodal.csv')
    multi_modal_df.to_csv(output_path, index=False)
    print(f'板块 {sector} 多模态特征 shape: {multi_modal_features.shape}, 标签 shape: {all_y.shape}, 已保存到: {output_path}') 