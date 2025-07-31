import pandas as pd
import numpy as np
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import sys
import os
import glob
import multi_stock_training
from sentence_transformers import SentenceTransformer
# 1. 批量读取所有银行+新闻csv文件
folder = './dataset/test'

# bankuai = folder.split('/')[-1]
# save_path = os.path.join('F:\djy\Attention-CLX-stock-prediction\dataset\test',bankuai)
# os.makedirs(save_path,exist_ok=True)
file_list = glob.glob(os.path.join(folder, '688271.SH_test.csv'))

all_struct_features = []
all_text_vecs = []
all_y = []
all_stock_code = []


# model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# embd = SentenceTransformerEmbeddings(model_name="/your/local/model/path")
embd = SentenceTransformerEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
window_size = 3  # 用过去3天新闻

for csv_path in file_list:
    df = pd.read_csv(csv_path)
    if 'title' not in df.columns:
        continue
    stock_code = os.path.basename(csv_path).replace('.csv', '')
    # 先生成y（基于原始数据）
    df['next_close'] = df['close'].shift(-1)
    df['y'] = (df['next_close'] > df['close']).astype(int)
    df = df.iloc[:-1]  # 去掉最后一行（没有next_close，标签无效）
    df = df.dropna(subset=['next_close'])
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
    # 构造滑窗embedding（均值和拼接二选一，以下为均值实现）
    text_vecs_window = []
    for i in range(len(text_vecs)):
        start = max(0, i - window_size)
        if i == 0:
            # T日不能用T日新闻，只能用T-1及以前
            text_vecs_window.append(np.zeros_like(text_vecs[0]))
        else:
            window_embeds = text_vecs[start:i]  # 不包含i本身
            if len(window_embeds) == 0:
                text_vecs_window.append(np.zeros_like(text_vecs[0]))
            else:
                text_vecs_window.append(np.mean(window_embeds, axis=0))
    text_vecs_window = np.array(text_vecs_window)
    # 结构化特征
    df_struct = df.copy()
    df_struct = multi_stock_training.enhanced_feature_engineering(df_struct)
    exclude_cols = ['title', 'y', 'stock_code', 'trade_date']
    struct_cols = [col for col in df_struct.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_struct[col])]
    struct_features = df_struct[struct_cols].fillna(0).values
    # y严格用原始df的y
    y = df['y'].values
    # 股票代码
    stock_codes = np.array([stock_code] * len(df_struct))
    # 对齐长度
    min_len = min(len(struct_features), len(text_vecs_window), len(y))
    all_struct_features.append(struct_features[:min_len])
    all_text_vecs.append(text_vecs_window[:min_len])
    all_y.append(y[:min_len])
    all_stock_code.append(stock_codes[:min_len])

# 合并所有股票
all_struct_features = np.vstack(all_struct_features)
all_text_vecs = np.vstack(all_text_vecs)
all_y = np.concatenate(all_y)
all_stock_code = np.concatenate(all_stock_code)

multi_modal_features = np.hstack([all_struct_features, all_text_vecs])

# 保存为csv，包含y和stock_code，便于后续训练
multi_modal_df = pd.DataFrame(multi_modal_features)
multi_modal_df['y'] = all_y
multi_modal_df['stock_code'] = all_stock_code
temp = './dataset/test'
save_path_future = os.path.join(temp, '688271.SH_multimodal.csv')
multi_modal_df.to_csv(save_path_future, index=False)
print(f'多模态特征 shape: {multi_modal_features.shape}, 标签 shape: {all_y.shape}') 


# # 生成“测试集最后一天的下一天”特征
# if len(file_list) == 1:
#     df = pd.read_csv(file_list[0])
#     if 'trade_date' in df.columns:
#         df = df.sort_values('trade_date').reset_index(drop=True)
#         last_row = df.iloc[-1]
#         # 生成下一天日期
#         try:
#             last_date = str(last_row['trade_date'])
#             date_obj = pd.to_datetime(last_date, format='%Y%m%d')
#             next_date = (date_obj + pd.Timedelta(days=1)).strftime('%Y%m%d')
#         except Exception:
#             next_date = 'future'
#     else:
#         next_date = 'future'
#     stock_code = os.path.basename(file_list[0]).replace('.csv', '')

#     # 结构化特征：用最后一天的结构化特征（或均值/滑窗均值）
#     df_struct = df.copy()
#     df_struct = multi_stock_training.enhanced_feature_engineering(df_struct)
#     exclude_cols = ['title', 'y', 'stock_code', 'trade_date']
#     struct_cols = [col for col in df_struct.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_struct[col])]
#     # 用最后一天的结构化特征
#     struct_feature_future = df_struct[struct_cols].iloc[[-1]].fillna(0).values

#     # 文本embedding：用最后window_size天新闻embedding均值
#     titles = df['title'].fillna('').astype(str).tolist()
#     text_vecs = embd.embed_documents(titles)
#     text_vecs = np.array(text_vecs)
#     if len(text_vecs) >= window_size:
#         text_vec_future = np.mean(text_vecs[-window_size:], axis=0, keepdims=True)
#     elif len(text_vecs) > 0:
#         text_vec_future = np.mean(text_vecs, axis=0, keepdims=True)
#     else:
#         text_vec_future = np.zeros((1, embd.embed_documents(['test']).shape[1]))

#     # 拼接多模态特征
#     multi_modal_feature_future = np.hstack([struct_feature_future, text_vec_future])
#     # 构造DataFrame
#     multi_modal_df_future = pd.DataFrame(multi_modal_feature_future)
#     multi_modal_df_future['stock_code'] = stock_code
#     multi_modal_df_future['trade_date'] = next_date
#     # 保存
#     temp = './dataset/process_text/互联网'
#     save_path_future = os.path.join(temp, 'multimodal.csv')
#     multi_modal_df_future.to_csv(save_path_future, index=False)
#     print(f'已生成未来一天({next_date})的多模态特征，shape: {multi_modal_feature_future.shape}')