# import pandas as pd
# import os
# import glob

# data_root = './dataset/2017-2023-news'
# output_root = './dataset'

# results = []

# for sector in os.listdir(data_root):
#     sector_path = os.path.join(data_root, sector)
#     if not os.path.isdir(sector_path):
#         continue
#     file_list = glob.glob(os.path.join(sector_path, '*.csv'))
#     if not file_list:
#         continue
#     print(f"处理板块: {sector}, 文件数: {len(file_list)}")
#     for csv_path in file_list:
#         df = pd.read_csv(csv_path)
#         stock_name = os.path.basename(csv_path)
#         # 用明天数据
#         if 'close' in df.columns:
#             df['next_close'] = df['close'].shift(-1)
#             df['y_next'] = (df['next_close'] > df['close']).astype(int)
#             df_next = df.iloc[:-1]  # 去掉最后一行
#             y_next_counts = df_next['y_next'].value_counts()
#             y_next_ratio = df_next['y_next'].value_counts(normalize=True)
#             results.append({
#                 'sector': sector,
#                 'stock': stock_name,
#                 'label_type': 'next',
#                 'y_0_count': y_next_counts.get(0, 0),
#                 'y_1_count': y_next_counts.get(1, 0),
#                 'y_0_ratio': y_next_ratio.get(0, 0),
#                 'y_1_ratio': y_next_ratio.get(1, 0)
#             })
#         # 用今天数据
#         if 'close' in df.columns and 'pre_close' in df.columns:
#             df['y_today'] = (df['close'] > df['pre_close']).astype(int)
#             y_today_counts = df['y_today'].value_counts()
#             y_today_ratio = df['y_today'].value_counts(normalize=True)
#             results.append({
#                 'sector': sector,
#                 'stock': stock_name,
#                 'label_type': 'today',
#                 'y_0_count': y_today_counts.get(0, 0),
#                 'y_1_count': y_today_counts.get(1, 0),
#                 'y_0_ratio': y_today_ratio.get(0, 0),
#                 'y_1_ratio': y_today_ratio.get(1, 0)
#             })

# # 保存结果
# results_df = pd.DataFrame(results)
# results_df.to_csv('label_distribution_summary.csv', index=False)
# print("标签分布统计已保存到 label_distribution_summary.csv")

import os
import glob
import pandas as pd

root_dir = './dataset/test'

for sector in os.listdir(root_dir):
    sector_path = os.path.join(root_dir, sector)
    if not os.path.isdir(sector_path):
        continue
    file_list = glob.glob(os.path.join(sector_path, '*.csv'))
    for csv_path in file_list:
        try:
            df = pd.read_csv(csv_path)
            # 只在有'trade_date'列时去重
            if 'trade_date' in df.columns:
                before = len(df)
                df = df.drop_duplicates(subset=['trade_date'], keep='first')
                after = len(df)
                if after < before:
                    print(f"{csv_path}: 去重 {before-after} 行")
                df.to_csv(csv_path, index=False)
            else:
                print(f"{csv_path}: 无'trade_date'列，跳过")
        except Exception as e:
            print(f"{csv_path}: 处理出错 - {e}")