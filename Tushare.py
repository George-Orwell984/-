from heapq import merge
import tushare as ts
import pandas as pd
import rqdatac
import time
import os
# 设置token
ts.set_token('7e3e86a9f7239691b1b9f8f28faa885806de799400fc2d58f054f43b')
pro = ts.pro_api()

# df = pro.news(src='eastmoney', start_date='2025-05-30 09:30:00', end_date='2025-05-30 15:00:00')

# df1 = pro.news(src='10jqka',start_date='2025-05-30 09:30:00', end_date='2025-05-30 15:00:00')

# df2 = pro.major_news(src='', start_date='2025-05-30 09:30:00', end_date='2025-05-30 15:00:00')

# df3 = pro.cctv_news(date='20250530')

# df4 = pro.anns_d(ann_date='20250530')

# # 假设df, df1, df2, df3, df4已获取


# # 1. 自动识别并重命名时间列为'time'
# def auto_rename_time_column(df):
#     for col in ['datetime', 'datetime', 'pub_time', 'date', 'rec_time', 'ann_date']:
#         if col in df.columns:
#             df = df.rename(columns={col: 'time'})
#             break
#     return df

# dfs = [df, df1, df2, df3, df4]

# dfs = [auto_rename_time_column(d) for d in dfs]

# # 2. 统一'time'格式
# def to_yyyymmdd(s):
#     if isinstance(s, str) and '-' in s:
#         return s[:10].replace('-', '')
#     elif isinstance(s, str) and len(s) == 8:
#         return s
#     elif pd.notnull(s):
#         return pd.to_datetime(s).strftime('%Y%m%d')
#     else:
#         return None

# for d in dfs:
#     if 'time' in d.columns:
#         d['time'] = d['time'].astype(str).apply(to_yyyymmdd)
#     else:
#         print(f"DataFrame {d} 没有 'time' 列，跳过")


# # 3. 合并
# # 给每个DataFrame加唯一前缀
# prefixes = ['em_', 'jq_', 'major_', 'cctv_', 'anns_']
# for i, d in enumerate(dfs):
#     prefix = prefixes[i]
#     cols = d.columns.tolist()
#     new_cols = [c if c == 'time' else f'{prefix}{c}' for c in cols]
#     d.columns = new_cols

# # 3. 合并
# from functools import reduce
# merged = reduce(lambda left, right: pd.merge(left, right, on='time', how='outer'), dfs)

# # 4. 可选：按time排序
# merged = merged.sort_values('time').reset_index(drop=True)

# # 5. 保存或查看
# merged.to_csv('merged_news.csv', index=False)
# print(merged.head())

rqdatac.init(username="license",
                password="JodaGyQK1heu_XyGDY7TGNf6IOTrOGh6tspwdOqkLy7RzPmuFWbdvkGLN62OOo8CnskgI2r7BU0plEceGCJewrUMRIkhtqwfT3IrUFMzbkNBoyYJm4pZ9bcnv7cOuKGmzAoiQt4Y9ZTr_H_9O5UrovvUiNFd802rdLKXhdBUEKo=gmQOO-AtwUJsrCkG9iYnEpTTbdrsteYQwkzWOoaKM8jh6sjHxQjqrKledyMLmSl-sHGRgynnGK9I-rzPqwocbAJqWgLIDYrZJYH4SKDghTNNvHJVjlrrIVDy8Zv-AfhCoaKxkLfGk5BViJ-cOy1BqDAm4mZvRNlS6WPPaBw3h4w=")

stock_codes = [
    '688271.SH'
]


output_dir = './dataset/test'

def convert_code(ts_code):
    if ts_code.endswith('.SH'):
        return ts_code.replace('.SH', '.XSHG')
    elif ts_code.endswith('.SZ'):
        return ts_code.replace('.SZ', '.XSHE')
    else:
        return ts_code


def get_stock_data(ts_code, start_date='20230530', end_date='20230630'):
    """获取单个股票的完整数据"""
    print(f"\n正在获取 {ts_code} 的数据...")
    
    try:
        # 获取日线行情数据
        print(f"正在获取 {ts_code} 日线行情数据...")
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        
        # 获取每日指标数据
        print(f"正在获取 {ts_code} 每日指标数据...")
        df2 = pro.daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_date, 
                              fields='ts_code,trade_date,turnover_rate,volume_ratio,pe,pb,ps,total_share,float_share,free_share,total_mv,circ_mv')
        
        # 获取新闻舆情数据
        print(f"正在获取 {ts_code} 新闻舆情数据...")
        rq_code = convert_code(ts_code)

        df['trade_date'] = df['trade_date'].astype(str)
        df2['trade_date'] = df2['trade_date'].astype(str)
        df['ts_code'] = df['ts_code'].astype(str)
        df2['ts_code'] = df2['ts_code'].astype(str)
        
      
        df3 = rqdatac.news.get_stock_news(order_book_ids=rq_code, start_date = start_date, end_date = end_date)
        df3 = df3.reset_index().rename(columns={'order_book_id': 'ts_code', 'datetime': 'trade_date'})

        ts_code_list = df3['ts_code'].tolist()
        ts_code_tushare_list = rqdatac.id_convert(ts_code_list, to='normal')
        df3['ts_code'] = ts_code_tushare_list
        df3['ts_code'] = df3['ts_code'].astype(str)

        df3['trade_date'] = df3['trade_date'].dt.strftime('%Y%m%d')
        df3['trade_date'] = df3['trade_date'].astype(str)
        # print(df3)
        # exit()
        # 如果两个数据框都不为空，则进行合并
        if not df.empty and not df2.empty and not df3.empty:
            # 使用merge方法按ts_code和trade_date合并数据
            temp_merged_df = pd.merge(df, df2, on=['ts_code', 'trade_date'], how='outer')
            merged_df = pd.merge(temp_merged_df, df3, on=['ts_code', 'trade_date'], how='outer')
            # 只保留无缺失值的行
            merged_df = merged_df.dropna()
            print(f"{ts_code} 合并后的数据形状: {merged_df.shape}")
            
            # 保存合并后的数据


            # 创建文件夹（如果不存在）
            os.makedirs(output_dir, exist_ok=True)

            # 保存文件到新建的文件夹
            filename = os.path.join(output_dir, f'{ts_code}_test.csv')
            merged_df.to_csv(filename, index=False)
            print(f"{ts_code} 数据已保存到 {filename}")
            
            # 显示数据统计
            print(f"{ts_code} 数据统计:")
            print(f"  行数: {len(merged_df)}")
            print(f"  列数: {len(merged_df.columns)}")
            print(f"  日期范围: {merged_df['trade_date'].min()} 到 {merged_df['trade_date'].max()}")
            
            return merged_df
            
        elif not df.empty:
            # 只有日线数据
            filename = f'{ts_code}.csv'
            df.to_csv(filename, index=False)
            print(f"{ts_code} 只有日线数据，已保存到 {filename}")
            return df
            
        elif not df2.empty:
            # 只有指标数据
            filename = f'{ts_code}.csv'
            df2.to_csv(filename, index=False)
            print(f"{ts_code} 只有指标数据，已保存到 {filename}")
            return df2
            
        else:
            print(f"错误：{ts_code} 没有获取到任何数据")
            return None
            
    except Exception as e:
        print(f"获取 {ts_code} 数据时发生错误: {e}")
        return None

def main():
    """主函数：批量获取所有股票数据"""
    print("="*60)
    print("开始批量获取股票数据")
    print("="*60)
    print(f"要获取的股票: {stock_codes}")
    
    successful_stocks = []
    failed_stocks = []
    
    for i, ts_code in enumerate(stock_codes, 1):
        print(f"\n进度: {i}/{len(stock_codes)}")
        
        # 获取股票数据
        result = get_stock_data(ts_code)
        
        if result is not None:
            successful_stocks.append(ts_code)
        else:
            failed_stocks.append(ts_code)
        
        # 添加延时避免API限制
        if i < len(stock_codes):  # 不是最后一个
            print("等待2秒...")
            time.sleep(2)
    
    # 总结
    print("\n" + "="*60)
    print("数据获取完成总结")
    print("="*60)
    print(f"成功获取的股票: {successful_stocks}")
    print(f"获取失败的股票: {failed_stocks}")
    print(f"成功率: {len(successful_stocks)}/{len(stock_codes)} = {len(successful_stocks)/len(stock_codes)*100:.1f}%")
    
    if successful_stocks:
        print(f"\n成功获取的股票数据文件:")
        for stock in successful_stocks:
            print(f"  - {stock}.csv")
    
    if failed_stocks:
        print(f"\n获取失败的股票:")
        for stock in failed_stocks:
            print(f"  - {stock}")


if __name__ == '__main__':
    main()