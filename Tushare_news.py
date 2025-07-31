from heapq import merge
import tushare as ts
import pandas as pd
import rqdatac
import time
import os
# 设置token
ts.set_token('2876ea85cb005fb5fa17c809a98174f2d5aae8b1f830110a5ead6211')
pro = ts.pro_api()

rqdatac.init(username="license",
                password="JodaGyQK1heu_XyGDY7TGNf6IOTrOGh6tspwdOqkLy7RzPmuFWbdvkGLN62OOo8CnskgI2r7BU0plEceGCJewrUMRIkhtqwfT3IrUFMzbkNBoyYJm4pZ9bcnv7cOuKGmzAoiQt4Y9ZTr_H_9O5UrovvUiNFd802rdLKXhdBUEKo=gmQOO-AtwUJsrCkG9iYnEpTTbdrsteYQwkzWOoaKM8jh6sjHxQjqrKledyMLmSl-sHGRgynnGK9I-rzPqwocbAJqWgLIDYrZJYH4SKDghTNNvHJVjlrrIVDy8Zv-AfhCoaKxkLfGk5BViJ-cOy1BqDAm4mZvRNlS6WPPaBw3h4w=")

root_dir = './dataset/2007-2025-no_news'
news_dir = './dataset/2017-2023-news/test'

def convert_code(ts_code):
    if ts_code.endswith('.SH'):
        return ts_code.replace('.SH', '.XSHG')
    elif ts_code.endswith('.SZ'):
        return ts_code.replace('.SZ', '.XSHE')
    else:
        return ts_code


def get_stock_data(ts_code, read_dir, output_dir, start_date='20230530', end_date='20230630'):
    """只获取新闻舆情数据，并与本地csv拼接"""
    print(f"\n正在处理 {ts_code} ...")
    try:
        # 读取本地csv
        local_file = os.path.join(read_dir, f'{ts_code}.csv')
        if not os.path.exists(local_file):
            print(f"本地文件不存在: {local_file}")
            return None
        df_local = pd.read_csv(local_file)
        if 'trade_date' not in df_local.columns:
            print(f"本地文件缺少 trade_date 列: {local_file}")
            return None
        # 获取新闻舆情数据
        rq_code = convert_code(ts_code)
        print(f"正在获取 {ts_code} 新闻舆情数据...")
        df3 = rqdatac.news.get_stock_news(order_book_ids=rq_code, start_date=start_date, end_date=end_date)
        if df3 is None or df3.empty:
            print(f"{ts_code} 没有新闻舆情数据，仅保存本地数据")
            merged_df = df_local
        else:
            df3 = df3.reset_index().rename(columns={'order_book_id': 'ts_code', 'datetime': 'trade_date'})
            ts_code_list = df3['ts_code'].tolist()
            ts_code_tushare_list = rqdatac.id_convert(ts_code_list, to='normal')
            df3['ts_code'] = ts_code_tushare_list
            df3['ts_code'] = df3['ts_code'].astype(str)
            df3['trade_date'] = df3['trade_date'].dt.strftime('%Y%m%d')
            df3['trade_date'] = df3['trade_date'].astype(int)
            # 合并本地数据和新闻舆情
            merged_df = pd.merge(df_local, df3, on=['ts_code', 'trade_date'], how='left')
            merged_df = merged_df.dropna()
        # 保存合并后的数据
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f'{ts_code}.csv')
        merged_df.to_csv(filename, index=False)
        print(f"{ts_code} 数据已保存到 {filename}")
        print(f"  行数: {len(merged_df)} 列数: {len(merged_df.columns)}")
        return merged_df
    except Exception as e:
        print(f"获取 {ts_code} 数据时发生错误: {e}")
        return None

def main():
    """主函数：批量获取所有股票数据"""
    print("="*60)
    print("开始批量获取股票数据")
    print("="*60)
    # stock_codes为read_dir目录下所有csv文件名（去掉扩展名）
    for sector in os.listdir(root_dir):
        sector_path = os.path.join(root_dir, sector)
        if not os.path.isdir(sector_path):
            continue
        print(f"\n处理板块: {sector}")
        stock_codes = [os.path.splitext(f)[0] for f in os.listdir(sector_path) if f.endswith('.csv')]
        print(f"要获取的股票: {stock_codes}")
        output_dir = os.path.join(news_dir, sector)
        successful_stocks = []
        failed_stocks = []
        for i, ts_code in enumerate(stock_codes, 1):
            print(f"\n进度: {i}/{len(stock_codes)}")
            # result = get_stock_data(ts_code)
            result = get_stock_data(ts_code, sector_path, output_dir)
            if result is not None:
                successful_stocks.append(ts_code)
            else:
                failed_stocks.append(ts_code)
            if i < len(stock_codes):
                print("等待2秒...")
                time.sleep(2)
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