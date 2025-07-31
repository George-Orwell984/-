#!/usr/bin/env python3
"""
诊断训练问题：验证集loss不下降
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from multi_stock_training_all import get_all_sectors, load_and_preprocess_data, create_time_series_features

def diagnose_data_quality(sector_name):
    """诊断数据质量"""
    print(f"="*60)
    print(f"诊断 {sector_name} 板块数据质量")
    print(f"="*60)
    
    # 加载数据
    df = load_and_preprocess_data(sector_name)
    if df is None:
        return
    
    # 特征工程
    df = create_time_series_features(df, lookback_window=30)
    
    # 1. 检查标签分布
    print("\n1. 标签分布分析:")
    label_dist = df['y'].value_counts()
    print(f"   标签分布: {label_dist.to_dict()}")
    print(f"   不平衡比例: {label_dist.max() / label_dist.min():.2f}")
    
    if label_dist.max() / label_dist.min() > 2:
        print("   ⚠️  标签分布不平衡，可能影响训练")
    
    # 2. 检查特征数量
    feature_cols = [col for col in df.columns if col not in ['y', 'stock_code', 'sector', 'trade_date']]
    print(f"\n2. 特征分析:")
    print(f"   特征数量: {len(feature_cols)}")
    
    # 3. 检查特征与标签的相关性
    print("\n3. 特征相关性分析:")
    correlations = []
    for col in feature_cols:
        corr = abs(df[col].corr(df['y']))
        correlations.append((col, corr))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    print(f"   最高相关性: {correlations[0][1]:.4f} ({correlations[0][0]})")
    print(f"   最低相关性: {correlations[-1][1]:.4f} ({correlations[-1][0]})")
    print(f"   平均相关性: {np.mean([c[1] for c in correlations]):.4f}")
    
    # 检查高相关特征
    high_corr = [c for c in correlations if c[1] > 0.5]
    print(f"   相关性>0.5的特征数量: {len(high_corr)}")
    
    if len(high_corr) == 0:
        print("   ⚠️  没有高相关性特征，模型可能难以学习")
    
    # 4. 检查数据泄露
    print("\n4. 数据泄露检查:")
    very_high_corr = [c for c in correlations if c[1] > 0.8]
    if very_high_corr:
        print(f"   ⚠️  发现 {len(very_high_corr)} 个极高相关性特征，可能存在数据泄露:")
        for col, corr in very_high_corr[:3]:
            print(f"      {col}: {corr:.4f}")
    else:
        print("   ✅ 未发现明显的数据泄露")
    
    # 5. 检查时间序列质量
    print("\n5. 时间序列质量检查:")
    stock_counts = df['stock_code'].value_counts()
    print(f"   股票数量: {len(stock_counts)}")
    print(f"   每只股票平均数据量: {stock_counts.mean():.1f}")
    print(f"   最少数据量: {stock_counts.min()}")
    print(f"   最多数据量: {stock_counts.max()}")
    
    # 检查是否有足够的数据进行训练
    lookback_window = 30
    min_required = lookback_window + 50
    sufficient_stocks = stock_counts[stock_counts >= min_required]
    print(f"   有足够数据的股票数量: {len(sufficient_stocks)}")
    
    if len(sufficient_stocks) < 5:
        print("   ⚠️  有足够数据的股票太少，可能影响训练效果")
    
    # 6. 可视化分析
    print("\n6. 生成诊断图表...")
    
    plt.figure(figsize=(15, 10))
    
    # 标签分布
    plt.subplot(2, 3, 1)
    label_dist.plot(kind='bar')
    plt.title('标签分布')
    plt.xlabel('标签')
    plt.ylabel('数量')
    
    # 特征相关性分布
    plt.subplot(2, 3, 2)
    corr_values = [c[1] for c in correlations]
    plt.hist(corr_values, bins=50, alpha=0.7)
    plt.title('特征相关性分布')
    plt.xlabel('相关性绝对值')
    plt.ylabel('特征数量')
    
    # 股票数据量分布
    plt.subplot(2, 3, 3)
    plt.hist(stock_counts.values, bins=20, alpha=0.7)
    plt.title('股票数据量分布')
    plt.xlabel('数据量')
    plt.ylabel('股票数量')
    
    # 前10个最相关特征
    plt.subplot(2, 3, 4)
    top_10 = correlations[:10]
    features = [c[0][:20] + '...' if len(c[0]) > 20 else c[0] for c in top_10]
    corrs = [c[1] for c in top_10]
    plt.barh(range(len(features)), corrs)
    plt.yticks(range(len(features)), features)
    plt.title('前10个最相关特征')
    plt.xlabel('相关性')
    
    # 时间序列长度分布
    plt.subplot(2, 3, 5)
    plt.hist(stock_counts.values, bins=20, alpha=0.7, cumulative=True, density=True)
    plt.title('股票数据量累积分布')
    plt.xlabel('数据量')
    plt.ylabel('累积比例')
    
    # 相关性vs特征重要性
    plt.subplot(2, 3, 6)
    plt.scatter(range(len(correlations)), [c[1] for c in correlations], alpha=0.6)
    plt.title('特征相关性排序')
    plt.xlabel('特征排名')
    plt.ylabel('相关性')
    
    plt.tight_layout()
    plt.savefig(f'{sector_name}_diagnosis.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    return {
        'sector': sector_name,
        'total_samples': len(df),
        'feature_count': len(feature_cols),
        'stock_count': len(stock_counts),
        'label_imbalance': label_dist.max() / label_dist.min(),
        'avg_correlation': np.mean([c[1] for c in correlations]),
        'high_corr_features': len(high_corr),
        'sufficient_stocks': len(sufficient_stocks)
    }

def diagnose_training_issues():
    """诊断训练问题"""
    print("="*60)
    print("训练问题诊断")
    print("="*60)
    
    sectors = get_all_sectors()
    if not sectors:
        print("没有找到可用板块")
        return
    
    results = []
    
    for sector in sectors[:3]:  # 只诊断前3个板块
        try:
            result = diagnose_data_quality(sector)
            if result:
                results.append(result)
        except Exception as e:
            print(f"诊断 {sector} 时出错: {e}")
    
    # 汇总分析
    if results:
        print("\n" + "="*60)
        print("诊断结果汇总")
        print("="*60)
        
        df_results = pd.DataFrame(results)
        print(df_results)
        
        # 分析问题
        print("\n潜在问题分析:")
        
        for _, row in df_results.iterrows():
            sector = row['sector']
            issues = []
            
            if row['label_imbalance'] > 2:
                issues.append("标签不平衡")
            
            if row['avg_correlation'] < 0.1:
                issues.append("特征相关性低")
            
            if row['high_corr_features'] < 10:
                issues.append("高相关特征少")
            
            if row['sufficient_stocks'] < 10:
                issues.append("有效股票少")
            
            if issues:
                print(f"  {sector}: {'; '.join(issues)}")
            else:
                print(f"  {sector}: ✅ 数据质量良好")

if __name__ == '__main__':
    diagnose_training_issues() 