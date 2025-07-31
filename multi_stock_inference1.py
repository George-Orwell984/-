import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

def load_test_data(test_file):
    """加载测试数据"""
    print(f"加载测试数据: {test_file}")
    test_df = pd.read_csv(test_file)
    
    # 检查必要的列
    required_cols = ['y', 'stock_code']
    missing_cols = [col for col in required_cols if col not in test_df.columns]
    if missing_cols:
        raise ValueError(f"测试数据缺少必要列: {missing_cols}")
    
    print(f"测试数据形状: {test_df.shape}")
    print(f"标签分布: {test_df['y'].value_counts().to_dict()}")
    
    return test_df

def prepare_features(test_df, scaler, feature_columns):
    """准备测试特征，确保特征与训练时一致"""
    print("准备测试特征并确保与训练数据一致...")
    
    # 确保所有特征列都存在
    missing_features = [col for col in feature_columns if col not in test_df.columns]
    if missing_features:
        print(f"添加缺失的 {len(missing_features)} 个特征，填充为0")
        for col in missing_features:
            test_df[col] = 0
    
    # 移除测试数据中多余的特征
    extra_features = [col for col in test_df.columns if col not in feature_columns and col not in ['y', 'stock_code']]
    if extra_features:
        print(f"移除 {len(extra_features)} 个多余的特征: {extra_features}")
        test_df = test_df.drop(columns=extra_features)
    
    # 确保特征顺序与训练时一致
    ordered_cols = [col for col in feature_columns if col in test_df.columns]
    assert 'y' not in ordered_cols
    X_test = test_df[ordered_cols].values
    
    # 应用标准化器
    print(f"应用标准化器 (输入形状: {X_test.shape})")
    
    # 检查特征数量是否匹配
    if X_test.shape[1] != len(feature_columns):
        print(f"警告: 特征数量不匹配! 测试数据有 {X_test.shape[1]} 个特征, 但标准化器期望 {len(feature_columns)} 个")
        
        # 尝试填充缺失的特征
        missing_count = len(feature_columns) - X_test.shape[1]
        if missing_count > 0:
            print(f"添加 {missing_count} 个缺失特征列")
            zeros = np.zeros((X_test.shape[0], missing_count))
            X_test = np.hstack((X_test, zeros))
    
    # 标准化特征
    X_test = scaler.transform(X_test)
    
    return X_test, test_df

def predict_and_evaluate(model, X_test, y_test):
    """进行预测并评估准确率"""
    print("进行预测...")
    y_pred = model.predict(X_test)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"预测准确率: {accuracy:.4f}")
    
    # 详细评估
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    print("\n混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    
    return accuracy

def main():
    # 配置路径
    config_path = './dataset/process_notext_tom/园林工程'
    sector_results = []
    # for sector in os.listdir(config_path):
    test_file = r'F:\djy\Attention-CLX-stock-prediction\dataset\test\688271.SH_multimodal.csv'
    model_path = os.path.join(config_path, 'best_multi_stock_model.pkl')
    scaler_path = os.path.join(config_path,  'multi_stock_scaler.pkl')
    feature_columns_path = os.path.join(config_path,  'multi_stock_feature_cols.pkl')
    selected_indices_path = os.path.join(config_path,  'multi_stock_features.pkl')
    
    # 加载测试数据
    test_df = load_test_data(test_file)
    
    # 加载特征列
    if os.path.exists(feature_columns_path):
        feature_columns = joblib.load(feature_columns_path)
        print(f"加载特征列: {len(feature_columns)} 个特征")
    else:
        raise FileNotFoundError(f"未找到特征列文件: {feature_columns_path}")
    
    # 加载特征选择索引
    if os.path.exists(selected_indices_path):
        selected_indices = joblib.load(selected_indices_path)
        print(f"加载特征选择索引: {len(selected_indices)} 个特征")
    else:
        raise FileNotFoundError(f"未找到特征选择索引文件: {selected_indices_path}")
    
    # 加载标准化器
    if os.path.exists(scaler_path):
        print(f"加载标准化器: {scaler_path}")
        scaler = joblib.load(scaler_path)
    else:
        raise FileNotFoundError(f"未找到标准化器文件: {scaler_path}")
    
    # 加载模型
    if os.path.exists(model_path):
        print(f"加载模型: {model_path}")
        model = joblib.load(model_path)
    else:
        raise FileNotFoundError(f"未找到模型文件: {model_path}")
    
    # 准备特征并确保一致性
    X_test, test_df = prepare_features(test_df, scaler, feature_columns)
    y_test = test_df['y'].values
    
    # 特征选择
    X_test_selected = X_test[:, selected_indices]
    print(f"推理用特征数量: {X_test_selected.shape[1]}")
    
    # 进行预测和评估
    accuracy = predict_and_evaluate(model, X_test_selected, y_test)
    # sector_results.append({
    #     'sector': sector,
    #     'accuracy': accuracy
    # })
        # 进行预测和评估
    accuracy = predict_and_evaluate(model, X_test_selected, y_test)

    # 保存推理结果
    y_pred = model.predict(X_test_selected)
    result_df = test_df.copy()
    result_df['y_pred'] = y_pred
    # 你可以自定义保存路径
    save_path = os.path.splitext(test_file)[0] + '_with_notext_pred.csv'
    result_df.to_csv(save_path, index=False)
    print(f"推理结果已保存到: {save_path}")

    print("\n" + "="*60)
    print(f"测试完成,准确率: {accuracy:.4f}")
    print("="*60)


    # sector_df = pd.DataFrame(sector_results)
    # sector_df.to_csv(os.path.join(config_path,'test_result_000886.csv'))
if __name__ == '__main__':
    main()