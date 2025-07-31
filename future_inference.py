import pandas as pd
import numpy as np
import joblib
import os

def main():
    # 路径配置
    data_path = r'F:\djy\Attention-CLX-stock-prediction\dataset\test\test_multimodal_688271_future.csv'
    config_path = './dataset/test/互联网'
    model_path = os.path.join(config_path, 'best_multi_stock_model.pkl')
    scaler_path = os.path.join(config_path, 'multi_stock_scaler.pkl')
    feature_columns_path = os.path.join(config_path, 'multi_stock_feature_cols.pkl')
    selected_indices_path = os.path.join(config_path, 'multi_stock_features.pkl')

    # 加载数据
    df = pd.read_csv(data_path)
    print(f"未来一天多模态特征 shape: {df.shape}")

    # 加载模型和工具
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_columns = joblib.load(feature_columns_path)
    selected_indices = joblib.load(selected_indices_path)

    # 只保留特征列（去除stock_code, trade_date等非特征列）
    feature_cols = [col for col in df.columns if col in feature_columns]
    X = df[feature_cols].fillna(0).values

    # 标准化
    X = scaler.transform(X)
    # 特征选择
    X_selected = X[:, selected_indices]

    # 预测
    y_pred = model.predict(X_selected)
    y_pred_proba = model.predict_proba(X_selected)[:, 1] if hasattr(model, 'predict_proba') else None

    # 输出结果
    print("\n====== 未来一天预测结果 ======")
    print(f"预测y值: {y_pred[0]} ({'上涨' if y_pred[0]==1 else '下跌'})")
    if y_pred_proba is not None:
        print(f"预测为上涨的概率: {y_pred_proba[0]:.4f}")
        if y_pred_proba[0] > 0.7:
            print("置信度: 高")
        elif y_pred_proba[0] > 0.6:
            print("置信度: 中等")
        else:
            print("置信度: 低")
    print("=============================")

if __name__ == '__main__':
    main() 