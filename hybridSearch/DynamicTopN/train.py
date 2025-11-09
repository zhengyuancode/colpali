import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
import joblib

def main():
    datasets_path = "./datasets/dynamic_topn_datasets.json"
    with open(datasets_path, 'r', encoding='utf-8') as file:
            STopN = json.load(file)
            
    # 提取特征和标签
    X = np.array([item["state"] for item in STopN]) 
    y = np.array([item["best_topn"] for item in STopN]) 
    
    # 创建随机森林分类器（15个类别，100棵树）
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=5, 
        random_state=42,
        n_jobs=-1  
    )

    model.fit(X, y)
    train_acc = model.score(X, y)
    print(f"训练准确率: {train_acc:.4f}")

    joblib.dump(model, './model/topn_model.pkl')
    print("模型已保存至 ./model/topn_model.pkl")
    
if __name__ == "__main__":
    main()



