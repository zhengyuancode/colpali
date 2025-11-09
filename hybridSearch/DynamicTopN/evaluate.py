# evaluate.py
import joblib
from pymilvus import MilvusClient
from feature import build_state_for_query
from env import compute_recall_at_5
from train import init_model, load_data_eval, patch_patch_search
import json
import pickle
import time

def load_data_test(path):
    with open(path, 'r', encoding='utf-8') as file:
        test_data = json.load(file)
    return test_data

def main():
    # load model
    model = joblib.load('./model/topn_model.pkl')
    embeder = init_model()
    milvus_client = MilvusClient(uri="http://127.0.0.1:19530")
    scaler = pickle.load(open("scaler.pkl", "rb"))
    max_time = 3.0
    beta = 0.15
    
    test_data = load_data_test("/home/gpu/dzy/M3-CaseRAG/MDP/datasets/test.json")
    eval_results = []
    total = 0
    overall_recall5 = 0.0
    overall_reward = 0.0
    acc_count = 0
    overall_time = 0.0
    for data in test_data:
        state, query_np = build_state_for_query(data["query"], embeder, scaler=scaler)
        topN = model.predict([state])[0]
        # topN = 15
        t0 = time.time()
        pages, scores = patch_patch_search(
            query_np, data["collection_name"], [data["file_name"]],
            milvus_client, coarse=topN
        )
        total += 1
        retrieval_time = time.time() - t0
        recall_at_5 = compute_recall_at_5(data["evidence_page_nums"], pages)
        retrieval_time_norm = min(1.0, retrieval_time / max_time)
        reward = recall_at_5 - beta * retrieval_time_norm
        if recall_at_5 == 1.0:
            accuracy = 1
            acc_count += 1
        else:
            accuracy = 0
        
        overall_recall5 += recall_at_5
        overall_reward += reward
        overall_time += retrieval_time
        eval_result = {
            "query": data["query"],
            "predicted_topN": topN,
            "retrieved_pages": pages[:5],
            "recall_at_5": recall_at_5,
            "retrieval_time": retrieval_time,
            "reward": reward,
            "accuracy": accuracy
        }
        eval_results.append(eval_result)
        
    statistics ={
        "total": total,
        "average_recall_at_5": round(overall_recall5 / total, 4),
        "average_reward": round(overall_reward / total, 4),
        "accuracy": round(acc_count / total, 4),
        "overall_retrieval_time": overall_time
    }
    
    with open("./results/test_results_15.json", "w", encoding="utf-8") as f:
        json.dump({"eval_results": eval_results, "statistics": statistics}, f, indent=4)
        

if __name__ == "__main__":
    main()