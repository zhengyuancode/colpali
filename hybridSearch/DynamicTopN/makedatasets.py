# train.py
import os
import numpy as np
import pickle
import time
import torch
from transformers import AutoModel
from pymilvus import MilvusClient
import concurrent.futures
from feature import build_state_for_query
from sklearn.preprocessing import StandardScaler
import json
import torch
from pathlib import Path
from feature import build_state_for_query


def init_model():
    #local load
    model_path = "/home/gpu/.cache/huggingface/hub/models--jinaai--jina-embeddings-v4/snapshots/737fa5c46f0262ceba4a462ffa1c5bcf01da416f"
    
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True, 
        torch_dtype=torch.float16,
        local_files_only=True
        )
    model.to("cuda")
    return model

def load_data_train(path):
    with open(path, 'r', encoding='utf-8') as file:
        train_data = json.load(file)
    return train_data

def load_data_eval(path):
    with open(path, 'r', encoding='utf-8') as file:
        eval_data = json.load(file)
    return eval_data
    
def patch_patch_search(multi_query, collection_name, file_names, milvus_client, coarse):  
    try: 
        results = milvus_client.search(
            collection_name,
            multi_query,
            limit=coarse,
            anns_field="patch_dense",
            filter=f'file_name in {file_names}',
            output_fields=["patch_dense", "seq_id", "page_num","file_name"],
            search_params={"metric_type": "IP"}
        )
    except Exception as e:
        print("Multi_img retrieval failed:\n")
        print(str(e))
                
    pages = []
    seen = set()  # 用于跟踪已见的唯一标识
    for r_id in range(len(results)):
        for r in range(len(results[r_id])):
            # 提取文档的唯一标识组合
            page_num = results[r_id][r]["entity"]["page_num"]
            file_name = results[r_id][r]["entity"]["file_name"]
            unique_key = (page_num, file_name)  # 创建不可变的唯一键
            
            # 仅当未出现过时才添加到列表
            if unique_key not in seen:
                seen.add(unique_key)
                pages.append({
                    "page_num": page_num,
                    "file_name": file_name
                })
    if len(pages) == 0:
        raise ValueError("No pages retrieved from Milvus.")      
    score_results = []
    def rerank_single_page(page, multi_query, client, collection_name):
        # Rerank a single document by retrieving its embeddings and calculating the similarity with the query.
        page_num = page["page_num"]
        file_name = page["file_name"]
        page_colbert_vecs = client.query(
            collection_name=collection_name,
            filter=f'page_num == {page_num} and file_name == "{file_name}"',
            output_fields=["seq_id", "patch_dense", "page_path"]
        )
        if len(page_colbert_vecs) == 0:
            raise ValueError(f"No embeddings found for page_num {page_num} and file_name {file_name}.")
        page_vecs = np.vstack(
            [page_colbert_vecs[i]["patch_dense"] for i in range(len(page_colbert_vecs))]
        )
        score = np.dot(multi_query, page_vecs.T).max(1).sum()
        page_path=""
        for item in page_colbert_vecs:
            if item["seq_id"] == 0:
                page_path = item["page_path"]
                break 
        return (score, file_name, page_num, page_path)

    with concurrent.futures.ThreadPoolExecutor(os.cpu_count()) as executor:
        workers = {
            executor.submit(
                rerank_single_page, page, multi_query, milvus_client, collection_name
            ): page
            for page in pages
        }
        for worker in concurrent.futures.as_completed(workers):
            score, file_name, page_num, page_path = worker.result()
            score_results.append((score, file_name, page_num, page_path))
    score_results.sort(key=lambda x: x[0], reverse=True)
    if len(score_results) == 0:
        raise ValueError("No score_results retrieved from Milvus.") 
    
    pages = []
    scores = []
    for sitem in score_results:
        pages.append(sitem[3])  
        scores.append(sitem[0])
    
    return pages, scores

def compute_recall_at_5(gt, docs):
    if gt is None or len(gt) == 0:
        return 0.0
    topk = docs[:5]
    topk_pages = [int(Path(p).stem) for p in topk]
    hits = sum(1 for d in topk_pages if d in gt)
    return hits / min(len(gt), 5)

# ----------------------------------------------------------------
if __name__ == "__main__":
    beta = 0.15
    max_time = 3.0
    datasets_path = "./datasets/dynamic_topn_datasets.json"

    train_data = load_data_train("/home/gpu/dzy/M3-CaseRAG/MDP/datasets/train.json")
    embeder = init_model()
    milvus_client = MilvusClient(uri="http://127.0.0.1:19530")
    
    
    # fit scaler
    if not os.path.exists("scaler.pkl"):
        scaler = StandardScaler()  
        X = [build_state_for_query(q["query"], embeder)[0] for q in train_data[:200]]
        scaler.fit(np.stack(X))
        pickle.dump(scaler, open("scaler.pkl","wb"))
        print("Scaler fitted and saved.")
    else:
        scaler = pickle.load(open("scaler.pkl","rb"))
    STopN = []
    total = 0
    for data in train_data:
        state, query_np = build_state_for_query(data["query"], embeder, scaler)
        best_reward = 0.0
        best_topn = 1
        best_recall_at_5 = 0.0
        best_retrieval_time = 0.0
        for topN in range(1,16):
            t0 = time.time()
            pages, scores = patch_patch_search(query_np, data["collection_name"], [data["file_name"]], milvus_client, topN) 
            retrieval_time = time.time() - t0
            recall_at_5 = compute_recall_at_5(data["evidence_page_nums"], pages) 
            retrieval_time_norm = min(1.0, retrieval_time / max_time)
            reward = recall_at_5 - beta * retrieval_time_norm
            reward = float(np.clip(reward, -1.0, 1.0))
            if reward > best_reward:
                best_reward = reward
                best_topn = topN
                best_recall_at_5 = recall_at_5
                best_retrieval_time = retrieval_time
        STopN.append({
            "query": data["query"],
            "state": state.tolist(),
            "best_topn": best_topn,
            "best_recall_at_5": best_recall_at_5,
            "best_retrieval_time": best_retrieval_time,
            "best_reward": best_reward
        })
        total += 1
        if total % 100 == 0:
            with open(datasets_path, 'w', encoding='utf-8') as f:
                json.dump(STopN, f, ensure_ascii=False, indent=4) 
    with open(datasets_path, 'w', encoding='utf-8') as f:
            json.dump(STopN, f, ensure_ascii=False, indent=4)             
    print(f"datasets saved to {datasets_path}")