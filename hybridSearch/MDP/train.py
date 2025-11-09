# train.py
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from env import RAGTopNEnv
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
    
def patch_patch_search(multi_query, collection_name, file_names, milvus_client, coarse = 5):  
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

# ----------------------------------------------------------------
if __name__ == "__main__":
    # hyperparams
    total_timesteps = 15000 # adjust
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)
    beta = 0.15
    max_time = 3.0

    train_data = load_data_train("./datasets/train.json")
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

    env = Monitor(RAGTopNEnv(train_data, patch_patch_search, embeder, milvus_client, scaler, beta=beta, max_time=max_time))
    model = PPO("MlpPolicy", 
                env,
                learning_rate=3e-5, 
                n_steps=512,
                batch_size=64,
                n_epochs=10,
                gamma=0.95,
                gae_lambda=0.9,
                clip_range=0.2,
                ent_coef=0.005,
                vf_coef=0.5,
                max_grad_norm=0.5,
                normalize_advantage=True,
                target_kl=0.01,
                policy_kwargs=dict(
                    net_arch=dict(pi=[64, 32], vf=[64, 32]),
                    activation_fn=torch.nn.ReLU),
                verbose=1,
                tensorboard_log="./logs/ppo_rag_topn",
                device="cpu")
    # Callbacks
    eval_data = load_data_eval("./datasets/eval.json")
    checkpoint_callback = CheckpointCallback(save_freq=1500, save_path=model_dir, name_prefix="ppo_rag")
    eval_env = Monitor(RAGTopNEnv(eval_data, patch_patch_search, embeder, milvus_client, scaler, beta=beta, max_time=max_time))
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=model_dir,
                                 log_path="./logs/eval", 
                                 eval_freq=1500, 
                                 n_eval_episodes=len(eval_data), 
                                 deterministic=True)

    model.learn(total_timesteps=total_timesteps, 
                callback=[checkpoint_callback, eval_callback],
                reset_num_timesteps=True)
    model.save(os.path.join(model_dir, "ppo_rag_final"))
    print("Training finished.")
