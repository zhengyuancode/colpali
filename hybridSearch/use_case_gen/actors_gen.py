from typing import List
import asyncio
import json
import torch
from milvus_conf import MilvusColbertRetriever, client as milvus_client
from tqdm import tqdm
import logging
from prompt import Actor_search_prompt,Actor_search_prompt2
from transformers import AutoModel
from openai import OpenAI
import os
from stable_baselines3 import PPO
import pickle
from feature import build_state_for_query
import base64
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QWENAPIKEY="sk-f78b07615c8a45128d760579e6d42e1f"
AIclient = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=QWENAPIKEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def init_model():
    #local load
    embeder_path = "/home/gpu/.cache/huggingface/hub/models--jinaai--jina-embeddings-v4/snapshots/737fa5c46f0262ceba4a462ffa1c5bcf01da416f"
    topNmodel_path = "/home/gpu/dzy/M3-CaseRAG/MDP/models/best_model.zip"
    scaler = pickle.load(open("/home/gpu/dzy/M3-CaseRAG/MDP/scaler.pkl", "rb"))
    topNmodel = PPO.load(topNmodel_path)
    embeder = AutoModel.from_pretrained(
        embeder_path,
        trust_remote_code=True, 
        torch_dtype=torch.float16,
        local_files_only=True
        )
    embeder.to("cuda")
    return embeder, topNmodel, scaler

def multi_img_search(queries: List[str], file_names: List[str], topk: int, embeder,  topNmodel, scaler, multi_img_retriever):
    search_results_list = []
    
    for i in range(len(queries)):
        query  = queries[i]
        
        state, query_np = build_state_for_query(query, embeder, scaler=scaler)
        action, _ = topNmodel.predict(state, deterministic=True)
        topN = int(action) + 1
        score_results = multi_img_retriever.multi_img_search(query_np, file_names, topk, topN)
        
        search_results = []
        
        for sitem in score_results:
            search_results.append(sitem[3])  
        search_results_list.append(search_results)
    
    return search_results_list

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        # 读取二进制数据并进行 Base64 编码
        base64_data = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_data

def qwen_vl_max(image_paths, userInput, assistant = ""):
    base64_images=[]
    try:
        for image_path in image_paths:
            base64_str = image_to_base64(image_path) 
            base64_images.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_str}"}
            })
    except Exception as e:
        logger.error(f"准备图片时出错: {str(e)}")
    if assistant == "":
        completion = AIclient.chat.completions.create(
            model="qwen-vl-max",
            messages=[
                {
                    "role": "user",
                    "content": base64_images + [{"type": "text", "text": userInput}]
                },
            ],
            stream=False,
        )
    else:
        completion = AIclient.chat.completions.create(
            model="qwen-vl-max",
            messages=[
                {
                    "role": "user",
                    "content": base64_images + [{"type": "text", "text": userInput}]
                },
                {
                    "role": "assistant",
                    "content": assistant
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "上一轮回答格式出错无法解析，请严格按照要求的JSON格式返回结果，不要返回多余的任何文本。"}]
                },
            ],
            stream=False,
        )
    return completion.choices[0].message.content


def gen_actor(
    system_description,
    user_query,
    topk: int,
    file_names,
    embeder,
    topNmodel,
    scaler,
    multi_img_retriever,
    page_count_max = 100,
    repeat_limit = 3
): 
    history_pages = []
    multiHop_count = 0
    query = user_query
    actors = []
    repeat = 0
    history_queries = []
    assistant = ""
    while(len(history_pages) < page_count_max):
        print(actors)
        if assistant == "":
            search_results_list =  multi_img_search([query], file_names, topk, embeder, topNmodel, scaler, multi_img_retriever)
            
            search_results = search_results_list[0]
                    
            history_pages.extend(search_results)
            history_queries.append(query)
            userInput = Actor_search_prompt + f"{system_description}\n [history actors]:" + json.dumps(actors) + "\n [history queries]:" + json.dumps(history_queries)   
        
                                
        answer = qwen_vl_max(search_results, userInput, assistant)
        try:
            res = json.loads(answer) 
            assistant = ""
            multiHop_count += 1
            if res["actors"] == actors:
                repeat += 1
                print(f"重复次数{repeat}")
                if repeat > repeat_limit:
                    break
            else:
                repeat = 0
            actors = res["actors"]
            if res["query"].strip() == "": 
                break
            else:
                query = res["query"]
        except Exception as e:
            logger.error(f"VLM answer format error: {str(e)}")
            assistant = answer
            continue

   
    return actors       
             

def main():    
    multi_img_collection_name = "OR2UC"
    multi_img_retriever = MilvusColbertRetriever(collection_name=multi_img_collection_name, milvus_client=milvus_client)
    file_names = ["ant rent"]
    page_count_max = multi_img_retriever.count_page_by_file(file_names)
    repeat_limit = 3
    system_description = "蚂蚁短租租房系统"
    first_query = "系统功能，项目介绍，业务流程是什么？"
    topk = 2
    
    embeder, topNmodel, scaler = init_model()
    
    result = gen_actor(
                system_description,
                first_query,
                topk,
                file_names,
                embeder,
                topNmodel,
                scaler,
                multi_img_retriever,
                page_count_max,
                repeat_limit
            )
    
    
    print(result)

if __name__ == "__main__":
    main()
