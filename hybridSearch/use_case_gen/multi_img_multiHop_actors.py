from typing import List
import asyncio
import json
import torch
from milvus_conf import MilvusColbertRetriever, client as milvus_client
from tqdm import tqdm
import logging
from text_embeding import QwenEmbeder
from prompt import Actor_search_prompt
from transformers import AutoModel, AutoTokenizer
from openai import OpenAI
import os
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
    model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v4", 
        trust_remote_code=True, 
        torch_dtype=torch.float16)
    model.to("cuda")
    return model

def process_querys(queries: List[str], mymodel) -> List[torch.Tensor]:
    """Process queries and generate embedding vectors"""
    multivector_embeddings = mymodel.encode_text(
        texts=queries,
        task="retrieval",
        prompt_name="query",
        return_multivector=True,
    )
    return multivector_embeddings

def multi_img_search(queries: List[str], file_names: List[str], topk: int, mymodel, multi_img_retriever):
    query_embeddings = process_querys(queries, mymodel)
    
    search_results_list = []
    
    for i in range(len(queries)):
        # query_np is a two-dimensional array of queries for each sentence in the query group
        query_np = query_embeddings[i].float().cpu().numpy()
        
        score_results = multi_img_retriever.multi_img_search(query_np, file_names, topk)
        
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

# By default, only one query is entered here
async def multi_img_search_multiHop_actors(
    sys_name,
    user_query,
    topk: int,
    file_names,
    mymodel,
    multi_img_retriever,
    page_count_max = 100,
): 
    history_pages = []
    multiHop_count = 0
    query = user_query
    actors = []
    history_queries = []
    assistant = ""
    while(True):
        print(actors)
        if assistant == "":
            search_results_list = await asyncio.to_thread(
                multi_img_search, [query], file_names, topk, mymodel, multi_img_retriever
            )
            search_results = search_results_list[0]
            
            if len(history_pages) >= page_count_max:
                logger.info("The maximum number of search pages has been reached, stop generating")
                break
                    
            history_pages.extend(search_results)
            history_queries.append(query)
            userInput = f"当前系统为：{sys_name}\n" + Actor_search_prompt + "\n [history actors]:" + json.dumps(actors) + "\n [history queries]:" + json.dumps(history_queries)   
        
                                
        answer = qwen_vl_max(search_results, userInput, assistant)
        try:
            res = json.loads(answer) 
            assistant = ""
            multiHop_count += 1
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
             

async def main():    
    multi_img_collection_name = "NonMD_Req"
    file_names = ["Smart City Big Data Center"]
    page_count_max = 100
    sys_name = "智慧城市大数据中心"
    queries = ["系统功能，项目介绍，业务流程是什么？",]
    
    model = init_model()
    multi_img_retriever = MilvusColbertRetriever(collection_name=multi_img_collection_name, milvus_client=milvus_client)
    
    # Create semaphore and limit maximum concurrency
    semaphore = asyncio.Semaphore(1)
    
    pbar = tqdm(total=len(queries), desc="Searching Actors")
    
    async def run_with_semaphore(query):
        async with semaphore:
            result = await multi_img_search_multiHop_actors(
                sys_name,
                query,
                5,
                file_names,
                model,
                multi_img_retriever,
                page_count_max
            )
            pbar.update(1)
            return result
            
    
    tasks = [run_with_semaphore(query) for query in queries]  
    results = await asyncio.gather(*tasks)
    
    pbar.close()
    print(results)

if __name__ == "__main__":
    asyncio.run(main())
