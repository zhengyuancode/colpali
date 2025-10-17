from typing import List
import asyncio
import json
import torch
from milvus_conf import MilvusColbertRetriever, client as milvus_client
from tqdm import tqdm
import logging
from text_embeding import QwenEmbeder
from prompt import use_case_search_prompt
from transformers import AutoModel
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
async def multi_img_search_multiHop_usecases(
    sys_name,
    actor,
    topk: int,
    file_names,
    mymodel,
    multi_img_retriever,
    multiHop_count_max = 10
): 
    multiHop_count = 0
    actor_valid = "yes"
    query = actor+"的目标是什么，在系统中有哪些用例？"
    use_cases = []
    history_queries = []
    assistant = ""
    while(multiHop_count < multiHop_count_max):
        if assistant == "":
            search_results_list = await asyncio.to_thread(
                multi_img_search, [query], file_names, topk, mymodel, multi_img_retriever
            )
            search_results = search_results_list[0]
            history_queries.append(query)
            
            userInput = f"当前系统为：{sys_name}\n" + use_case_search_prompt + "\n [actor]:" + actor + "\n [history use cases]:" + json.dumps(use_cases) + "\n [history queries]:" + json.dumps(history_queries)                       
        answer = qwen_vl_max(search_results, userInput, assistant)
        try:
            res = json.loads(answer)
            assistant = ""
            multiHop_count += 1
            if res["actor_valid"] == "no":
                use_cases = []
                actor_valid = "no"
                break
            use_cases = res["use_cases"]
            if res["query"].strip() == "": 
                break
            else:
                query = res["query"]
        except Exception as e:
            logger.error(f"VLM answer format error: {str(e)}")
            assistant = answer
            continue

   
    return {"actor": actor, "use_cases": use_cases,"actor_valid": actor_valid}       
             

async def main():    
    multi_img_collection_name = "NonMD_Req"
    file_names = ["Smart City Big Data Center"]
    sys_name = "智慧城市大数据中心"
    
    # actors = ['租户', '系统管理员', '超级管理员', '运维人员', '部门管理员', '订阅管理员', '中心管理员', '普通用户', '外部系统', '服务器', '数据采集系统', '数据集成系统', 'Hadoop分布式文件系统(HDFS)', 'HBase', 'Elasticsearch', 'MySQL', 'Hive', 'API网关平台', '消息服务中间件', '短信服务提供商', '产品设计人员', 'DevOps工程师', '监控告警系统', '资源目录管理员', '场景管理员', '数据资产管理员', 'API发布专员', '微应用发布专员', '角色管理员', '数据质量审核员', '安全审计员', '合规性审查员', '内容审核员', 'CMS操作员', '自动化运维工具', '数据分析师', '架构师']
    actors = ['租户', '系统管理员']
    
    model = init_model()
    multi_img_retriever = MilvusColbertRetriever(collection_name=multi_img_collection_name, milvus_client=milvus_client)
    
    # Create semaphore and limit maximum concurrency
    semaphore = asyncio.Semaphore(10)
    
    pbar = tqdm(total=len(actors), desc="Searching UseCases")
    
    async def run_with_semaphore(actor):
        async with semaphore:
            result = await multi_img_search_multiHop_usecases(
                sys_name,
                actor,
                5,
                file_names,
                model,
                multi_img_retriever
            )
            pbar.update(1)
            return result
            
    
    tasks = [run_with_semaphore(actor) for actor in actors]  
    results = await asyncio.gather(*tasks)
    
    pbar.close()
    print(results)

if __name__ == "__main__":
    asyncio.run(main())
