from typing import List
from fastapi_server import logger,processing_lock,process_queries_hybrid,image_to_base64,AIclient
import time
from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device
import asyncio
import json
import torch
from text_embeding import QwenEmbeder
import os
from transformers import pipeline
from transformers import AutoModel
from datasets import load_dataset,load_from_disk
from tqdm import tqdm



# 获取设备
device = get_torch_device("cuda")
logger.info(f"Using device: {device}")

# 模型路径配置
cachedir = "/root/autodl-tmp/cpdfr-data/modelcache/huggingface/hub"
model_name = "/root/autodl-tmp/cpdfr-data/modelcache/huggingface/hub/models--vidore--colpali-v1.3/snapshots/1b5c8929330df1a66de441a9b5409a878f0de5b0"
model_name_2 = "/root/autodl-tmp/cpdfr-data/modelcache/huggingface/hub/models--jinaai--jina-embeddings-v4/snapshots/737fa5c46f0262ceba4a462ffa1c5bcf01da416f"

# 加载模型 1
logger.info(f"Loading model: {model_name}")
model_load_start = time.time()
model = ColPali.from_pretrained(
    model_name,
    cache_dir=cachedir,
    torch_dtype=torch.bfloat16,
    device_map=device,
    local_files_only=True,
    use_safetensors=True
).eval()

model_2 = None
# model_2 = AutoModel.from_pretrained(
#         model_name_2,
#         trust_remote_code=True,
#         torch_dtype=torch.float16,
#         cache_dir=cachedir,                # 指定缓存路径
#         local_files_only=True,              # 强制离线加载
#     )
# model_2.to("cuda")
model_load_time = time.time() - model_load_start
logger.info(f"Model loaded in {model_load_time:.2f} seconds")

# 初始化处理器
processor = ColPaliProcessor.from_pretrained(model_name)
embeder=QwenEmbeder(url="https://api.siliconflow.cn/v1/embeddings")
    

async def hybridSearch(
    queries: List[str],
    uid: str,
    topk: int,
    searchMethod: str,
    customNames,
    collection_name
):
    """
    执行多模态混合检索查询
    
    - **queries**: 搜索查询列表
    - **uniqueIds**: 知识库对应的uid列表
    - **customNames**: 查找知识库列表
    - **topk**: 返回的结果数量
    """ 
    
    try:
        if(searchMethod not in ["Muti_hybrid_search","Muti_hybrid_search_intersection","Muti_hybrid_search_img_in_text","Muti_hybrid_search_text_in_img","Muti_vector_Img_search","Muti_hybrid_search_multiple_in_single","Muti_hybrid_search_single_in_multiple"]):
            logger.warning("非法的检索方法")
            return
        

        
        logger.info(f"使用{searchMethod}方法")
        print(f"使用{searchMethod}方法")
        # 调用同步函数，处理第一个 for 循环
        if(searchMethod in ["Muti_hybrid_search_single_in_multiple","Muti_hybrid_search_multiple_in_single"]):
            search_results_list = await asyncio.to_thread(
                process_queries_hybrid, collection_name, queries, customNames, topk,searchMethod,processor,model,model_2,device,embeder,needRewrit = False
            )
        else:
            search_results_list = await asyncio.to_thread(
                process_queries_hybrid, collection_name, queries, customNames, topk,searchMethod,processor,model,None,device,embeder,needRewrit = False
            )

        for j in range(len(search_results_list)):
            search_results = search_results_list[j]
                                        
            # 准备图片用于生成器   
            # base64_images=[]
            # try:
            #     for image_path in search_results:
            #         base64_str = image_to_base64(image_path) 
            #         base64_images.append({
            #             "type": "image_url",
            #             "image_url": {"url": f"data:image/png;base64,{base64_str}"}
            #         })
            # except Exception as e:
            #     logger.error(f"准备图片时出错: {str(e)}")
            #     continue  # 跳过当前查询
            
            #需要进行检索结果与答案都正确的实验    
            # try:
            #     response  = AIclient.chat.completions.create(
            #         model="qwen-vl-max-latest", 
            #         messages=[
            #         {"role":"system","content":[{"type": "text", "text": "You need to combine the image information provided by the user's document page with your own knowledge base to answer the user's query. Your answer should be in English.Your answer needs to be as concise as possible."}]},
            #         {
            #             "role": "user",
            #             "content": base64_images + [{"type": "text", "text": queries[j]}]
            #         }
            #         ]
            #     )
            # except Exception as e:
            #     logger.error(f"调用阿里云 API 失败: {str(e)}")
            #     continue
            
            # # 这里默认queries只有一个query
            # answer = {
            #     "uid":uid,
            #     "query":queries[j],
            #     "answer":response.choices[0].message.content,
            #     "pages":search_results
            #     }
            
            # 只进行检索结果的实验
            answer = {
                "uid":uid,
                "query":queries[j],
                "answer":"",
                "pages":search_results
                }
            
            # async with result_lock:
            #         result_data["singleHop"].append(answer)
                      
            return answer
            
                  
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        return   
    
def getQueries(ds):
    Queries = []
    for item in tqdm(ds, desc="Getting Queries", total=len(ds)):
        Queries.append({"uid":item["questionId"],"query":item["query"]})
    print("get queries done")
    return Queries

async def main():    
    
    collection_name = "vidore_docvqa"
    searchMethod = "Muti_hybrid_search_text_in_img"
    
    ds = load_from_disk("./vidore_data/docvqa_test_subsampled")
    
    queries = getQueries(ds)
    
    # 资源不够时改动下方代码以控制批次
    # with open("vidoseek_singleHop.json", 'r', encoding='utf-8') as file:
    #     dataset = json.load(file)
        
    # for i in range(len(dataset["examples"])): 
    #         data = dataset["examples"][i]
    #         queries.append({"uid":data["uid"], "query": data["query"]})
    
    # 创建信号量，限制最大并发数
    semaphore = asyncio.Semaphore(10)

    async def run_with_semaphore(query):
        async with semaphore:
            return await hybridSearch(
                [query["query"]],
                query["uid"],
                5,
                searchMethod,
                [collection_name],
                collection_name
            )
    
    tasks = [run_with_semaphore(query) for query in queries]  
    results = await asyncio.gather(*tasks)
    result_file = f"{searchMethod}_results.json"   
    result_data = None 
    
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        result_data["singleHop"] = result_data["singleHop"] + results
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4) 
    else:
        result_data = {"singleHop":results} 
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4) 
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
