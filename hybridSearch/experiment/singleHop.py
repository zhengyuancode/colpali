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




# 获取设备
device = get_torch_device("cuda")
logger.info(f"Using device: {device}")

# 模型路径配置
model_name = "/home/gpu/milvus/backend/colpali/modelcache/models--vidore--colpali-v1.2/snapshots/6b89bc63c16809af4d111bfe412e2ac6bc3c9451"
cachedir = "/home/gpu/milvus/backend/colpali/modelcache/"

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
model_load_time = time.time() - model_load_start
logger.info(f"Model loaded in {model_load_time:.2f} seconds")

# 初始化处理器
processor = ColPaliProcessor.from_pretrained(model_name)
embeder=QwenEmbeder(url="https://api.siliconflow.cn/v1/embeddings")
    
# 创建一个全局锁，用于保护文件读写
file_lock = asyncio.Lock()

async def hybridSearch(
    queries: List[str],
    uid: str,
    topk: int,
    searchMethod: str
):
    """
    执行多模态混合检索查询
    
    - **queries**: 搜索查询列表
    - **uniqueIds**: 知识库对应的uid列表
    - **customNames**: 查找知识库列表
    - **topk**: 返回的结果数量
    """ 
    collection_name = "vidoseek"
    
    customNames = []
    with open("/home/gpu/milvus/backend/colpali/ViDoSeek/subfolders.json", 'r', encoding='utf-8') as file:
        subfolders = json.load(file)
    subfolders_list = subfolders["subfolders"]
    for pdf in subfolders_list:
        customNames.append(pdf["pdfId"])
    
    try:
        if(searchMethod not in ["Muti_hybrid_search","Muti_hybrid_search_intersection","Muti_hybrid_search_img_in_text","Muti_hybrid_search_text_in_img","Muti_hybrid_search_multiple_in_single","Muti_hybrid_search_single_in_multiple"]):
            logger.warning("非法的检索方法")
            return
        

        
        logger.info(f"使用{searchMethod}方法")
        print(f"使用{searchMethod}方法")
        # 调用同步函数，处理第一个 for 循环
        search_results_list = await asyncio.to_thread(
            process_queries_hybrid, collection_name, queries, customNames, topk,searchMethod,processor,model,device,embeder,needRewrit = True
        )

        for j in range(len(search_results_list)):
            search_results = search_results_list[j]
                                        
            # 准备图片用于生成器   
            base64_images=[]
            try:
                for image_path in search_results:
                    base64_str = image_to_base64(image_path) 
                    base64_images.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_str}"}
                    })
            except Exception as e:
                logger.error(f"准备图片时出错: {str(e)}")
                continue  # 跳过当前查询
                
            try:
                response  = AIclient.chat.completions.create(
                    model="qwen-vl-max-latest", 
                    messages=[
                    {"role":"system","content":[{"type": "text", "text": "You need to combine the image information provided by the user's document page with your own knowledge base to answer the user's query. Your answer should be in English.Your answer needs to be as concise as possible."}]},
                    {
                        "role": "user",
                        "content": base64_images + [{"type": "text", "text": queries[j]}]
                    }
                    ]
                )
            except Exception as e:
                logger.error(f"调用阿里云 API 失败: {str(e)}")
                continue
            
            # 这里默认queries只有一个query
            answer = {
                "uid":uid,
                "query":queries[j],
                "answer":response.choices[0].message.content,
                "pages":search_results
                }
            
            result_file = f"{searchMethod}_results.json"
            async with file_lock:
                if os.path.exists(result_file):
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                        
                    result_data["singleHop"].append(answer)
                    
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(result_data, f, ensure_ascii=False, indent=4)
                else:
                    result_data = {"singleHop":[answer]} 
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(result_data, f, ensure_ascii=False, indent=4)
                      
            return answer
            
                  
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        return
    
    
    
    
async def main():    
    # 资源不够时改动下方代码以控制批次
    queries = []
    # problem_dataset = {"examples":[]}
            
    with open("vidoseek_singleHop.json", 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    with open("Muti_hybrid_search_img_in_text_results.json", 'r', encoding='utf-8') as file:
        results = json.load(file)
    for data in dataset["examples"]:
        needreasoning = True
        uid = data["uid"]
        for result in results["singleHop"]:
            if result["uid"] == uid:
                needreasoning = False
                break
        if needreasoning:
            queries.append({"uid":data["uid"], "query": data["query"]})
            # problem_dataset["examples"].append(data) 
            
    # with open("problem_dataset.json", 'w', encoding='utf-8') as f:
    #     json.dump(problem_dataset, f, ensure_ascii=False, indent=4)
    
    if len(queries) + len(results["singleHop"]) != len(dataset["examples"]):
        print(f"未实验数据列表构建失败")
        return
    print(len(queries))
    print(len(results["singleHop"]))
    print(len(dataset["examples"]))
    # 创建信号量，限制最大并发数为 10
    semaphore = asyncio.Semaphore(10)

    async def run_with_semaphore(query):
        async with semaphore:
            return await hybridSearch(
                [query["query"]],
                query["uid"],
                5,
                "Muti_hybrid_search_img_in_text"
            )
    
    tasks = [run_with_semaphore(query) for query in queries]  
    results = await asyncio.gather(*tasks)
    return results

if __name__ == "__main__":
    asyncio.run(main())