from typing import List
import asyncio
import json
from milvus_conf import MilvusColbertRetriever, client as milvus_client
from tqdm import tqdm
from zhipuai import ZhipuAI
from text_embeding import QwenEmbeder
import logging

ZHIPUAPIKEY="f890fa44ea384a6baab00c725701a04b.1h0evvTQSAZALIp0"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def query_rewrit(queries,language):
    client = ZhipuAI(api_key=ZHIPUAPIKEY)
    rewriteQuerys=[]
    for query in queries:  
        response = client.chat.completions.create(
            model="GLM-4-Flash-250414",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional information retrieval query statement optimization expert, skilled in rewriting colloquial queries into precise statements suitable for professional document retrieval."
                },
                {
                    "role": "user",
                    "content": f"Rewrite the following query statement, which can be made more suitable for multimodal professional document retrieval through reasonable methods such as adding, deleting, and modifying words. Rewrite using {language}: \n {query} \n Rewrite result:"
                },
            ],
        )
        logger.info(f"{query} rewrite to {response.choices[0].message.content}")
        rewriteQuerys.append(response.choices[0].message.content) 
    return rewriteQuerys

def hybrid_text_search(queries: List[str], file_names: List[str], topk: int,embeder, retriever, needRewrit = True):
    if needRewrit:   
        queries = query_rewrit(queries, "english")
    else:
        queries = queries
    
    search_results_list = []
    
    for query in queries:   
        query_params={
            "text_dense": embeder.getTextEmbeddings(query),
            "text": query,
            "file_names": file_names
        }
        
        results = retriever.hybrid_text_search(query_params, topk)

        page_paths = []
        for item in results:
            page_paths.append(item["page_path"])
        search_results_list.append(page_paths)
    
    return search_results_list


# By default, only one query is entered here
async def hybrid_text_search_answer(
    query,
    img_path,
    topk: int,
    file_names,
    embeder, 
    retriever
): 
    try:
        search_results_list = await asyncio.to_thread(
            hybrid_text_search, [query], file_names, topk, embeder, retriever, needRewrit = False
        )

        search_results = search_results_list[0]
                                    
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
        
        # Conduct experiments solely based on search results
        answer = {
            "query":query,
            "answer":"",
            "pages":search_results,
            "judge":1 if img_path in search_results else 0
            }
                      
        return answer                
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        return   


# Retrieve query statements from the dataset, Customizable logic to adapt to different datasets
def get_queries():
    queries = []
    cqdf_pqa_path = "/root/autodl-tmp/cpdfr-data/cpdfr/hybridSearch/experiment_cpdfpqa/pdfs/cqdf_pqa.json"
    
    with open(cqdf_pqa_path, 'r', encoding='utf-8') as file:
        cqdf_pqa = json.load(file)
    samples = cqdf_pqa["samples"]
    
    for item in tqdm(samples, desc="Getting Queries", total=len(samples)):
        queries.append({"query":item["query"], "img_path": item["img_path"]})
    print("get queries done")
    return queries

async def main():    
    
    experiment_name = "hybrid_text_singleHop_cpdfpqa"
    collection_name = "cpdf_pqa_text"
    queries = get_queries()
    
    embeder=QwenEmbeder(url="https://api.siliconflow.cn/v1/embeddings")
    retriever = MilvusColbertRetriever(collection_name=collection_name, milvus_client=milvus_client)
    
    # Create semaphore and limit maximum concurrency
    semaphore = asyncio.Semaphore(10)
    
    pbar = tqdm(total=len(queries), desc="Searching Queries")
    
    async def run_with_semaphore(query):
        async with semaphore:
            result = await hybrid_text_search_answer(
                query["query"],
                query["img_path"],
                5,
                [collection_name],
                embeder,
                retriever
            )
            pbar.update(1)
            return result
            
    
    tasks = [run_with_semaphore(query) for query in queries]  
    results = await asyncio.gather(*tasks)
    
    pbar.close()
    
    total = len(results)
    acc = 0
    for item in results:
        if item["judge"] == 1:
            acc += 1
    accuracy = round(acc / total,3)
    
    eval_result = {
        "total": total,
        "acc": acc,
        "accuracy": accuracy
    }
    
    result_file = f"{experiment_name}_results.json"   
    result_data = {
        "singleHop":results,
        "eval_result":eval_result
        } 
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4) 
    print(f"check {result_file}")
    return result_file

if __name__ == "__main__":
    asyncio.run(main())
