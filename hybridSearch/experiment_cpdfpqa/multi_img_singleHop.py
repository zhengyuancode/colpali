from typing import List
from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device, ListDataset
import asyncio
import json
import torch
from torch.utils.data import DataLoader
import os
from milvus_conf import MilvusColbertRetriever, client as milvus_client
from datasets import load_dataset,load_from_disk
from tqdm import tqdm
from zhipuai import ZhipuAI
import logging

ZHIPUAPIKEY="f890fa44ea384a6baab00c725701a04b.1h0evvTQSAZALIp0"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_model():
    device = get_torch_device("cuda")
    model_name = "vidore/colpali-v1.3"
    model = ColPali.from_pretrained(
        model_name,
        cache_dir="/root/autodl-tmp/cpdfr-data/modelcache/huggingface/hub",
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=True,
        use_safetensors=True
    ).eval()
    processor = ColPaliProcessor.from_pretrained(model_name)
    return model, processor, device

def process_querys(queries: List[str],myprocessor,mymodel,mydevice) -> List[torch.Tensor]:
    """Process queries and generate embedding vectors"""
    dataloader = DataLoader(
        dataset=ListDataset[str](queries),
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: myprocessor.process_queries(x),
    )
    qs: List[torch.Tensor] = []
    for batch_query in dataloader:
        with torch.no_grad():
            batch_query = {k: v.to(mymodel.device) for k, v in batch_query.items()}
            embeddings_query = mymodel(**batch_query)
        qs.extend(list(torch.unbind(embeddings_query.to(mydevice))))
    return qs

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

def multi_img_search(retriever ,queries: List[str], file_names: List[str], topk: int, mymodel, myprocessor, mydevice, needRewrit = False):
    if needRewrit:   
        query_embeddings = process_querys(query_rewrit(queries, "english"),myprocessor,mymodel,mydevice)
    else:
        query_embeddings = process_querys(queries,myprocessor,mymodel,mydevice)
    
    search_results_list = []
    
    for query_emb in query_embeddings:
        # query_np is a two-dimensional array of queries for each sentence in the query group
        query_np = query_emb.float().cpu().numpy()
        
        score_results = retriever.multi_img_search(query_np, file_names, topk)
        
        search_results = []
        
        for sitem in score_results:
            search_results.append(sitem[3])  
        search_results_list.append(search_results)
    
    return search_results_list


# By default, only one query is entered here
async def multi_img_search_answer(
    query,
    img_path,
    topk: int,
    file_names,
    mymodel, 
    myprocessor, 
    mydevice,
    retriever
): 
    try:
        search_results_list = await asyncio.to_thread(
            multi_img_search, retriever, [query], file_names, topk, mymodel, myprocessor, mydevice, needRewrit = False
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
    dataset_json_path = "/root/autodl-tmp/cpdfr-data/cpdfr/hybridSearch/experment_vidore/vidore_data/tatdqa_test/documents/tatdqa.json"
    
    with open(dataset_json_path, 'r', encoding='utf-8') as file:
        dataset_json = json.load(file)
    samples = dataset_json["examples"]
    
    for item in tqdm(samples, desc="Getting Queries", total=len(samples)):
        queries.append({"query":item["query"], "img_path": item["image"]})
    print("get queries done")
    return queries

async def main():    
    
    experiment_name = "multi_img_singleHop_vidore_tatdqa"
    collection_name = "vidore_tatdqa"
    file_names = ["vidore_tatdqa"]
    queries = get_queries()
    
    mymodel, myprocessor, mydevice = init_model()
    retriever = MilvusColbertRetriever(collection_name=collection_name, milvus_client=milvus_client)
    
    # Create semaphore and limit maximum concurrency
    semaphore = asyncio.Semaphore(10)
    
    pbar = tqdm(total=len(queries), desc="Searching Queries")
    
    async def run_with_semaphore(query):
        async with semaphore:
            result = await multi_img_search_answer(
                query["query"],
                query["img_path"],
                5,
                file_names,
                mymodel, 
                myprocessor, 
                mydevice,
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
