from typing import List
import asyncio
import json
import torch
from milvus_conf import MilvusColbertRetriever, client as milvus_client
from tqdm import tqdm
import logging
from text_embeding import QwenEmbeder
from prompt import add_reflect_prompt, all_reflect_prompt, summary_prompt, DEFAULT_JUDGE_TEMPLATE, orType_check_prompt,DEFAULT_JUDGE_TEMPLATE_UNA
from transformers import AutoModel
from openai import OpenAI
import base64
from pathlib import Path
import time
import os
from io import BytesIO
from image_merge import split_list_into_5_ordered, merge_images
from PIL import Image
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELNAME="qwen3-vl-235b-a22b-instruct"
QWENAPIKEY="sk-f78b07615c8a45128d760579e6d42e1f"
DMXAPIKEY="sk-gWMA9DJgGb2QzeIa7L7nOvWeXpeESrBAB6SXVflIjnafbonl"

QWENclient = OpenAI(
    api_key=QWENAPIKEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    max_retries=3
)

DMXAPIclient = OpenAI(
    api_key=DMXAPIKEY,
    base_url="https://www.dmxapi.cn/v1",
    max_retries=3
)

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

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        # Read binary data and perform Base64 encoding
        base64_data = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_data

def transform_path(original_path):
    """
    将原始路径转换为指定的新路径格式
    """
    base_dir = "documents/"
    index = original_path.find(base_dir)
    
    if index == -1:
        raise ValueError(f"路径中未找到 '{base_dir}'")
    suffix = original_path[index + len(base_dir):]
    new_prefix = "/home/gpu/dzy/M3-CaseRAG/experiment_multiHop/MMLongBench-Doc/documents/"
    
    # 组合新路径
    new_path = new_prefix + suffix
    
    return new_path

def multi_img_in_text_search(queries: List[str], file_names: List[str], topk: int, multi_img_retriever, hybrid_text_retriever, embeder, coarse):
    query_embeddings = embeder.encode_text(
                        texts=queries,
                        task="retrieval",
                        prompt_name="query",
                        return_multivector=True,
                    )
    
    search_results_list = []
    
    for i in range(len(queries)):
        # query_np is a two-dimensional array of queries for each sentence in the query group
        query_np = query_embeddings[i].float().cpu().numpy()
        text_dense = embeder.encode_text(
                            texts=[queries[i]],
                            task="retrieval",
                            prompt_name="query"
                        )[0].float().cpu().numpy()
        
        text_query_param={
            "text_dense": text_dense,
            "text": queries[i],
            "file_names": file_names
        }

        page_paths = hybrid_text_retriever.hybrid_text_search(text_query_param, coarse)
        
        # new_page_paths = []
        # for path in page_paths:
        #     new_page_paths.append(transform_path(path))
            
        # score_results = multi_img_retriever.multi_img_in_pages_search(new_page_paths, query_np, topk)
        score_results = multi_img_retriever.multi_img_in_pages_search(page_paths, query_np, topk)
        
        search_results = []
        
        for sitem in score_results:
            search_results.append(sitem[3])  
        search_results_list.append(search_results)
    
    return search_results_list

def multi_img_in_token_search(queries: List[str], file_names: List[str], topk: int, multi_img_retriever, multi_token_retriever, embeder):
    query_embeddings = embeder.encode_text(
                        texts=queries,
                        task="retrieval",
                        prompt_name="query",
                        return_multivector=True,
                    )
    
    search_results_list = []
    
    for i in range(len(queries)):
        # query_np is a two-dimensional array of queries for each sentence in the query group
        query_np = query_embeddings[i].float().cpu().numpy()
        
        page_paths = multi_token_retriever.coarse_multi_token_search(query_np, file_names, topk)      
        score_results = multi_img_retriever.multi_img_in_pages_search(page_paths, query_np, topk)
        
        search_results = []
        
        for sitem in score_results:
            search_results.append(sitem[3])  
        search_results_list.append(search_results)
    
    return search_results_list

def multi_img_search(queries: List[str], file_names: List[str], topk: int, multi_img_retriever, embeder, coarse):
    query_embeddings = embeder.encode_text(
                        texts=queries,
                        task="retrieval",
                        prompt_name="query",
                        return_multivector=True,
                    )
    
    search_results_list = []
    
    for i in range(len(queries)):
        # query_np is a two-dimensional array of queries for each sentence in the query group
        query_np = query_embeddings[i].float().cpu().numpy()
        
        score_results = multi_img_retriever.multi_img_search(query_np, file_names, topk, coarse)
        
        search_results = []
        
        for sitem in score_results:
            search_results.append(sitem[3])  
        search_results_list.append(search_results)
    
    return search_results_list

def hybrid_text_search(queries: List[str], file_names: List[str], topk: int, hybrid_text_retriever, embeder):   
    search_results_list = []
    ans_chunks_list = []
    
    for i in range(len(queries)):
        text_dense = embeder.encode_text(
                            texts=[queries[i]],
                            task="retrieval",
                            prompt_name="query"
                        )[0].float().cpu().numpy()
        
        text_query_param={
            "text_dense": text_dense,
            "text": queries[i],
            "file_names": file_names
        }

        ans_chunks, page_paths = hybrid_text_retriever.hybrid_pure_text_search(text_query_param, topk)
        
        
        search_results_list.append(page_paths)
        ans_chunks_list.append(ans_chunks)
    return ans_chunks_list, search_results_list

def check_retriever_single_pages(pages,reference_page):
    for p in pages:
        if str(reference_page) == Path(p).stem:
            return 1
    return 0

def check_retriever_multi_pages(pages,reference_pages):
    count = 0
    for rp in reference_pages:
        for p in pages:
            if str(rp) == Path(p).stem:
                count += 1
    return round(count/len(reference_pages),2)

def check_answer(question,reference_answer,generated_answer,una = False):
    if not una:
        judge_prompt = DEFAULT_JUDGE_TEMPLATE.format(
                    query=question,
                    reference_answer=reference_answer,
                    generated_answer=generated_answer
                )
    else:
        judge_prompt = DEFAULT_JUDGE_TEMPLATE_UNA.format(
                        # query=question,
                        reference_answer=reference_answer,
                        generated_answer=generated_answer
                    )
    try:
        completion = QWENclient.chat.completions.create(
            model="qwen3-max-2025-09-23",
            messages=[
                {"role": "system", "content": "You can only answer one number"},
                {"role": "user", "content": judge_prompt},
            ],
        )  
        return (json.loads(completion.model_dump_json())["choices"][0]["message"]["content"])
    except Exception as e:
        logger.error(f"check_answer error:{e}")
        return ""


def single_img_in_text_search_answer(
    query,
    topk: int,
    multi_img_retriever,
    hybrid_text_retriever,
    multi_token_retriever,
    embeder,
    coarse = 15,
    filenames = None
):  
    if "answer" in query and "reference_answer" in query:
        if query["answer"] != "qwen api error":
            if "answer_score" in query:
                return query
            else:
                query["answer_score"] = check_answer(query["query"],query["reference_answer"],query["answer"])
                query["page_judge"] = check_retriever_multi_pages(query["pages"],query["reference_pages"])
                return query
                
    if "pages" in query:
        search_results_list = [query["pages"]]
        query["question"] = query["query"]
        query["answer"] = query["reference_answer"]
        query["evidence_pages"] = query["reference_pages"]
    else:
        if not filenames:
            filenames = [query["doc_id"]]
        questions = [query["query"]]
        # search_results_list = multi_img_in_text_search(questions, filenames, topk, multi_img_retriever, hybrid_text_retriever, embeder, coarse)
        # search_results_list = await asyncio.to_thread(
        #     multi_img_in_token_search, [query["question"]], [query["doc_id"]], topk, multi_img_retriever, multi_token_retriever, embeder
        # )
        # search_results_list = await asyncio.to_thread(
        #     multi_img_in_token_search, [query["question"]], [query["doc_id"]], topk, multi_img_retriever, multi_token_retriever, embeder
        # )
        search_results_list = multi_img_search(questions, filenames, topk, multi_img_retriever, embeder, coarse)
        
        # ans_chunks_list, search_results_list = await asyncio.to_thread(
        #     hybrid_text_search, [query["question"]], [query["doc_id"]], topk, hybrid_text_retriever, embeder
        # )
    
    answer_list = []
    for j in range(len(search_results_list)):
        # ans_chunks = ans_chunks_list[j]
        # search_results = list(dict.fromkeys(search_results_list[j]))   
        search_results = search_results_list[j]                  
        # Prepare images for the generator  
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
        
        try:
            # response  = QWENclient.chat.completions.create(
            #     model=MODELNAME, 
            #     messages=[
            #     {"role":"system","content":[{"type": "text", "text": "You need to combine the image information provided by the user's document page with your own knowledge base to answer the user's query. Your answer should be in English.Your answer needs to be as concise as possible. If the provided page cannot answer the question, simply return 'Not answerable'"}]},
            #     {
            #         "role": "user",
            #         "content": base64_images + [{"type": "text", "text": query["question"]}]
            #     }
            #     ]
            # )
            answer = {
            # "doc_id":query["doc_id"],
            "query":questions[j],
            # "answer":response.choices[0].message.content,
            # "answer":"qwen api error",
            "pages":search_results,
            # "reference_answer":query["answer"],
            "reference_page":query["page_num"],
            # "answer_score":"1",
            "page_judge_topk1":check_retriever_single_pages(search_results[:1], query["page_num"]),
            "page_judge_topk3":check_retriever_single_pages(search_results[:3], query["page_num"]),
            "page_judge_topk5":check_retriever_single_pages(search_results, query["page_num"])
            }
            
        except Exception as e:
            logger.error(f"reasoning api error: {str(e)}")
            answer = {
            # "doc_id":query["doc_id"],
            "query":questions[j],
            # "answer":"qwen api error",
            "pages":search_results,
            # "reference_answer":query["answer"],
            "reference_page":query["page_num"],
            "page_judge_topk1":check_retriever_single_pages(search_results[:1], query["page_num"]),
            "page_judge_topk3":check_retriever_single_pages(search_results[:3], query["page_num"]),
            "page_judge_topk5":check_retriever_single_pages(search_results, query["page_num"])
            }
    
        answer_list.append(answer)
        
    # By default, there is only one query for queries here 
    return answer_list[0]

def orType_check(
    node_query_or,
    user_query,
    query,
    last_conclusion,
    topk: int,
    multi_img_retriever,
    hybrid_text_retriever,
    embeder):
    search_results_list_or = multi_img_in_text_search([node_query_or],[query["doc_id"]], topk, multi_img_retriever, hybrid_text_retriever, embeder)
    search_results_or = search_results_list_or[0]
    # Prepare images for the generator  
    base64_images=[]
    try:
        for image_path in search_results_or:
            base64_str = image_to_base64(image_path) 
            base64_images.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_str}"}
            })
    except Exception as e:
        logger.error(f"准备图片时出错: {str(e)}")
        
    prompt =  orType_check_prompt + f"[user query]: {user_query}\n[last conclusion]: {last_conclusion}"
    
    response  = AIclient.chat.completions.create(
                model=MODELNAME, 
                messages=[
                {
                    "role": "user",
                    "content": base64_images + [{"type": "text", "text": prompt}]
                }
                ]
            )
    
    orType_check_ans = json.loads(response.choices[0].message.content)
    return search_results_or, orType_check_ans

def force_final_answer(user_query):
    response  = AIclient.chat.completions.create(
                model=MODELNAME, 
                messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"Based on the conclusions provided in [user query] and what you already know, think about and answer the questions in [user query].\n[user query]: {user_query}"}]
                }
                ]
            )
    return response.choices[0].message.content


async def multi_img_in_text_search_answer(
    query,
    topk: int,
    multi_img_retriever,
    hybrid_text_retriever,
    embeder
): 
    count_page = multi_img_retriever.count_page_by_file([query["doc_id"]])
    conclusion = []
    query_graph = []
    history_queries = []
    history_pages = []
    while(True):
        node_type = ""
        node_query = ""
        node_query_or = ""
        
        if conclusion == []:
            user_query = "Question: " + query["question"]
        else:
            given = "Given: "
            for c in range(len(conclusion)):
                given += str(c+1)+". "+conclusion[c]+" "
                
            user_query = given + "Question: "+query["question"]
            
        if len(history_pages) >= int(count_page):
            # 执行强制回答并结束多跳
            ans = force_final_answer(user_query)
            break
        if query_graph != []:
            for i in range(len(query_graph)):
                q = query_graph[i]
                if q["query"] not in history_queries:
                    if q["or"].strip() != "":
                        node_type = q["type"]
                        node_query = q["query"]
                        node_query_or = q["or"]
                        break       
            for i in range(query_graph):
                q = query_graph[i]
                if q["query"] not in history_queries:
                    if q["type"] == "add":
                        node_type = q["type"]
                        node_query = q["query"]
                        if q["or"].strip() != "":
                            node_query_or = q["or"]
                        else:
                            node_query_or = ""
                        break
                    
            if node_query == "":
                # 执行节点全遍历后的总体反思
                node_query = history_queries[-1]
        else:
            node_query = query["question"]
                
        search_results_list = await asyncio.to_thread(
            multi_img_in_text_search, [node_query], [query["doc_id"]], topk, multi_img_retriever, hybrid_text_retriever, embeder
        )
        search_results = search_results_list[0]
        history_queries.append(node_query)
        history_pages = list(set(history_pages) | set(search_results))  
                          
        # Prepare images for the generator  
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
            if node_type == "":
                prompt = all_reflect_prompt + f"[user query]:{user_query}\n[history queries]:{history_queries}"
            elif node_type == "add":
                prompt = add_reflect_prompt + f"[user query]:{user_query}\n[history queries]:{history_queries}\n[generated queries]:{query_graph}"
            elif node_type == "sub":
                prompt = summary_prompt + f"[user query]:{user_query}"
            else:
                logger.error("node_type error")    
                
            response  = QWENclient.chat.completions.create(
                model=MODELNAME, 
                messages=[
                {
                    "role": "user",
                    "content": base64_images + [{"type": "text", "text": prompt}]
                }
                ]
            )
            res = json.loads(response.choices[0].message.content) 
            
            print(res)    
            if res["answer"].strip() == "":
                if "queries" in res:
                    query_graph = res["queries"]
                if res["conclusion"].strip() != "":
                    if node_query_or == "":
                        conclusion.append(res["conclusion"])
                    else:
                        search_results_or, orType_check_ans = orType_check(
                            node_query_or,
                            user_query,
                            query,
                            res["conclusion"],
                            topk,
                            multi_img_retriever,
                            hybrid_text_retriever,
                            embeder)
                        
                        print(orType_check_ans)
                        history_queries.append(node_query_or)
                        history_pages = list(set(history_pages) | set(search_results_or)) 
                        
                        if orType_check_ans["answer"].strip() == "":
                            if orType_check_ans["conclusion"].strip() != "":
                                conclusion.append(orType_check_ans["conclusion"])
                        else:
                            ans = orType_check_ans["answer"]
                            break
            else:
                ans = res["answer"]
                break

        except Exception as e:
            logger.error(f"MLLM error: {str(e)}")
            answer = {
            "doc_id":query["doc_id"],
            "query":query["question"],
            "answer":"qwen api error",
            "pages":[],
            "reference_answer":query["answer"],
            "reference_pages":query["evidence_pages"]
            }
            return answer


    answer = {
        "doc_id":query["doc_id"],
        "query":query["question"],
        "answer":ans,
        "pages":history_pages,
        "reference_answer":query["answer"],
        "reference_pages":query["evidence_pages"],
        "answer_score":check_answer(query["question"],query["answer"],ans),
        "page_judge":check_retriever_multi_pages(search_results,query["evidence_pages"])
        }
          
    return answer       


def LVLM_search_answer(
    queries
):  
    ans_list = []
    doc_id = queries[0]["doc_id"] 
    pages_dir = f"/home/gpu/dzy/M3-CaseRAG/experiment_multiHop/MMLongBench-Doc/documents/{doc_id}" 
    pages = []
    pageNum = 1
    while(1):
        image_path = pages_dir+f"/{pageNum}.png"
        if os.path.exists(image_path):
            pages.append(image_path)
            pageNum += 1
        else:
            break
    sublists = split_list_into_5_ordered(pages)

    base64_images=[]
    try:
        ImgList = []
        for i in range(len(sublists)):
            item = sublists[i]
            if len(item) < 1:
                continue
            ImgList.append(merge_images(item,f"{i}.jpg",3879731))
            
        for image_path in ImgList:
            base64_str = image_to_base64(image_path) 
            base64_images.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_str}"}
            })
    except Exception as e:
        logger.error(f"准备图片时出错: {str(e)}")
        base64_images=[]
        
    for query in queries:
        if "answer" in query and "reference_answer" in query:
            if query["answer"] != "qwen api error":
                if "answer_score" in query:
                    return query
                else:
                    query["answer_score"] = check_answer(query["query"],query["reference_answer"],query["answer"])
                    return query

        if base64_images == []:
            answer = {
            "doc_id":query["doc_id"],
            "query":query["question"],
            "answer":"qwen api error",
            "reference_answer":query["answer"]
            }
        else:
            try:
                response  = QWENclient.chat.completions.create(
                    model=MODELNAME, 
                    messages=[
                    {"role":"system","content":[{"type": "text", "text": "You need to combine the image information provided by the user's document page with your own knowledge base to answer the user's query. Your answer should be in English.Your answer needs to be as concise as possible. If the provided page cannot answer the question, simply return 'Not answerable'"}]},
                    {
                        "role": "user",
                        "content": base64_images + [{"type": "text", "text": query["question"]}]
                    }
                    ]
                )
                res = response.choices[0].message.content
                if len(query["evidence_pages"]) == 0:
                    answer_score = check_answer(query["question"], query["answer"], res, una=True)
                else:
                    answer_score = check_answer(query["question"], query["answer"], res)
                answer = {
                "doc_id":query["doc_id"],
                "query":query["question"],
                "answer":res,
                "reference_answer":query["answer"],
                "answer_score":answer_score
                }
                
            except Exception as e:
                logger.error(f"reasoning api error: {str(e)}")
                answer = {
                "doc_id":query["doc_id"],
                "query":query["question"],
                "answer":"qwen api error",
                "reference_answer":query["answer"]
                }
            
        ans_list.append(answer)
    
    for img_path in ImgList:
        if os.path.exists(img_path):
            os.remove(img_path)
        else:
            print(f"文件不存在，跳过: {img_path}")   
    return ans_list

def group_by_doc_id(input_list):
    grouped = defaultdict(list)
    for item in input_list:
        doc_id = item["doc_id"]
        grouped[doc_id].append(item)
    return list(grouped.values())
        
# Retrieve query statements from the dataset, Customizable logic to adapt to different datasets
def get_queries():
    with open("/home/gpu/dzy/M3-CaseRAG/experiment_singleHop/vidore/docvqa/docvqa.json", 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def main():    
    experiment_name = "patch_to_patch_docvqa_5"
    multi_img_collection_name = "docvqa"
    multi_token_collection_name = "docvqa_colbert_text"
    hybrid_text_collection_name = "docvqa_text"
    queries = get_queries()
    print(f"need process {len(queries)} queries")
    
    embeder = init_model()
    # embeder = ""
    multi_img_retriever = MilvusColbertRetriever(collection_name=multi_img_collection_name, milvus_client=milvus_client)
    
    multi_token_retriever = MilvusColbertRetriever(collection_name=multi_token_collection_name, milvus_client=milvus_client)
    
    hybrid_text_retriever = MilvusColbertRetriever(collection_name=hybrid_text_collection_name, milvus_client=milvus_client)
    # hybrid_text_retriever = ""
    
    # Create semaphore and limit maximum concurrency
    # semaphore = asyncio.Semaphore(1)
    
    # pbar = tqdm(total=len(queries), desc="Searching Queries")
    # async def run_with_semaphore(query):
    #     async with semaphore:
    #         result = single_img_in_text_search_answer(
    #             query,
    #             5,
    #             multi_img_retriever,
    #             hybrid_text_retriever,
    #             multi_token_retriever,
    #             embeder,
    #             coarse=30,
    #             filenames=["tatdqa"]
    #         )
    #         pbar.update(1)
    #         return result   
    # start_time = time.time()
    # tasks = [run_with_semaphore(query) for query in queries]  
    # results = await asyncio.gather(*tasks)
    # end_time = time.time()
    # pbar.close()
    
    results = []
    start_time = time.time()
    for query in tqdm(queries, desc="Searching Queries"):
        result = single_img_in_text_search_answer(
                query,
                5,
                multi_img_retriever,
                hybrid_text_retriever,
                multi_token_retriever,
                embeder,
                coarse=5,
                filenames=["docvqa"]
            )
        results.append(result)
    end_time = time.time()
    # 测试模型（无检索全输入）
    # qs = group_by_doc_id(queries)
    # results = []
    # for q in tqdm(qs, desc="Searching Queries"):
    #     ans_list = LVLM_search_answer(q)
    #     results.extend(ans_list)
    
    total = 0
    # answer_acc = 0
    page_acc_topk1 = 0
    page_acc_topk3 = 0
    page_acc_topk5 = 0
    for result in results:
        # if "answer_score" in result:
        if "page_judge_topk1" in result:
            total += 1
            if result["page_judge_topk1"] == 1:
                page_acc_topk1 += 1
            if result["page_judge_topk3"] == 1:
                page_acc_topk3 += 1
            if result["page_judge_topk5"] == 1:
                page_acc_topk5 += 1
            # if result["answer_score"] == "4" or result["answer_score"] == "5":
            #     answer_acc += 1
            # elif result["answer_score"] == "1" or result["answer_score"] == "2" or result["answer_score"] == "3":
            #     continue
            # else:
            #     logger.error("LLM's grading format for answers is incorrect")
            #     continue
                
    result_file = f"./vidore/docvqa/coarse5/{experiment_name}_results.json"   
    result_data = {
        # "model":MODELNAME,
        "singleHop":results,
        "eval_results":{
            "total":total,
            # "answer_acc":answer_acc,
            "page_acc_topk1":page_acc_topk1,
            "page_acc_topk3":page_acc_topk3,
            "page_acc_topk5":page_acc_topk5,
            "time":end_time-start_time
        }
        } 
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4) 
    print(f"check {result_file}")
    return result_file

if __name__ == "__main__":
    main()
