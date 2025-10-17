import json
import os
from milvus_conf_hybrid import MilvusColbertRetriever, client
from pathlib import Path
from milvus_conf_img_hybrid import MilvusColbertRetriever as MilvusColbertRetriever_img,client as client_img
from openai import OpenAI
from tqdm import tqdm
from zhipuai import ZhipuAI
from transformers import pipeline
import torch
import base64


def getSingleHopExamples(orgin_path,output_path):
    with open(orgin_path, 'r', encoding='utf-8') as file1:
        data = json.load(file1)
    examples=data["examples"]
    singlehopexamples={"examples":[]}
    for item in examples:
        if item["meta_info"]["query_type"] == "single_hop":
            singlehopexamples["examples"].append(item)
    with open(output_path, 'w', encoding='utf-8') as file2:
        json.dump(singlehopexamples, file2, indent=4, ensure_ascii=False)
        
def getParseList():
    parseDir = "/home/gpu/milvus/backend/colpali/ViDoSeek/minerU_pdf"
    subfolders = {"subfolders":[]}
    # 遍历 parseDir 下的所有条目
    for name in os.listdir(parseDir):
        path = os.path.join(parseDir, name)
        if os.path.isdir(path):  # 判断是否为目录
            subfolders["subfolders"].append({"path":path,"pdfId":name})
    with open("subfolders.json", 'w', encoding='utf-8') as file:
        json.dump(subfolders, file, indent=4, ensure_ascii=False)

def setId():
    with open("subfolders.json", 'r', encoding='utf-8') as file1:
        data = json.load(file1)
    subfolders=data["subfolders"]
    for i in range(len(subfolders)):
        subfolders[i]["sort"] = i+1
    new_subfolders={"subfolders":subfolders}
    with open("subfolders.json", 'w', encoding='utf-8') as file:
        json.dump(new_subfolders, file, indent=4, ensure_ascii=False)
        
def createVidoseekCollection(collection_name):
    # 初始化Milvus
    if(client.has_collection(collection_name=collection_name)):
        # logger.info("已存在该向量数据库")
        print(f"已存在{collection_name}向量数据库")
    else:
        retriever = MilvusColbertRetriever(collection_name=collection_name, milvus_client=client)
        retriever.create_collection()
        retriever.create_index()
        print(f"已创建{collection_name}向量数据库")

def check_rewrite_caption_text(path):
    pipe = pipeline(
        "image-text-to-text",
        model="llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float16,
        device_map="auto"
        )
    
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    rewrite_data = []
    for item in tqdm(data,desc="rewriting",total=len(data)):  
        if len(item) <= 3000: 
            messages = [
                {
                "role": "user",
                "content": [
                    # {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"},
                    {"type": "text", "text": f"The text paragraph provided by the user may have missing spaces, poor language, etc., which are not suitable for vector embedding. You need to improve this paragraph and return it, try not to change the original words as much as possible, and keep the original vocabulary as much as possible:\n[example]:\n5.ExceptionalitemsTheimagedisplaysafinancialtablecomparingspecificcostsandadjustmentsfortheyears2019and2018,inmillionsofdollars. \n[answer]:\n5.Exceptional items  The image displays a financial table comparing specific costs and adjustments for the years 2019 and 2018, in millions of dollars.\n-----------------------------\n[user input]:\n{item}"},
                    ],
                },
            ]

            out = pipe(text=messages,max_new_tokens=1024)
            rewrite = out[0]["generated_text"][1]["content"]
        else:
            # print(item)
            # print("-----------------------------------------")
            rewrite = ""
            max_len = 3000
            segments = []
            start = 0
            n = len(item)
            while start < n:
                # 当前段结束位置：从 start + max_len 往前找断点
                end = min(start + max_len, n)

                # 如果已经到结尾，直接切最后一段
                if end == n:
                    segments.append(item[start:end])
                    break

                # 在 [start, end] 范围内从右往左找最后一个合法断点
                # 断点必须是 '.', ',', ';' 且后面不能是字母（避免把缩写如 "Mr." 中间断开？但这里我们信任标点就是断句）
                break_chars = {'.', ',', ';'}

                # 从 end-1 往前找
                found = -1
                for i in range(end - 1, start - 1, -1):
                    if item[i] in break_chars:
                        found = i + 1  # 断点在 i 之后（包含该标点）
                        break

                # 如果没找到合法断点，迫不得已就在 max_len 处断开（避免无限循环）
                if found == -1:
                    found = end

                segments.append(item[start:found])
                start = found  # 下一段从此开始
                
            for seg in segments:
                out = pipe(
                    text=[
                            {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"The text paragraph provided by the user may have missing spaces, poor language, etc., which are not suitable for vector embedding. You need to improve this paragraph and return it, try not to change the original words as much as possible, and keep the original vocabulary as much as possible:\n[example]:\n5.ExceptionalitemsTheimagedisplaysafinancialtablecomparingspecificcostsandadjustmentsfortheyears2019and2018,inmillionsofdollars. \n[answer]:\n5.Exceptional items  The image displays a financial table comparing specific costs and adjustments for the years 2019 and 2018, in millions of dollars.\n-----------------------------\n[user input]:\n{seg}"},
                                ],
                            },
                        ],
                    max_new_tokens=1024
                    )
                rewrite +=  out[0]["generated_text"][1]["content"]
            # print(rewrite)
        rewrite_data.append(rewrite)    
        
    with open(path, 'w', encoding='utf-8') as file2:
        json.dump(rewrite_data, file2, indent=2, ensure_ascii=False)
    print("caption text 重写成功")
    
    
    
    
    
# check_rewrite_caption_text("/root/autodl-tmp/cpdfr-data/cpdfr/hybridSearch/experment_vidore/vidore_data/tatdqa_test/documents/parse/vidore_tatdqa/caption_text_list.json")

# getSingleHopExamples("experiment/vidoseek.json","experiment/vidoseek_singleHop.json")
# getParseList()
# setId()

# createVidoseekCollection("vidore_tatdqa")
        

# 上传本地的向量数据到minio
# retriever = MilvusColbertRetriever(collection_name="MMLongDoc", milvus_client=client)
# remote_files = retriever.bulk_LocalData_upload("/home/gpu/milvus/backend/colpali/ViDoSeek/bulkInsert","c494711a-1dbe-43f2-9d18-994eb651957d")

#将minio的向量数据插入milvus,minio上的数据，登录localhost:9001查看数据存储路径
# 也可以观察控制台输出的路径和数量
# dir = "vidore_docvqa/7ad44368-2991-48c8-a383-06e49d8fcba5/"
# files = []
# for i in range(5):
#     files.append([dir+str(i+1)+".parquet"])
# retriever.bulk_minio_insert_milvus("vidore_docvqa",files)


# #检查导入进度
# jobId="460918093663852092"
# resp = retriever.search_import_progress(jobId)
# print(len(resp["data"]["details"]))
