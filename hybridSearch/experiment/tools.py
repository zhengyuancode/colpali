import json
import os
from milvus_conf_hybrid import MilvusColbertRetriever, client
from pathlib import Path
from milvus_conf_img_hybrid import MilvusColbertRetriever as MilvusColbertRetriever_img,client as client_img

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
        
def createVidoseekCollection():
    # 初始化Milvus
    if(client.has_collection(collection_name="vidore_docvqa_img")):
        # logger.info("已存在该向量数据库")
        print("已存在vidore_docvqa_img向量数据库")
    else:
        retriever = MilvusColbertRetriever_img(collection_name="vidore_docvqa_img", milvus_client=client)
        retriever.create_collection()
        retriever.create_index()
        print("已创建vidore_docvqa_img向量数据库")
        

# getSingleHopExamples("experiment/vidoseek.json","experiment/vidoseek_singleHop.json")
# getParseList()
# setId()

# createVidoseekCollection()
        

# 上传本地的向量数据到minio
retriever = MilvusColbertRetriever(collection_name="vidoseek", milvus_client=client)
# remote_files = retriever.bulk_LocalData_upload("/home/gpu/milvus/backend/colpali/ViDoSeek/bulkInsert","c494711a-1dbe-43f2-9d18-994eb651957d")

#将minio的向量数据插入milvus,minio上的数据，登录localhost:9001查看数据存储路径
# 也可以观察控制台输出的路径和数量
# dir = "vidore_docvqa/7ad44368-2991-48c8-a383-06e49d8fcba5/"
# files = []
# for i in range(5):
#     files.append([dir+str(i+1)+".parquet"])
# retriever.bulk_minio_insert_milvus("vidore_docvqa",files)


# #检查导入进度
jobId="460918093663852092"
resp = retriever.search_import_progress(jobId)
print(len(resp["data"]["details"]))



# 导入结果检查
# with open("/home/gpu/milvus/backend/colpali/ViDoSeek/vidoseek_import_record.json", 'r', encoding='utf-8') as file:
#         data = json.load(file)


# totalRows = 0
# pdfCount = 0        
# details = data["data"]["details"]
# for item in details:
#     if item["progress"] == 100 and item["state"] == "Completed":
#         if item["totalRows"] == item["importedRows"]:
#             totalRows += item["totalRows"]
#             pdfCount += 1
# print(totalRows)
# print(pdfCount)
