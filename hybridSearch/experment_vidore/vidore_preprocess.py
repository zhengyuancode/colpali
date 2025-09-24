from datasets import load_dataset,load_from_disk
import os
from tqdm import tqdm
from typing import cast
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device
import torch
from PIL import Image
from transformers import AutoModel
from pathlib import Path
import logging
from colpali_process import processImg,processImg_single
from milvus_conf_hybrid import MilvusColbertRetriever, client
from text_embeding import QwenEmbeder
from milvus_conf_img_hybrid import MilvusColbertRetriever as MilvusColbertRetriever_img,client as client_img
from pymilvus import Collection,connections
from pymilvus.bulk_writer import LocalBulkWriter, BulkFileType,RemoteBulkWriter
import json

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("uvicorn.access").propagate = True
logging.getLogger("uvicorn.error").propagate = True



# model_name = "vidore/colpali-v1.3"
cachedir = "/root/autodl-tmp/cpdfr-data/modelcache/huggingface/hub"
model_name = "/root/autodl-tmp/cpdfr-data/modelcache/huggingface/hub/models--vidore--colpali-v1.3/snapshots/1b5c8929330df1a66de441a9b5409a878f0de5b0"
model_name_2 = "/root/autodl-tmp/cpdfr-data/modelcache/huggingface/hub/models--jinaai--jina-embeddings-v4/snapshots/737fa5c46f0262ceba4a462ffa1c5bcf01da416f"

device = get_torch_device("cuda")
model = ColPali.from_pretrained(
    model_name,
    cache_dir=cachedir,
    torch_dtype=torch.bfloat16,
    device_map=device,
    local_files_only=True,
    use_safetensors=True
).eval()
processor = ColPaliProcessor.from_pretrained(model_name)
embeder=QwenEmbeder(url="https://api.siliconflow.cn/v1/embeddings")

# model_2 = AutoModel.from_pretrained(
#         model_name_2,
#         trust_remote_code=True,
#         torch_dtype=torch.float16,
#         cache_dir=cachedir,                # 指定缓存路径
#         local_files_only=True,              # 强制离线加载
#     )
# model_2.to("cuda")


# Login using e.g. `huggingface-cli login` to access this dataset
# ds = load_dataset("vidore/tatdqa_test",
#                    split="test")
# ds.save_to_disk("./vidore_data/tatdqa_test")

# ds = load_from_disk("./vidore_data/docvqa_test_subsampled")

documents_folder_path = './vidore_data/tatdqa_test/documents'
if not os.path.exists(documents_folder_path):
    os.makedirs(documents_folder_path)

# docvqa
# def getDocumentImg(documents_folder_path,ds):
#     for item in tqdm(ds, desc="save img", total=len(ds)):
#         docement_name = str(item["questionId"])+"_"+str(item["docId"])+"_"+item["image_filename"]+"_"+item["page"]+".png"
#         img = item['image']
#         img.save(documents_folder_path+"/"+docement_name)
#     print("get img done")
    
# tatdqa
def getDocumentImg(documents_folder_path,ds):
    tatdqa = {"examples":[]}
    for i in tqdm(range(len(ds)), desc="save img", total=len(ds)):
        document_name = Path(ds[i]["image_filename"]).stem + "_" + ds[i]["page"] + ".png"
        document_path = documents_folder_path+"/"+document_name
        if not Path(document_path).exists():
            img = ds[i]['image']
            img.save(str(document_path))
        tatdqa["examples"].append({
            "query":ds[i]["query"],
            "image":document_path
        })
    with open(documents_folder_path+"/tatdqa.json", 'w', encoding='utf-8') as f:
        json.dump(tatdqa, f, ensure_ascii=False, indent=4) 
    print("get img done")
    
def preProcess_milvus_vidore(pages_path: Path,collection_name,customName,writer):
    parse_path = Path(str(pages_path)+"/parse")
    mapping_path = Path(str(pages_path)+"/pdf/page_to_image.json")
    caption_text_list_path = Path(str(parse_path)+f"/{customName}/caption_text_list.json")
    
    if not parse_path.exists():
        logger.info("不存在文档解析处理结果")
        return
    
    if not Path(str(parse_path)+f"/{customName}/caption_text_list.json").exists():
        logger.info("caption_text_list不存在")
        return
    
    
    #存入milvus
    
    # 支持的图片格式
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
    ImagePaths = [f for f in pages_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not ImagePaths:
        print("错误：未在指定文件夹中找到支持的图片文件。")
        return
    
    # 排序（按文件名字母序，可根据需要改为时间等）
    ImagePaths.sort(key=lambda x: x.name)
    
    with open(mapping_path, 'r', encoding='utf-8') as file:
        mapping = json.load(file)
    with open(caption_text_list_path, 'r', encoding='utf-8') as f:
        text_list = json.load(f)
    
    #获取图片向量组
    logger.info("获取图片向量组和单向量...")
    ds = processImg(ImagePaths,model,processor,device)
    # single_img_vecs = processImg_single(ImagePaths,model_2)
    
    # 初始化Milvus
    if(client.has_collection(collection_name=collection_name)):
        # logger.info("已存在该向量数据库")
        print(f"已存在{collection_name}向量数据库")
        retriever = MilvusColbertRetriever(collection_name=collection_name, milvus_client=client)
    else:
        retriever = MilvusColbertRetriever(collection_name=collection_name, milvus_client=client)
        retriever.create_collection()
        retriever.create_index()
        print(f"已创建{collection_name}向量数据库")
    # if(client_img.has_collection(collection_name=collection_name)):
    #     logger.info("用户已存在向量数据库(纯图像RAG版本)")
    #     retriever_img = MilvusColbertRetriever_img(collection_name=collection_name, milvus_client=client_img)
    # else:
    #     retriever_img = MilvusColbertRetriever_img(collection_name=collection_name, milvus_client=client_img)
    #     retriever_img.create_collection()
    #     retriever_img.create_index()
    
    # logger.info("开始写入向量数据库...") 
    for i, (Imgpath, embedding) in enumerate(tqdm(zip(ImagePaths, ds),desc="Making BulkData", total=len(ImagePaths))):
        pageNum = mapping[os.path.basename(Imgpath)]
        text = text_list[int(pageNum)-1]
        # 判断 text 是否为空（None 或空字符串）
        if text is None or text.strip() == "":
            text_dense_value = [0.0] * 1024
        else:
            text_dense_value = embeder.getTextEmbeddings(text)
        data = {
            "colbert_vecs": embedding.float().cpu().numpy(),
            "doc_id": i,
            "filepath": str(Imgpath),
            "text": text,
            "customName": customName,
            "text_dense": text_dense_value
            }
        # retriever.insert(data)
        retriever.bulk_insert_prepare(data,writer)
        if (i+1) % 100 == 0:
            retriever.bulk_prepare_commit(writer)  
    retriever.bulk_prepare_commit(writer) 
       
    # for i, (Imgpath, embedding, single_img_embedding) in enumerate(tqdm(zip(ImagePaths, ds, single_img_vecs),desc="Making BulkData", total=len(ImagePaths))):  
        # data_img = {
        #     "multiple_image_dense": embedding.float().cpu().numpy(),
        #     "doc_id": i,
        #     "filepath": Imgpath,
        #     "single_image_dense":single_img_embedding.float().cpu().numpy(),
        #     "customName": customName
        #     }
        # retriever_img.insert(data_img)
    #     retriever_img.bulk_insert_prepare(data_img,writer)
    #     if (i+1) % 100 == 0:
    #         retriever_img.bulk_prepare_commit(writer)  
    # retriever_img.bulk_prepare_commit(writer)
    
    # return {"message": "RAG知识库搭建成功"}

def main():
    collection_name = "vidore_tatdqa"
    
    connections.connect(
        uri="http://127.0.0.1:19530", 
        token="root:Milvus"
    )

    collection = Collection(
        name=collection_name,
        using="default"
    )

    schema = collection.schema
    
    #default
    ACCESS_KEY="minioadmin"
    SECRET_KEY="minioadmin"
    BUCKET_NAME="a-bucket"
    
    conn = RemoteBulkWriter.S3ConnectParam(
        endpoint="localhost:9000", # the default MinIO service started along with Milvus
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        bucket_name=BUCKET_NAME,
        secure=False
    )

    # writer = LocalBulkWriter(
    #     schema=schema,
    #     local_path='./bulkInsert',
    #     segment_size=512 * 1024 * 1024, # Default value
    #     file_type=BulkFileType.PARQUET
    # )
    writer = RemoteBulkWriter(
        schema=schema,
        remote_path="/vidore_tatdqa",
        connect_param=conn,
        file_type=BulkFileType.PARQUET
    )

    
 
    preProcess_milvus_vidore(Path(documents_folder_path),collection_name,collection_name,writer)
    

if __name__ == "__main__":
    main()
	#  getDocumentImg(documents_folder_path,ds)