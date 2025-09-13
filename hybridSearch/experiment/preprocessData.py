from pathlib import Path
import logging
import os
import torch
from colpali_process import processImg
from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device,ListDataset
import time
from milvus_conf_hybrid import MilvusColbertRetriever, client
from fastapi_server import getTextByPath
from text_embeding import QwenEmbeder
import json

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("uvicorn.access").propagate = True
logging.getLogger("uvicorn.error").propagate = True

 # 获取设备
device = get_torch_device("cuda")
logger.info(f"Using device: {device}")

# 模型路径配置
model_name = "/home/gpu/milvus/backend/colpali/modelcache/models--vidore--colpali-v1.2/snapshots/6b89bc63c16809af4d111bfe412e2ac6bc3c9451"
cachedir = "/home/gpu/milvus/backend/colpali/modelcache/"

# 加载模型
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


def preProcess_milvus(parse_pdf_path: Path,collection_name,pdfId):
    pages_path = Path(str(parse_pdf_path)+f"/pages")
    caption_text_list_path = Path(str(parse_pdf_path)+f"/caption_text_list.json")
    
    # if not parse_pdf_path.exists():
    #     logger.info("不存在文档解析处理结果")
    #     return
    
    # # 检查页面文件夹是否存在
    # if not Path(str(parse_pdf_path)+f"/pages").exists():
    #     logger.info("页面文件夹不存在")
    #     return
    
    # if not Path(str(parse_pdf_path)+f"/caption_text_list.json").exists():
    #     logger.info("caption_text_list不存在")
    #     return
  
    #存入milvus
    ImagePaths = [os.path.join(pages_path, name) for name in os.listdir(pages_path)]
    #获取图片向量组
    logger.info("获取图片向量组和单向量...")
    ds = processImg(ImagePaths,model,processor,device)
    # single_img_vecs = processImg_single(ImagePaths,model_2)
    
    # 初始化Milvus
    if(client.has_collection(collection_name=collection_name)):
        # logger.info("已存在该向量数据库")
        retriever = MilvusColbertRetriever(collection_name=collection_name, milvus_client=client)
    else:
        retriever = MilvusColbertRetriever(collection_name=collection_name, milvus_client=client)
        retriever.create_collection()
        retriever.create_index()
        
    # if(client_img.has_collection(collection_name=username+"_img")):
    #     logger.info("用户已存在向量数据库(纯图像RAG版本)")
    #     retriever_img = MilvusColbertRetriever_img(collection_name=username+"_img", milvus_client=client_img)
    # else:
    #     retriever_img = MilvusColbertRetriever_img(collection_name=username+"_img", milvus_client=client_img)
    #     retriever_img.create_collection()
    #     retriever_img.create_index()
    
    # logger.info("开始写入向量数据库...")    
    for i, (Imgpath, embedding) in enumerate(zip(ImagePaths, ds)):
        text = getTextByPath(Imgpath,caption_text_list_path)
        # 判断 text 是否为空（None 或空字符串）
        if text is None or text.strip() == "":
            text_dense_value = [0.0] * 1024
        else:
            text_dense_value = embeder.getTextEmbeddings(text)
        data = {
            "colbert_vecs": embedding.float().cpu().numpy(),
            "doc_id": i,
            "filepath": Imgpath,
            "text": text,
            "customName": pdfId,
            "text_dense": text_dense_value
            }
        retriever.insert(data)
        
        # data_img = {
        #     "multiple_image_dense": embedding.float().cpu().numpy(),
        #     "doc_id": i,
        #     "filepath": Imgpath,
        #     "single_image_dense":single_img_embedding[0],
        #     "customName": customName
        #     }
        # retriever_img.insert(data_img)

    # return {"message": "RAG知识库搭建成功"}

def main():
    collection_name = "vidoseek"
    with open("/home/gpu/milvus/backend/colpali/ViDoSeek/subfolders.json", 'r', encoding='utf-8') as file:
        subfolders = json.load(file)
        
    #TODO：最后一次写入，2025/9/13 2:34 第22份pdf已写入，问题是写入过快或者其他原因导致容器异常
    subfolders_list = subfolders["subfolders"]
    for i in range(len(subfolders_list)):
        # 手动分批次存入
        if i <= 21:
            print(f"跳过第{i+1}份pdf")
            continue
        else: 
            preProcess_milvus(subfolders_list[i]["path"],collection_name,subfolders_list[i]["pdfId"])
            print(f"写入第{i+1}份pdf")  
    return
    

if __name__ == "__main__":
    main()