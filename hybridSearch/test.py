from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device
from torch.utils.data import DataLoader
import torch
from typing import List, cast
from tqdm import tqdm
from PIL import Image
import os
from milvus_conf_hybrid import MilvusColbertRetriever, client
import time
import logging
import re
import json
from text_embeding import QwenEmbeder,Embedding_client
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = get_torch_device("cuda")
# model_name = "vidore/colpali-v1.2"

# 若已有模型文件直接指示到模型文件所在目录位置
colpali_model_name="/home/gpu/milvus/backend/colpali/modelcache/models--vidore--colpali-v1.2/snapshots/6b89bc63c16809af4d111bfe412e2ac6bc3c9451"

# colpali模型需要协同其他模型，指示第一次下载后的总缓存位置
cachedir="/home/gpu/milvus/backend/colpali/modelcache/"

# 只获取图片路径列表，不实际加载图片
image_dir = "./pages"
filepaths = [os.path.join(image_dir, name) for name in os.listdir(image_dir)]
logger.info(f"Found {len(filepaths)} images in directory")



# 加载模型并显示加载时间
logger.info(f"Loading model: {colpali_model_name}")
model_load_start = time.time()
colpali_model = ColPali.from_pretrained(
    colpali_model_name,
    cache_dir=cachedir,
    torch_dtype=torch.bfloat16,
    device_map=device,
    local_files_only=True,
    use_safetensors=True
).eval()
model_load_time = time.time() - model_load_start
logger.info(f"Model loaded in {model_load_time:.2f} seconds")

processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(colpali_model_name))

def processImageQuery(queries):
    dataloader = DataLoader(
        dataset=ListDataset[str](queries),
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: processor.process_queries(x),
    )

    qs: List[torch.Tensor] = []
    for batch_query in dataloader:
        with torch.no_grad():
            batch_query = {k: v.to(colpali_model.device) for k, v in batch_query.items()}
            embeddings_query = colpali_model(**batch_query)
        qs.extend(list(torch.unbind(embeddings_query.to(device))))
    
    return qs

#用于将pdf的每一页图像存入向量库

def processImg(filepaths: List[str],Mymodel,Myprocessor,Mydevice):
    # 自定义collate函数，按需加载图片
    def collate_fn(path_batch: List[str]):
        images = []
        for path in path_batch:
            try:
                img = Image.open(path)
                images.append(img)
            except Exception as e:
                print(f"Error loading {path}: {str(e)}")
                # 添加空图像占位符
                images.append(Image.new('RGB', (224, 224)))
        return Myprocessor.process_images(images)
    
    dataloader = DataLoader(
        dataset=ListDataset[str](filepaths),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    ds: List[torch.Tensor] = []
    for batch_doc in tqdm(dataloader, desc="Processing images"):
        with torch.no_grad():
            batch_doc = {k: v.to(Mymodel.device) for k, v in batch_doc.items()}
            embeddings_doc = Mymodel(**batch_doc)
        ds.extend(list(torch.unbind(embeddings_doc.to(Mydevice))))
    
    return ds

def getTextByPath(filepath: str) -> str:
    try:
        # 1. 从文件名中提取页数
        filename = os.path.basename(filepath)  # 获取纯文件名（不含路径）
        match = re.search(r'page_(\d+)\.', filename)  # 匹配 page_数字. 的模式
        
        if not match:
            print(f"警告: 无法从文件名 '{filename}' 中提取页数")
            return ""
        
        page_num = int(match.group(1))  # 提取数字部分并转为整数
        
        # 2. 读取 textList.json 文件 
        json_path = "./textList.json"
        with open(json_path, 'r', encoding='utf-8') as f:
            text_list = json.load(f)
        
        # 3. 获取对应页的文本（索引从0开始）
        # 假设textList.json中的索引0对应第1页
        index = page_num - 1
        
        if index < 0 or index >= len(text_list):
            print(f"警告: 页数 {page_num} 超出范围 (文本列表长度: {len(text_list)})")
            return ""
        
        return text_list[index]
    
    except Exception as e:
        print(f"处理文件 {filepath} 时出错: {str(e)}")
        return ""
    
def main():
    # 初始化Milvus
    retriever = MilvusColbertRetriever(collection_name="admin", milvus_client=client)
    # retriever.create_collection()
    # retriever.create_index()
    embeder=QwenEmbeder(client=Embedding_client)
    
    # # 处理图片并存入数据库
    # ds = processImg(filepaths)
    # for i, (path, embedding) in enumerate(zip(filepaths, ds)):
    #     data = {
    #         "colbert_vecs": embedding.float().cpu().numpy(),
    #         "doc_id": i,
    #         "filepath": path,
    #         "text": getTextByPath(path),
    #         "text_dense": embeder.getTextEmbeddings(getTextByPath(path),768)
    #         }
    #     retriever.insert(data)
    
    # 查询处理（可选）
    queries = ["what is Structure for Counting PERSON-TYPEs?"]
    Image_qs = processImageQuery(queries)
    customNames = ["2007.JC3IEDM"]
    # 实际单次交互中一次只会有一句查询
    query_params={
        "image_query": Image_qs[0].float().cpu().numpy(),
        "text_query": queries[0],
        "customNames": customNames,
        "text_dense_vector": embeder.getTextEmbeddings(queries[0],768)
    }
    res=retriever.Muti_hybrid_search_intersection(query_params,5)
    print(f"res:{res}")
    
    
    # for query in Image_qs:
    #     query = query.float().cpu().numpy()
    #     result = retriever.search(query, topk=5)
    #     for item in result:
    #         print(filepaths[item[1]]+"||"+item[2])

if __name__ == "__main__":
    main()