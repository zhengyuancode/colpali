from datasets import load_dataset,load_from_disk
from tqdm import tqdm
from typing import cast,List
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device, ListDataset
import torch
from torch.utils.data import DataLoader
from PIL import Image
from pathlib import Path
import logging
from milvus_conf import MilvusColbertRetriever, client as milvus_client
from pymilvus import Collection,connections
from pymilvus.bulk_writer import BulkFileType,RemoteBulkWriter
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




# cachedir = "/root/autodl-tmp/cpdfr-data/modelcache/huggingface/hub"
# model_name = "/root/autodl-tmp/cpdfr-data/modelcache/huggingface/hub/models--vidore--colpali-v1.3/snapshots/1b5c8929330df1a66de441a9b5409a878f0de5b0"

def init_model():
    device = get_torch_device("cuda")
    model_name = "vidore/colpali-v1.3"
    model = ColPali.from_pretrained(
        model_name,
        # cache_dir=cachedir,
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=True,
        use_safetensors=True
    ).eval()
    processor = ColPaliProcessor.from_pretrained(model_name)
    return model, processor, device

def processImg(page_paths: List[str],Mymodel,Myprocessor,Mydevice):
    # Customize the collate function to load images as needed
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
        dataset=ListDataset[str](page_paths),
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

def upload_minio(page_paths, page_map_path, retriever, file_name, writer, mymodel, myprocessor, mydevice):
    
    if not page_paths:
        logger.error("No supported image files were found in the specified folder.")
        return
    
    logger.info("Obtain image vector groups and single vectors ..")
    colbert_vecs_list = processImg(page_paths, mymodel, myprocessor, mydevice)
    
    logger.info("Start writing to the vector database ..") 
    for i, (page_path, colbert_vecs) in enumerate(tqdm(zip(page_paths, colbert_vecs_list),desc="Making BulkData", total=len(page_paths))):
        page_num = i
        for j in range(len(page_map_path)):
            if page_map_path[j] == page_path:
                page_num = j
                
        data = {
            "colbert_vecs": colbert_vecs.float().cpu().numpy(),
            "page_num": page_num,
            "page_path": str(page_path),
            "file_name": file_name,
            }
        # A small amount of insertion can be done without going through Minio
        # retriever.insert(data)
        
        # Batch insertion requires going through Minio
        retriever.bulk_insert_prepare_multi_img(data,writer)
        if (i+1) % 100 == 0:
            retriever.bulk_prepare_commit(writer)  
    retriever.bulk_prepare_commit(writer) 

def main():
    collection_name = "vidore_tatdqa"
    # Dual channel design, the 'file_name' needs to be the same as the other channel, and the 'collection_name' must be different
    file_name = "vidore_tatdqa"
    # Initialize Milvus
    if(milvus_client.has_collection(collection_name=collection_name)):
        logger.info(f"{collection_name} vector database already exists.")
        retriever = MilvusColbertRetriever(collection_name=collection_name, milvus_client=milvus_client)
    else:
        retriever = MilvusColbertRetriever(collection_name=collection_name, milvus_client=milvus_client)
        retriever.create_collection_multi_img()
        retriever.create_index_multi_img()
        logger.info(f"Created {collection_name} vector database.")
    
    connections.connect(
        uri="http://127.0.0.1:19530", 
        token="root:Milvus"
    )

    collection = Collection(
        name=collection_name,
        using="default"
    )

    schema = collection.schema
    
    #default minio account
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

    writer = RemoteBulkWriter(
        schema=schema,
        remote_path="/"+collection_name,
        connect_param=conn,
        file_type=BulkFileType.PARQUET
    )
    
    
    # Modify the logic of different datasets
    page_paths = []
    dataset_path = Path("/root/autodl-tmp/cpdfr-data/cpdfr/hybridSearch/experment_vidore/vidore_data/tatdqa_test/documents")
    
    if not Path(str(dataset_path)+"/text/page_map_path.json"):
        logger.warning("Need to first build a text scheme vector library. If you want to generate only the graph scheme vector library, please modify the judgment statement.")
        return
    with open(str(dataset_path)+"/text/page_map_path.json", 'r', encoding='utf-8') as file:
        page_map_path = json.load(file)
    
    image_extensions = {'.jpg', '.jpeg', '.png'}
    if not dataset_path.exists():
        logger.error(f"Path does not exist: {str(dataset_path)}")
        return
    for path in dataset_path.glob('*'):
        if path.is_file() and path.suffix.lower() in image_extensions:
            page_paths.append(str(path))
                    
    
    
    # Formally importing milvus requires building through this path
    # Remember the Minio bucket path prompted during execution
    # eg. 
    # Upload file '/root/autodl-tmp/cpdfr-data/cpdfr/hybridSearch/experiment_cpdfpqa/bulk_writer/5be22173-7649-4f78-9cd8-c36be645d74f/1.parquet' to 'cpdf_pqa/5be22173-7649-4f78-9cd8-c36be645d74f/1.parquet' (remote_bulk_writer.py:256)
    mymodel, myprocessor, mydevice = init_model()
    upload_minio(page_paths, page_map_path, retriever, file_name, writer, mymodel, myprocessor, mydevice)
    
    
    

if __name__ == "__main__":
    main()