from datasets import load_dataset,load_from_disk
from tqdm import tqdm
from typing import cast,List
import torch
from torch.utils.data import DataLoader
from PIL import Image
from pathlib import Path
import logging
from milvus_conf import MilvusColbertRetriever, client as milvus_client
from pymilvus import Collection,connections
from pymilvus.bulk_writer import BulkFileType,RemoteBulkWriter
import json
import os
from transformers import AutoModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




# cachedir = "/root/autodl-tmp/cpdfr-data/modelcache/huggingface/hub"
# model_name = "/root/autodl-tmp/cpdfr-data/modelcache/huggingface/hub/models--vidore--colpali-v1.3/snapshots/1b5c8929330df1a66de441a9b5409a878f0de5b0"

def init_model():
    model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v4", 
        trust_remote_code=True, 
        torch_dtype=torch.float16)
    model.to("cuda")
    return model

def processImg(page_paths: List[str],Mymodel):
    # Customize the collate function to load images as needed
    multivector_image_embeddings = Mymodel.encode_image(
        images=page_paths,
        task="retrieval",
        return_multivector=True,
    )
    return multivector_image_embeddings

def upload_minio(page_paths, retriever, file_name, writer, mymodel):
    
    if not page_paths:
        logger.error("No supported image files were found in the specified folder.")
        return
    
    logger.info("Obtain image vector groups and single vectors ..")
    colbert_vecs_list = processImg(page_paths, mymodel)
    
    logger.info("Start writing to the vector database ..") 
    for i, (page_path, colbert_vecs) in enumerate(tqdm(zip(page_paths, colbert_vecs_list),desc="Making BulkData", total=len(page_paths))):
        page_num = int(Path(page_path).stem)
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
    collection_name = "MMLongDoc"
    
    
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
    
    mymodel = init_model()
    # Modify the logic of different datasets
    pdf_dir = "/root/autodl-tmp/cpdfr-data/cpdfr/hybridSearch/experiment_multiHop/MMLongBench-Doc/documents"
    for filename in tqdm(os.listdir(pdf_dir), desc="Processing PDFs",total=len(os.listdir(pdf_dir))):
        if filename.lower().endswith('.pdf'):      
            pdf_name = os.path.splitext(filename)[0]
            output_dir = os.path.join(pdf_dir, pdf_name)
    
            page_paths = []
            dataset_path = Path(output_dir)
            
            image_extensions = {'.jpg', '.jpeg', '.png'}
            if not dataset_path.exists():
                os.makedirs(dataset_path)
                logger.info(f"Create Path: {str(dataset_path)}")
            for path in dataset_path.glob('*'):
                if path.is_file() and path.suffix.lower() in image_extensions:
                    page_paths.append(str(path))     
            
            
            # Formally importing milvus requires building through this path
            # Remember the Minio bucket path prompted during execution
            # eg. 
            # Upload file '/root/autodl-tmp/cpdfr-data/cpdfr/hybridSearch/experiment_cpdfpqa/bulk_writer/5be22173-7649-4f78-9cd8-c36be645d74f/1.parquet' to 'cpdf_pqa/5be22173-7649-4f78-9cd8-c36be645d74f/1.parquet' (remote_bulk_writer.py:256)
            if page_paths == []:
                logger.warning(f"No image files found for {pdf_name}, please check if the PDF has been converted to images.")
                continue
            upload_minio(page_paths, retriever, pdf_name, writer, mymodel)
    
    
    

if __name__ == "__main__":
    main()