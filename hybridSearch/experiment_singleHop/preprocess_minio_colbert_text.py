import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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

from transformers import AutoModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




# cachedir = "/root/autodl-tmp/cpdfr-data/modelcache/huggingface/hub"
# model_name = "/root/autodl-tmp/cpdfr-data/modelcache/huggingface/hub/models--vidore--colpali-v1.3/snapshots/1b5c8929330df1a66de441a9b5409a878f0de5b0"

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

def processText(page_text_paths: List[str],Mymodel,batch_size):
    with open(page_text_paths, 'r', encoding='utf-8') as file:
        texts = json.load(file)
    # Customize the collate function to load images as needed
    chunk_texts = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    results = []
    for ct in chunk_texts:
        with torch.no_grad():
            multivector_text_embeddings = Mymodel.encode_text(
                texts=ct,
                task="retrieval",
                prompt_name="passage",
                return_multivector=True,
            )
            
        result = [
            emb.float().cpu().numpy() if isinstance(emb, torch.Tensor) else emb
            for emb in multivector_text_embeddings
        ]
        results.extend(result)
        del multivector_text_embeddings
        torch.cuda.empty_cache()  
        
    return results

def upload_minio(output_dir, page_text_paths, retriever, file_name, writer, mymodel):
    
    if not page_text_paths:
        logger.error("No supported text files were found in the specified folder.")
        return
    colbert_vecs_list = processText(page_text_paths, mymodel, 1)
    for i in tqdm(range(len(colbert_vecs_list)), desc="Making BulkData", total=len(colbert_vecs_list)):
        colbert_vecs = colbert_vecs_list[i]
        page_path = output_dir + f"/{i+1}.png"
        page_num = int(Path(page_path).stem)
        data = {
            "colbert_vecs": colbert_vecs,
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
    collection_name = "MMLongDoc_colbert_text"
    
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
    pdf_dir = "/home/gpu/dzy/M3-CaseRAG/experiment_multiHop/MMLongBench-Doc/documents"
    for filename in tqdm(os.listdir(pdf_dir), desc="Processing PDFs",total=len(os.listdir(pdf_dir))):
        if filename.lower().endswith('.pdf'):      
            pdf_name = os.path.splitext(filename)[0]
            output_dir = os.path.join(pdf_dir, pdf_name)
    
            page_text_paths = output_dir + f"/parse/{pdf_name}/caption_text_list.json"
            if not Path(page_text_paths).exists:
                logger.error(f"{page_text_paths} not exists")
                return
            
            # Formally importing milvus requires building through this path
            # Remember the Minio bucket path prompted during execution
            # eg. 
            # Upload file '/root/autodl-tmp/cpdfr-data/cpdfr/hybridSearch/experiment_cpdfpqa/bulk_writer/5be22173-7649-4f78-9cd8-c36be645d74f/1.parquet' to 'cpdf_pqa/5be22173-7649-4f78-9cd8-c36be645d74f/1.parquet' (remote_bulk_writer.py:256)
            upload_minio(output_dir, page_text_paths, retriever, pdf_name, writer, mymodel)
    
    
    

if __name__ == "__main__":
    main()