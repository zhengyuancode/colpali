from tqdm import tqdm
from pathlib import Path
import logging
# from milvus_conf import MilvusColbertRetriever, client as milvus_client
from pymilvus import Collection,connections
from pymilvus.bulk_writer import BulkFileType,RemoteBulkWriter
import os
import json
from text_embeding import QwenEmbeder
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import re
from FlagEmbedding import BGEM3FlagModel
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def upload_minio(chunks, retriever, writer, embeder):
    
    logger.info("Start writing to the minio ..") 
    chunk_list = chunks["chunk"]
    for i in tqdm(range(len(chunk_list)),desc="Making BulkData", total=len(chunk_list)):
        text = chunk_list[i]["chunk"]
        if text is None or text.strip() == "":
            if chunk_list[i]["page_num"] == -1:
                print(f"{chunk_list[i]["page_path"]} need to be caption and chunk")
                continue
            else:
                print(f"{chunk_list[i]["page_path"]} need to be caption again")
                continue
        else:
            text_dense = embeder.getTextEmbeddingsByLocal(text)
        data = {
            "text": text,
            "text_dense": text_dense,
            "page_num": chunk_list[i]["page_num"],
            "page_path": chunk_list[i]["page_path"],
            "file_name": chunks["file_name"],
            }
        # A small amount of insertion can be done without going through Minio
        # retriever.insert(data)
        
        # Batch insertion requires going through Minio
        retriever.bulk_insert_prepare_text_hybrid(data,writer)
        if (i+1) % 10000 == 0:
            retriever.bulk_prepare_commit(writer)  
    retriever.bulk_prepare_commit(writer) 

def split_long_chunks(chunks, max_length=1500):
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    
    new_chunks = []

    sentence_end_pattern = re.compile(r'[.!?;][\s\"\'’”)]*')
    
    for chunk in chunks:
        if len(chunk) < max_length:
            new_chunks.append(chunk)
            continue
        
        start = 0
        while start < len(chunk):
            end = start + max_length
            if end >= len(chunk):
                new_chunks.append(chunk[start:])
                break
            
            search_window = chunk[start:end]
            matches = list(sentence_end_pattern.finditer(search_window))
            
            if matches:
                last_match = matches[-1]
                split_pos = start + last_match.end()
                if split_pos == start:
                    split_pos = end 
            else:
                split_pos = end
            
            new_chunks.append(chunk[start:split_pos])
            start = split_pos
    return new_chunks

def get_chunk_by_vlm(chunk_path, page_img_dir, file_name, vlm, tokenizer):
    SEP = "<S-E-P>"
    caption_prompt = f"""You are a preprocessing module for an information retrieval system. Your task is to convert a single document image page—of any type (e.g., report, invoice, academic paper, form, etc.)—into an article described in natural language. Then, use separators({SEP}) to split this article into uniformly sized, semantically coherent text chunks.

    Perform the following steps:

    1. Comprehensively analyze the entire visible content of the document image.
    2. Extract and represent all elements faithfully:
        - Text: Transcribe all visible text verbatim, preserving original wording, line breaks, and logical groupings.
        - Tables: Convert each table into natural-language descriptions, clearly stating row and column relationships (e.g., 'Row 1: Name – John, Age – 32; Row 2: Name – Lisa, Age – 28').
        - Mathematical or scientific formulas: Render them accurately using plain-text notation that preserves structure and symbols (e.g., 'E equals m times c squared' or '∫₀¹ x² dx').
        - Figures, charts, or diagrams: Describe their visual content, key labels, data trends, and spatial context (e.g., 'A bar chart on the right shows sales increasing from Q1 to Q4, with Q4 peaking at $1.2M').
        - Other: A reasonable description can be in natural language.
    3. Synthesize all extracted content into fluent, natural-language prose, maintaining logical flow and contextual coherence. Do not add interpretations, summaries, or external knowledge. Then, use separators({SEP}) to split this prose into uniformly sized, semantically coherent text chunks.
    4. Insert separators({SEP}) into the complete description and return it, where each segmented string (chunk) has approximately the same length (e.g. 150-250 characters or 25-40 words), ensuring semantic boundaries (e.g. avoiding splitting table descriptions into middle rows).
    5. If there is a number representing the page number, describe this number as Page X.
    6. Ensure that each block size does not exceed 1000 plain text, unless there are special circumstances such as long tables.
    Example output:
    chunk1 {SEP} chunk2 {SEP} ... {SEP} chunkN
    """
    example_image = Image.open("/root/autodl-tmp/cpdfr-data/cpdfr/hybridSearch/experiment_multiHop/MMLongBench-Doc/documents/0b85477387a9d0cc33fca0f4becaa0e5/1.png").convert('RGB')
    
    example_answer = f"""NAVAL MEDICAL RESEARCH AND DEVELOPMENT NEWS Volume IV, Issue 12 December 2012 {SEP} In this issue... CO's Messages 2 Building Afghan Medical Capacity 3 USNS Mercy Pacific Partnership 4 DoD Bone Marrow Donor Program 5 ID Joint Planning Group 6 Capacity Building in Liberia 7 Kazakh Scientists Train at NMRC 8 Patient Condition Occurrence Tool 9 Combat Casualty Research Team 10 Accelerating Technology Transfer 11 NMRC Hosts Dining Out 12 Villasante Speaks at Notre Dame 13 Keane-Myers Speaks at Hopkins 14 Cub Scouts Learn Flag Etiquette 14 NMRC High School Outreach 15 NMRC Officers Teach Science 15 2012 Combined Federal Campaign 16 Ombudsman's Note 16 {SEP} NMRC Hosts Visit from U.S. Global Malaria Coordinator, President's Malaria Initiative SILVER SPRING, Md. - Rear Adm. (Ret.) Tim Ziemer, the U.S. Global Malaria Coordinator, President's Malaria Initiative, visited the Naval Medical Research Center (NMRC), November 29, for a brief on the current malaria vaccine research efforts and to tour the facility. He was interested in learning more about the malaria program at the laboratory. {SEP} Capt. John Sanders, NMRC commanding officer, provided a general overview of the NMRC enterprise with emphasis on the infectious diseases research efforts, specifically in the area of malaria. "The NMRC malaria research program is at the forefront of malaria research worldwide," Sanders pointed out and added, "Researchers here have been investigating methods to control and conquer malaria for more than two decades and have made some exciting discoveries in the last few years." {SEP} Ziemer visited a laboratory focused on investigating the liver stage of infection as a vaccine target, spoke with a researcher about antigen discovery and another researcher on the humanized mouse model developed at NMRC. He also had the opportunity to visit the insectary and hear about clinical immunology and current malaria (Continued on page 14) {SEP} A photograph shows Dr. Xiaoyan "Cathy" Zou, staff scientist from the Henry Jackson Foundation, discussing research on malaria with Rear Adm. (Ret.) Tim Ziemer, the U.S. Global Malaria Coordinator, President, Malaria Initiative. The setting is a laboratory environment with visible equipment including a microscope, biosafety cabinet, and pipettes. Both individuals are standing and engaged in conversation. {SEP} NM&D News is an authorized publication of the Naval Medical Research Center, 503 Robert Grant Avenue, Silver Spring, MD 20910. NM&D News is published monthly by the NMRC Public Affairs Office, 301-319-9378 or svc.pao.nmrc@med.navy.mil. {SEP} Commanding Officer Capt. John W. Sanders Executive Officer Capt. Elizabeth Montcalm-Smith Director for Administration Lt. Cmdr. Nathaniel Smith Public Affairs Officer Doris Ryan Editors Jan Helman Makeda Knott {SEP} http://www.facebook.com/navalmedicalresearchcenter {SEP} Use your smart-phone to access our website! {SEP} Page 1"""

    
    image_extensions = {'.jpg', '.jpeg', '.png'}
    if not os.path.isdir(page_img_dir):
        raise FileNotFoundError(f"Directory does not exist: {page_img_dir}")
    
    image_files = []
    for pagename in os.listdir(page_img_dir):
        page_path = os.path.join(page_img_dir, pagename)
        if os.path.isfile(page_path) and os.path.splitext(page_path)[1].lower() in image_extensions:
            image_files.append(page_path)
    
    caption_chunks = {"chunk": [], "file_name": file_name}
    for i in tqdm(range(len(image_files)),desc="chunking",total=len(image_files)):  
        image_path = image_files[i]
        image = Image.open(image_path).convert('RGB')
        msgs = [
            {'role': 'user', 'content': [example_image, caption_prompt]},
            {'role': 'assistant', 'content': [example_answer]},
            {'role': 'user', 'content': [image, caption_prompt]}
            ]
        answer = vlm.chat(
            msgs=msgs,
            tokenizer=tokenizer,
            enable_thinking=False,
            stream=False
        )
        try:
            chunks = answer.split(SEP)
            chunks = [chunk.replace('\n', ' ').replace('\r', ' ') for chunk in chunks]
            chunks = split_long_chunks(chunks, max_length=1500)
            for chunk in chunks:
                caption_chunks["chunk"].append({
                    "chunk": chunk,
                    "page_num": int(Path(image_path).stem),
                    "page_path": image_path
                })
        except Exception as e:
            logger.error(f"VLM returns format error: {e}")
            caption_chunks["chunk"].append({
                    "chunk": "",
                    "page_num": -1,
                    "page_path": image_path
                })
            continue
        
    with open(chunk_path, 'w', encoding='utf-8') as file2:
        json.dump(caption_chunks, file2, indent=4, ensure_ascii=False)
    logger.info("chunk sucess")
    return caption_chunks

def main():
    # collection_name = "MMLongDoc_text"

    # # Initialize Milvus
    # if(milvus_client.has_collection(collection_name=collection_name)):
    #     logger.info(f"{collection_name} vector database already exists.")
    #     retriever = MilvusColbertRetriever(collection_name=collection_name, milvus_client=milvus_client)
    # else:
    #     retriever = MilvusColbertRetriever(collection_name=collection_name, milvus_client=milvus_client)
    #     retriever.create_collection_text_hybrid()
    #     retriever.create_index_text_hybrid()
    #     logger.info(f"Created {collection_name} vector database.")
    
    # connections.connect(
    #     uri="http://127.0.0.1:19530", 
    #     token="root:Milvus"
    # )

    # collection = Collection(
    #     name=collection_name,
    #     using="default"
    # )

    # schema = collection.schema
    
    # #default minio account
    # ACCESS_KEY="minioadmin"
    # SECRET_KEY="minioadmin"
    # BUCKET_NAME="a-bucket"
    
    # conn = RemoteBulkWriter.S3ConnectParam(
    #     endpoint="localhost:9000", # the default MinIO service started along with Milvus
    #     access_key=ACCESS_KEY,
    #     secret_key=SECRET_KEY,
    #     bucket_name=BUCKET_NAME,
    #     secure=False
    # )

    # writer = RemoteBulkWriter(
    #     schema=schema,
    #     remote_path="/"+collection_name,
    #     connect_param=conn,
    #     file_type=BulkFileType.PARQUET
    # )
    
    vlm = AutoModel.from_pretrained(
        'openbmb/MiniCPM-V-4_5-int4', 
        trust_remote_code=True, # or openbmb/MiniCPM-o-2_6
        attn_implementation='sdpa', 
        torch_dtype=torch.bfloat16
        ) # sdpa or flash_attention_2, no eager
    vlm = vlm.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-4_5-int4', trust_remote_code=True) # or openbmb/MiniCPM-o-2_6
    
    embed_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    embeder=QwenEmbeder(url="https://api.siliconflow.cn/v1/embeddings",model=embed_model)
    
    # Modify the logic of different datasets
    pdf_dir = "/root/autodl-tmp/cpdfr-data/cpdfr/hybridSearch/experiment_multiHop/MMLongBench-Doc/documents"
    for filename in tqdm(os.listdir(pdf_dir), desc="Processing PDFs",total=len(os.listdir(pdf_dir))):
        if filename.lower().endswith('.pdf'):    
            pdf_name = os.path.splitext(filename)[0]
            page_img_dir = os.path.join(pdf_dir, pdf_name)
            if not os.path.exists(page_img_dir):
                os.makedirs(page_img_dir)
                
            if not Path(str(page_img_dir)+"/chunk.json").exists():
                chunks = get_chunk_by_vlm(str(page_img_dir)+"/chunk.json", page_img_dir, pdf_name, vlm, tokenizer)
            else:
                with open(str(page_img_dir)+"/chunk.json", 'r', encoding='utf-8') as file2:
                    chunks = json.load(file2)    
            # upload_minio(chunks, retriever, writer, embeder)          
      
            
    # Formally importing milvus requires building through this path
    # Remember the Minio bucket path prompted during execution
    # eg. 
    # Upload file '/root/autodl-tmp/cpdfr-data/cpdfr/hybridSearch/experiment_cpdfpqa/bulk_writer/5be22173-7649-4f78-9cd8-c36be645d74f/1.parquet' to 'cpdf_pqa/5be22173-7649-4f78-9cd8-c36be645d74f/1.parquet' (remote_bulk_writer.py:256)         
    # logger.info("Data processing and uploading to Minio completed, please remember the data path on Minio!")
    
    
    

if __name__ == "__main__":
    main()
