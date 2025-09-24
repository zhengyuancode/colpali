from tqdm import tqdm
import torch
from PIL import Image
from pathlib import Path
import logging
from milvus_conf import MilvusColbertRetriever, client as milvus_client
from pymilvus import Collection,connections
from pymilvus.bulk_writer import BulkFileType,RemoteBulkWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import os
import json
from mineru_process import run_mineru
import base64
from openai import OpenAI
from transformers import pipeline
from text_embeding import QwenEmbeder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


QWENAPIKEY="sk-f78b07615c8a45128d760579e6d42e1f"

AIclient = OpenAI(
    api_key=QWENAPIKEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def upload_minio(chunks, retriever, writer, embeder):
    
    logger.info("Start writing to the minio ..") 
    chunk_list = chunks["chunk"]
    for i in tqdm(range(len(chunk_list)),desc="Making BulkData", total=len(chunk_list)):
        text = chunk_list[i]["chunk"]
        if text is None or text.strip() == "":
            text_dense = [0.0] * 1024
        else:
            text_dense = embeder.getTextEmbeddings(text)
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
        if (i+1) % 1000 == 0:
            retriever.bulk_prepare_commit(writer)  
    retriever.bulk_prepare_commit(writer) 

def create_pdf_from_images(page_paths, output_dir, output_filename):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    width, height = A4

    page_map_path = []
    c = canvas.Canvas(output_path, pagesize=A4)
    
    for img_path in tqdm(page_paths,desc="Img to PDF",total=len(page_paths)):
        try:
            if not os.path.exists(img_path):
                logger.warning(f"The file does not exist-{img_path}")
                continue

            img = Image.open(img_path).convert('RGB')
            img_width, img_height = img.size
            
            scale = min(width / img_width, height / img_height)
            new_width = img_width * scale
            new_height = img_height * scale
            
            x = (width - new_width) / 2
            y = (height - new_height) / 2
            
            c.drawImage(img_path, x, y, new_width, new_height)
            c.showPage() 
            page_map_path.append(img_path)
            
        except Exception as e:
            print(f"Processing failed: {img_path} - {str(e)}")

    c.save()
    with open(output_dir+"/page_map_path.json", 'w', encoding='utf-8') as file:
        json.dump(page_map_path, file, indent=2, ensure_ascii=False)
        
    logger.info(f"PDF has been generated: {output_path}")
    return output_path, page_map_path

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        # Read binary data and perform Base64 encoding
        base64_data = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_data

def getCaption(image_path,language,client):
    try:
        completion  = client.chat.completions.create(
            model="qwen-vl-max", 
            messages=[
                {"role":"system",
                 "content":[
                     {"type": "text", 
                      "text": f"You are an assistant capable of extracting captions from images or tables.Your answer can only be in {language}."
                      }
                     ]
                 },
                {
                    "role": "user",
                    "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_to_base64(image_path)}"},
                            },
                            {"type": "text", "text": f"Please perform caption extraction on this image or table, etc.Your answer can only be in {language}."},
                        ]
                }
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Failed to call Alibaba Cloud API: {str(e)}")
        return ""

def getTextList(block_path,language,image_dir,output_path,client):  
    textList=[]
    with open(block_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
    # Find the largest page_idx
    max_page_idx = max(item["page_idx"] for item in data)
    
    # Initialize textList, with each element starting as an empty string
    textList = [""] * (max_page_idx + 1)
    
    for item in tqdm(data, desc="Processing items", unit="item"):
        page_idx = item["page_idx"]

        if item["type"] == "text":
            textList[page_idx] += item["text"] + " "
            
        elif item["type"] == "image":
            img_path=image_dir + item["img_path"]
            caption = ""
            if img_path and os.path.isfile(img_path):
                caption += getCaption(image_dir + item["img_path"], language,client)
            textList[page_idx] += (caption+ " ") if caption is not None else " "
            
        elif item["type"] == "table":
            img_path=image_dir + item["img_path"]
            caption = ""
            if img_path and os.path.isfile(img_path):
                caption += getCaption(image_dir + item["img_path"], language,client)
            elif (str(item["table_caption"]) != "[]"):
                caption += str(item["table_caption"])
            textList[page_idx] += (caption+ " ") if caption is not None else " "
            
        elif item["type"] == "equation":
            img_path=image_dir + item["img_path"]
            caption = ""
            if img_path and os.path.isfile(img_path):
                caption += getCaption(image_dir + item["img_path"], language,client)
            caption += item["text"]
            textList[page_idx] += (caption+ " ") if caption is not None else " "    
            
        else:
            print(item)
            
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(textList, file, indent=2)
        
    return output_path

def get_chunk_by_rewrite_caption(chunk_path, caption_path, page_map_path, file_name):
    pipe = pipeline(
        "image-text-to-text",
        model="llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float16,
        device_map="auto"
        )
    
    with open(caption_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    rewrite_data = {"chunk": [], "file_name": file_name}
    for i in tqdm(range(len(data)),desc="rewriting",total=len(data)):  
        if len(data[i]) <= 1500: 
            messages = [
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"The text paragraph provided by the user may have missing spaces, poor language, etc., which are not suitable for vector embedding. You need to improve this paragraph and return it, try not to change the original words as much as possible, and keep the original vocabulary as much as possible:\n[example]:\n5.ExceptionalitemsTheimagedisplaysafinancialtablecomparingspecificcostsandadjustmentsfortheyears2019and2018,inmillionsofdollars. \n[answer]:\n5.Exceptional items  The image displays a financial table comparing specific costs and adjustments for the years 2019 and 2018, in millions of dollars.\n-----------------------------\n[user input]:\n{data[i]}"},
                    ],
                },
            ]

            out = pipe(text=messages,max_new_tokens=512)
            rewrite = {
                "chunk":out[0]["generated_text"][1]["content"],
                "page_num": i,
                "page_path":page_map_path[i]
                }
            rewrite_data["chunk"].append(rewrite)
        else:
            max_len = 1500
            segments = []
            start = 0
            n = len(data[i])
            while start < n:
                end = min(start + max_len, n)

                if end == n:
                    segments.append(data[i][start:end])
                    break

                break_chars = {'.', ',', ';'}

                found = -1
                for j in range(end - 1, start - 1, -1):
                    if data[i][j] in break_chars:
                        found = j + 1 
                        break
                    
                if found == -1:
                    found = end

                segments.append(data[i][start:found])
                start = found
                
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
                    max_new_tokens=512
                    )
                rewrite = {
                    "chunk":out[0]["generated_text"][1]["content"],
                    "page_num": i,
                    "page_path":page_map_path[i]
                    }
                rewrite_data["chunk"].append(rewrite) 
        
    with open(chunk_path, 'w', encoding='utf-8') as file2:
        json.dump(rewrite_data, file2, indent=4, ensure_ascii=False)
    logger.info("chunk sucess")
    return rewrite_data

def main():
    collection_name = "vidore_tatdqa_text"
    # Dual channel design, the 'file_name' needs to be the same as the other channel, and the 'collection_name' must be different
    file_name = "vidore_tatdqa"
    # Initialize Milvus
    if(milvus_client.has_collection(collection_name=collection_name)):
        logger.info(f"{collection_name} vector database already exists.")
        retriever = MilvusColbertRetriever(collection_name=collection_name, milvus_client=milvus_client)
    else:
        retriever = MilvusColbertRetriever(collection_name=collection_name, milvus_client=milvus_client)
        retriever.create_collection_text_hybrid()
        retriever.create_index_text_hybrid()
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
    # The dataset-path of different datasets or documents should be different. Here, by default, one dataset-path is one document
    dataset_path = Path("/root/autodl-tmp/cpdfr-data/cpdfr/hybridSearch/experment_vidore/vidore_data/tatdqa_test/documents")
    image_extensions = {'.jpg', '.jpeg', '.png'}
    if not dataset_path.exists():
        logger.error(f"Path does not exist: {str(dataset_path)}")
        return
    for path in dataset_path.glob('*'):
        if path.is_file() and path.suffix.lower() in image_extensions:
            page_paths.append(str(path))
    
    if not Path(str(dataset_path)+f"/text/{file_name}.pdf").exists():
        pdf_path, page_map_path = create_pdf_from_images(page_paths,str(dataset_path)+"/text",f"{file_name}.pdf")
    else:
        pdf_path = Path(str(dataset_path)+f"/text/{file_name}.pdf")
        with open(str(dataset_path)+"/text/page_map_path.json", 'r', encoding='utf-8') as file1:
            page_map_path = json.load(file1)
    
    if not Path(str(dataset_path)+"/parse").exists():
        logger.info("Document parsing in progress...")
        run_mineru(pdf_path,str(dataset_path)+"/parse")
    
    if not os.path.exists(str(dataset_path)+f"/parse/{file_name}/caption_text_list.json"):
        logger.info("Start executing document caption")
        caption_text_list_path = getTextList(
                                    str(dataset_path)+f"/parse/{file_name}/auto/{file_name}_content_list.json",
                                    "english",
                                    str(dataset_path)+f"/parse/{file_name}/auto/",
                                    str(dataset_path)+f"/parse/{file_name}/caption_text_list.json",
                                    AIclient
                                    )
        print(f"caption_text_list_path:{caption_text_list_path}")
    else:
        caption_text_list_path = str(dataset_path)+f"/parse/{file_name}/caption_text_list.json"
    
    if not Path(str(dataset_path)+f"/parse/{file_name}/chunk.json").exists():
        chunks = get_chunk_by_rewrite_caption(str(dataset_path)+f"/parse/{file_name}/chunk.json", caption_text_list_path, page_map_path, file_name)
    else:
        with open(str(dataset_path)+f"/parse/{file_name}/chunk.json", 'r', encoding='utf-8') as file2:
            chunks = json.load(file2)              
    
    # Formally importing milvus requires building through this path
    # Remember the Minio bucket path prompted during execution
    # eg. 
    # Upload file '/root/autodl-tmp/cpdfr-data/cpdfr/hybridSearch/experiment_cpdfpqa/bulk_writer/5be22173-7649-4f78-9cd8-c36be645d74f/1.parquet' to 'cpdf_pqa/5be22173-7649-4f78-9cd8-c36be645d74f/1.parquet' (remote_bulk_writer.py:256)
    embeder=QwenEmbeder(url="https://api.siliconflow.cn/v1/embeddings")
    upload_minio(chunks, retriever, writer, embeder)
    logger.info("Data processing and uploading to Minio completed, please remember the data path on Minio!")
    
    
    

if __name__ == "__main__":
    main()