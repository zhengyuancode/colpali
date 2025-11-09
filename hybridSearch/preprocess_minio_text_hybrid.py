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
from transformers import AutoModel
from pdf_image import pdfToImage
import ast
from upload_milvus import upload_milvus
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

def upload_minio(chunks, retriever, writer, embeder):
    
    logger.info("Start writing to the minio ..") 
    chunk_list = chunks["chunks"]
    for i in tqdm(range(len(chunk_list)),desc="Making BulkData", total=len(chunk_list)):
        text = chunk_list[i]["chunk"]
        if text is None or text.strip() == "":
            print(f"Empty chunk at index {i}, skipping embedding.")
            text_dense = [0.0] * 2048
        else:
            text_dense = embeder.encode_text(
                            texts=[text],
                            task="retrieval",
                            prompt_name="passage"
                        )[0].float().cpu().numpy()
        data = {
            "text": text,
            "text_dense": text_dense,
            "page_num": chunk_list[i]["page_num"],
            "page_path": chunk_list[i]["page_path"],
            "file_name": chunks["file_name"],
            }

        retriever.bulk_insert_prepare_text_hybrid(data,writer)
        if (i+1) % 1000 == 0:
            retriever.bulk_prepare_commit(writer)  
    retriever.bulk_prepare_commit(writer) 
    
    batch_files = writer.batch_files
    if isinstance(batch_files, str):
        batch_files = ast.literal_eval(batch_files)
    path_str = batch_files[0][0]
    return path_str[:path_str.rfind('/') + 1]


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

def getCaption(image_path,language,client,isPage = False):
    if isPage:
        prompt = f"Please carefully analyze the currently provided page images, identify all the content (including text, tables, images, formulas, codes, and other meaningful information), and describe them in detail with textual information. Pay attention to and describe page numbers, chart numbers, or other identifiers (if any). Your answer must be in {language} and kept concise and accurate"
    else:
        prompt = f"Carefully analyze the image and describe its content in detail, using concise language.Your answer can only be in {language}."
    try:
        completion  = client.chat.completions.create(
            model="qwen-vl-plus", 
            # model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_to_base64(image_path)}"},
                            },
                            {"type": "text", "text": prompt},
                        ]
                }
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating caption for {image_path}: {str(e)}")
        return ""

def getTextList(block_path,language,image_dir,output_path,client,page_dir):  
    textList=[]
    with open(block_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
    # Find the largest page_idx
    max_page_idx = max(item["page_idx"] for item in data)
    
    # Initialize textList, with each element starting as an empty string
    textList = [""] * (max_page_idx + 1)
    
    for item in data:
        page_idx = item["page_idx"]

        if item["type"] == "text":
            textList[page_idx] += item["text"] + " "
            
        elif item["type"] == "image":
            img_path=image_dir + item["img_path"]
            caption = ""
            if img_path and os.path.isfile(img_path):
                caption += getCaption(image_dir + item["img_path"], language,client)
            if (str(item["image_caption"]) != "[]"):
                caption += "\n"+str(item["image_caption"])+"\n"
            textList[page_idx] += (caption+ " ") if caption is not None else " "
            
        elif item["type"] == "table":
            img_path=image_dir + item["img_path"]
            caption = ""
            if img_path and os.path.isfile(img_path):
                caption += getCaption(image_dir + item["img_path"], language,client)
            if (str(item["table_caption"]) != "[]"):
                caption += "\n"+str(item["table_caption"])+"\n"
            textList[page_idx] += (caption+ " ") if caption is not None else " "
            
        elif item["type"] == "equation":
            caption = ""
            if "img_path" in item:
                img_path=image_dir + item["img_path"]
                if img_path and os.path.isfile(img_path):
                    caption += getCaption(image_dir + item["img_path"], language,client)
            caption += " " + item["text"]
            textList[page_idx] += (caption+ " ") if caption is not None else " "    
            
        else:
            print(item)
    
    for i in range(len(textList)):
        item = textList[i]
        if item is None or item.strip() == "" or item == " ":
            textList[i] = getCaption(page_dir + f"/{i+1}.png", language,client,isPage=True)  
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(textList, file, indent=2)
        
    return output_path

def get_chunk_by_caption(chunks_path, caption_path, page_dir, file_name, chunk_size = 800):
    
    with open(caption_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    chunks_data = {"chunks": [], "file_name": file_name}
    for i in range(len(data)):  
        if len(data[i]) <= chunk_size: 
            chunk = {
                "chunk":data[i],
                "page_num": i+1,
                "page_path": page_dir + f"/{i+1}.png"
                }
            chunks_data["chunks"].append(chunk)
        else:
            segments = []
            start = 0
            n = len(data[i])
            while start < n:
                end = min(start + chunk_size, n)

                if end == n:
                    segments.append(data[i][start:end])
                    break

                break_chars = {'.', ',', ';', '!', '?', '\n'}

                found = -1
                for j in range(end - 1, start + (chunk_size // 2) - 1, -1):
                    if data[i][j] in break_chars:
                        found = j + 1 
                        break
                    
                if found == -1:
                    found = end

                segments.append(data[i][start:found])
                start = found
                
            for seg in segments:
                chunk = {
                    "chunk":seg,
                    "page_num": i+1,
                    "page_path": page_dir + f"/{i+1}.png"
                    }
                chunks_data["chunks"].append(chunk)
        
    with open(chunks_path, 'w', encoding='utf-8') as file2:
        json.dump(chunks_data, file2, indent=4, ensure_ascii=False)
    return chunks_path


def process_pdf_after_mineru(parse_path, pdf_name, language, output_dir, AIclient, chunk_size):
    """处理PDF在run_mineru完成后的后续步骤"""
    try:
        if not os.path.exists(str(parse_path)+f"/{pdf_name}/caption_text_list.json"):
            caption_text_list_path = getTextList(
                str(parse_path)+f"/{pdf_name}/auto/{pdf_name}_content_list.json",
                language,
                str(parse_path)+f"/{pdf_name}/auto/",
                str(parse_path)+f"/{pdf_name}/caption_text_list.json",
                AIclient,
                output_dir
            )
        else:
            caption_text_list_path = str(parse_path)+f"/{pdf_name}/caption_text_list.json" 

        if not Path(str(parse_path)+f"/{pdf_name}/chunk.json").exists():
            get_chunk_by_caption(str(parse_path)+f"/{pdf_name}/chunk.json", caption_text_list_path, output_dir, pdf_name, chunk_size)
        
        logger.info(f"Completed processing for {pdf_name}")
        
    except Exception as e:
        logger.error(f"Error in processing {pdf_name}: {str(e)}")


def preprocess_hybridText(collection_name, language, pdf_name, pdf_path, output_dir, parse_dir, milvus_client, embeder, AIclient, chunk_size=800):
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

    run_mineru(pdf_path,parse_dir)
    
    process_pdf_after_mineru(parse_dir, pdf_name, language, output_dir, AIclient, chunk_size)       
    with open(str(parse_dir)+f"/{pdf_name}/chunk.json", 'r', encoding='utf-8') as file:
        chunks = json.load(file)
        
    prefix = upload_minio(chunks, retriever, writer, embeder)
    
    job_id = upload_milvus(prefix, collection_name)
    # job = check_job(job_id,collection_name)
    # state = job["data"]["state"]
    # if state = "Completed" ,it means build success
    # if try to get job,but an error occurred, it may means build success
    milvus_client.release_collection(
        collection_name=collection_name
    )
    return job_id

def main():

    QWENAPIKEY="sk-f78b07615c8a45128d760579e6d42e1f"

    AIclient = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=QWENAPIKEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    collection_name = "test"
    language = "english"
    pdf_name = "cctns"
    pdf_path = "/home/gpu/milvus/backend/colpali/pdfs/dzy/cctns.pdf"
    output_dir = f"./pdfs/dzy/{pdf_name}"
    os.makedirs(output_dir, exist_ok=True)
    pdfToImage(pdf_path, output_dir)
    parse_dir = f"./pdfs/dzy/{pdf_name}/parse"
    os.makedirs(parse_dir, exist_ok=True)
    embeder = init_model()
    preprocess_hybridText(collection_name, language, pdf_name, pdf_path, output_dir, parse_dir, milvus_client, embeder, AIclient, chunk_size=800)
    
if __name__ == "__main__":
    main()
