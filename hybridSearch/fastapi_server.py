from fastapi import FastAPI, HTTPException, Query,Request,UploadFile,Form,Request,Body
from fastapi.responses import JSONResponse,StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
import json
from typing import List, Tuple
import torch
import time
import logging
import os
from openai import OpenAI
import base64
import re
from pathlib import Path
from FileUtil import delete_file_if_exists,delete_directory_and_contents
import aiofiles
from pdf_image import pdfToImage
import sys
from transformers import AutoModel
from preprocess_minio_text_hybrid import preprocess_hybridText
from preprocess_minio_multi_img import preprocess_multiImg
from milvus_conf import MilvusColbertRetriever, client as milvus_client
from check_import_milvus_progress import check_job
from milvus_search import hybrid_text_search, chunk_to_patch_search, patch_to_patch_search
import pickle
from stable_baselines3 import PPO
print("Python executable:", sys.executable)

ZHIPUAPIKEY="f890fa44ea384a6baab00c725701a04b.1h0evvTQSAZALIp0"
QWENAPIKEY="sk-f78b07615c8a45128d760579e6d42e1f"

AIclient = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=QWENAPIKEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 创建静态文件目录（如果不存在）
os.makedirs("static", exist_ok=True)
# 可以将favicon.ico放在static目录下，或使用默认图标

# 全局计数器：当前正在处理的请求数量
processing_requests = 0

# 使用 asyncio.Lock 保护对 processing_requests 的修改
processing_lock = asyncio.Lock()


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("uvicorn.access").propagate = True
logging.getLogger("uvicorn.error").propagate = True
#-----------------------------------------------------------------------------------------------------------------------------

def image_to_base64(image_path):
    """
    将图片文件转换为 Base64 编码的字符串
    :param image_path: 图片文件的路径
    :return: Base64 编码的字符串
    """
    with open(image_path, "rb") as image_file:
        # 读取二进制数据并进行 Base64 编码
        base64_data = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_data

#-----------------------------------------------------------------------------------------------------------------------------
# 全局模型实例
embeder = None
topNmodel = None
scaler = None
searching_user=[]
def initialize_service():
    """初始化服务所需的模型和检索器"""
    global embeder, topNmodel, scaler, searching_user, processing_requests
    
    embeder_path = "/home/gpu/.cache/huggingface/hub/models--jinaai--jina-embeddings-v4/snapshots/737fa5c46f0262ceba4a462ffa1c5bcf01da416f"
    topNmodel_path = "/home/gpu/dzy/M3-CaseRAG/MDP/models/best_model.zip"
    scaler = pickle.load(open("/home/gpu/dzy/M3-CaseRAG/MDP/scaler.pkl", "rb"))
    topNmodel = PPO.load(topNmodel_path)
    embeder = AutoModel.from_pretrained(
        embeder_path,
        trust_remote_code=True, 
        torch_dtype=torch.float16,
        local_files_only=True
        )
    embeder.to("cuda")
#-----------------------------------------------------------------------------------------------------------------------------

# 初始化FastAPI应用
app = FastAPI(
    title="ColPali Image Retrieval Service",
    description="Service for querying image database using ColPali model",
    version="2.0.0"
)

# 允许跨域请求（根据需要配置）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rise-swu.cn:6565","https://www.rise-swu.cn:6565","https://ssh.rise-swu.cn:8024","http://ssh.rise-swu.cn:8024"],
    allow_credentials=True,
    allow_methods=["GET", "POST","OPTIONS"],
    allow_headers=["*"],
)

# 挂载静态文件服务
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化服务"""
    logger.info("Application startup event")
    initialize_service()

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时释放资源"""
    logger.info("Application shutdown event")
    # 这里可以添加资源释放逻辑
    try:
        # 1. 释放模型占用的GPU内存（核心操作）
        global embeder
        if embeder is not None:
            logger.info("Releasing model from GPU memory")
            del embeder  # 删除模型对象，释放其占用的显存

        # 2. 清空当前进程的GPU缓存（可选，更彻底）
        # 注意：此操作仅影响当前进程的缓存，不影响其他进程
        if torch.cuda.is_available():
            logger.info("Clearing current process GPU cache")
            torch.cuda.empty_cache()
        
        # 3. 关闭其他资源（如Milvus连接）
        global milvus_client
        if milvus_client is not None:
            milvus_client.close()
        
        logger.info("Current process resources released successfully")
        
    except Exception as e:
        logger.error(f"Resource cleanup error: {str(e)}")
@app.get("/")
async def root():
    """根路径重定向到API文档"""
    return {"message": "Welcome to ColPali Image Retrieval Service. Visit /docs for API documentation."}

# 处理favicon.ico请求
@app.get("/favicon.ico", include_in_schema=False)
async def get_favicon():
    """返回favicon图标，避免404"""
    return StaticFiles(directory="static").serve("favicon.ico")


# #TODO:学习进程，协程，线程机制，了解前沿的并发策略
# @app.post("/search/")
# async def search(
#     request: Request,
#     queries: List[str] = Body(..., description="List of search queries",embed=True),
#     topk: int = Body(5, ge=1, le=100, description="Number of results to return",embed=True)
# ) -> StreamingResponse:
#     """
#     执行图像检索查询
    
#     - **queries**: 搜索查询列表
#     - **topk**: 返回的结果数量
#     """
#     logger.info(f"Received search request with {len(queries)} queries, topk={topk}")
#     async def generate_stream():
#         try:
#             if await request.is_disconnected():
#                 logger.warning("客户端已断开，终止流式响应")
#                 return
            
#             global processing_requests
            
            
#              # 使用锁确保修改 processing_requests 是原子的
#             async with processing_lock:
#                 processing_requests += 1
#                 logger.info(f"待处理请求+1,当前{processing_requests}个")
                
#             # 调用同步函数，处理第一个 for 循环
#             try:
#                 query_embeddings, page_numbers_list, search_results_list = await asyncio.to_thread(
#                     process_queries, queries, topk
#                 )
#             except Exception as e:
#                 async with processing_lock:
#                     processing_requests -= 1
#                     logger.info(f"待处理请求-1,当前{processing_requests}个")
#                 logger.error(e)
#                 return
            
            
#             async with processing_lock:
#                 processing_requests -= 1
#                 logger.info(f"待处理请求-1,当前{processing_requests}个")
            
#             if await request.is_disconnected():
#                 logger.warning("客户端已断开，终止流式响应")
#                 return
#             #--------------------------------------------------------------------------------------------------
#             for j in range(len(search_results_list)):
#                 if page_numbers_list[j]:
#                     pages_str = f"最相关的{topk}个页码：\n[{','.join(map(str, page_numbers_list[j]))}]"
#                     first_block = {
#                         "type": "pages",
#                         "content": pages_str,
#                         "first_page": page_numbers_list[j][0] if page_numbers_list[j] else 0
#                     }
#                     yield json.dumps(first_block) + "\n\n"
                 
#                 # 准备图片用于RAG   
#                 base64_images=[]
#                 try:
#                     for item in search_results_list[j]:
#                         image_path = filepaths[item[1]]
#                         base64_str = image_to_base64(image_path) 
#                         base64_images.append({
#                             "type": "image_url",
#                             "image_url": {"url": f"data:image/png;base64,{base64_str}"}
#                         })
#                 except Exception as e:
#                     logger.error(f"准备图片时出错: {str(e)}")
#                     continue  # 跳过当前查询
                   
#                 try:
#                      response_stream  = AIclient.chat.completions.create(
#                         model="qwen-vl-max", 
#                         messages=[
#                         # {"role":"system","content":[{"type": "text", "text": "You are a military technology document understanding assistant and you answer questions in Chinese. You need to combine the provided document images to answer the questions."}]},
#                         {
#                             "role": "user",
#                             "content": base64_images + [{"type": "text", "text": queries[j]}]
#                         }
#                         ],
#                         stream=True
#                     )
#                 except Exception as e:
#                     logger.error(f"调用阿里云 API 失败: {str(e)}")
#                     error_block = {
#                         "type": "error",
#                         "content": f"调用 VLM 失败: {str(e)}"
#                     }
#                     yield json.dumps(error_block) + "\n\n"
#                     continue
                
#                 # 流式传输RAG结果
#                 try:
#                     logger.info("流式响应同步迭代")
#                     # 如果不是异步迭代器，使用同步迭代
#                     for chunk in await asyncio.to_thread(lambda: response_stream):
#                         if await request.is_disconnected():
#                             logger.warning("客户端已断开，终止流式响应")
#                             break
#                         if chunk.choices and chunk.choices[0].delta.content:
#                             token = chunk.choices[0].delta.content
#                             rag_block = {
#                                 "type": "rag",
#                                 "content": token
#                             }
#                             yield json.dumps(rag_block) + "\n\n"
                
#                 except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError) as e:
#                     logger.warning(f"客户端断开或网络异常: {e}")    
#                     break  # 终止当前查询的流式传输   
#                 except Exception as e:
#                     logger.error(f"Error in streaming RAG: {str(e)}")
#                     error_block = {
#                         "type": "error",
#                         "content": f"流式传输错误: {str(e)}"
#                     }
#                     yield json.dumps(error_block) + "\n\n"
#                 finally:
#                     # 发送结束标记
#                     yield json.dumps({"type": "end"}) + "\n\n"
#                     if response_stream and hasattr(response_stream, 'close'):
#                         response_stream.close()
            
            
                
#         except Exception as e:
#             logger.error(f"Error during search: {str(e)}")
#             error_block = {
#                 "type": "error",
#                 "content": f"处理请求时出错: {str(e)}"
#             }
#             yield json.dumps(error_block) + "\n\n"
        
#     return StreamingResponse(
#         generate_stream(),
#         media_type="application/x-ndjson",
#         headers={
#             "Connection": "keep-alive",
#             "Cache-Control": "no-cache",
#             "X-Accel-Buffering": "no"
#         }
#     )

@app.post("/convertPDF/")
async def convert_pdf(
    file: UploadFile, 
    uniqueId: str = Form(...), 
    username: str = Form(...)
):
    try:
        # 1. 保存上传的PDF文件
        pdf_path = Path(f"pdfs/{username}/{uniqueId}.pdf")
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(pdf_path, "wb") as f:
            content = await file.read()
            await f.write(content)
    except OSError as e:
        delete_file_if_exists(pdf_path)
        print(f"文件操作失败: {e}")
        block_content = [{"blocks": [], "error": "文件存储失败"}] 
        status = "文件存储失败"
        return JSONResponse(content={"blockContent": block_content,"status":status})
        
    try:
        # 2. 调用转换逻辑作为rag知识库存入milvus，生成并返回Block文件
        output_dir = f"pdfs/{username}/{uniqueId}"
        parse_dir = f"pdfs/{username}/{uniqueId}/parse"
        os.makedirs(output_dir, exist_ok=True)
        pdfToImage(pdf_path, output_dir)
        
        os.makedirs(parse_dir, exist_ok=True)
        collection_name = username
        collection_name_text = username+"_text"
        language = "english"
        
        
        job_id_text = preprocess_hybridText(collection_name_text, language, uniqueId, pdf_path, output_dir, parse_dir, milvus_client, embeder, AIclient ,chunk_size = 800)
        job_id = preprocess_multiImg(collection_name, output_dir, uniqueId, embeder)
        
        state_text = check_job(job_id_text,collection_name_text)
        state = check_job(job_id,collection_name)
        start_time = time.time()
        timeout_seconds=900
        while(state != "Completed" or state_text != "Completed"):
            # 检查是否超时
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(
                    f"任务等待超时 ({timeout_seconds}秒)。当前状态: state={state}, state_text={state_text}, 请联系管理员"
                )
            time.sleep(20)
            state_text = check_job(job_id_text,collection_name_text)
            state = check_job(job_id,collection_name)
            
        
        block_path = Path(f"{parse_dir}/{uniqueId}/auto/{uniqueId}_content_list.json")
        
        
        with block_path.open("r", encoding="utf-8") as f:
            block_content = json.load(f)
        for i in range(len(block_content)):
            block_content[i]["index"] = i
            block_content[i]["page_idx"] += 1
        status = "success"
    except FileNotFoundError:
        delete_file_if_exists(pdf_path)
        delete_directory_and_contents(output_dir)
        print(f"警告: JSON区块文件未找到: {block_path}")
        block_content = [{"blocks": [], "error": "JSON区块文件未找到"}] 
        status = "JSON区块文件未找到"
    except json.JSONDecodeError as e:
        delete_file_if_exists(pdf_path)
        delete_directory_and_contents(output_dir)
        print(f"JSON解析错误: {e}")
        block_content = [{"blocks": [], "error": "JSON解析失败"}]
        status = "JSON解析失败"
    except TimeoutError as e:
        delete_file_if_exists(pdf_path)
        delete_directory_and_contents(output_dir)
        print(f"milvus上传超时，需要管理员检查: {e}")
        block_content = [{"blocks": [], "error": "milvus上传超时"}]
        status = f"任务等待超时 ({timeout_seconds}秒)。当前状态: state={state}, state_text={state_text}, 请联系管理员"
    except Exception as e:
        print(f"错误: {e}")
        delete_file_if_exists(pdf_path)
        delete_directory_and_contents(output_dir)
        block_content = [{"blocks": [], "error": e}]
        status = e
        
    return JSONResponse(content={"blockContent": block_content,"status":status})
        
    
@app.post("/DeletePDF/")
async def delete_pdf(
    uniqueId: str = Form(...), 
    username: str = Form(...)
):
    dir_deleted = delete_directory_and_contents(f"./pdfs/{username}/{uniqueId}")
    file_deleted = delete_file_if_exists(f"./pdfs/{username}/{uniqueId}.pdf")
    # 初始化Milvus
    if(milvus_client.has_collection(collection_name=username+"_text")):
        retriever_text = MilvusColbertRetriever(collection_name=username+"_text", milvus_client=milvus_client)
    else:
        raise HTTPException(status_code=404, detail="用户无retriever_text向量库")
    
    if(milvus_client.has_collection(collection_name=username)):
        retriever_img = MilvusColbertRetriever(collection_name=username, milvus_client=milvus_client)
    else:
        raise HTTPException(status_code=404, detail="用户无retriever_img向量库")

    retriever_text.delete_entity_by_file(uniqueId)
    retriever_img.delete_entity_by_file(uniqueId)
    
    milvus_client.release_collection(
        collection_name=username+"_text"
    )
    
    milvus_client.release_collection(
        collection_name=username
    )
    
    # 判断操作是否成功
    if dir_deleted and file_deleted:
        return JSONResponse(
            status_code=200,
            content={"status": "success", "message": "文件及目录已成功删除"}
        )
    else:
        # 至少有一个操作失败
        error_message = []
        if not dir_deleted:
            error_message.append("目录删除失败")
        if not file_deleted:
            error_message.append("文件删除失败")
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": ", ".join(error_message)}
        )
        
# @app.post("/preProcessRAG/")
# async def pre_process_rag(
#     uniqueId: str = Form(...), 
#     username: str = Form(...),
#     customName: str = Form(...)
# ):
#     #目前默认先支持英文
#     language = "english"
    
#     # 构建PDF文件路径和页面文件夹路径
#     pdf_path = Path(f"./pdfs/{username}/{uniqueId}.pdf")
#     pages_path = Path(f"./pdfs/{username}/{uniqueId}/pages")
#     custom_path = Path(f"./pdfs/{username}/{uniqueId}")

#     print(f"pdf_path:{pdf_path}")
#     # 检查PDF文件是否存在且是文件
#     if not pdf_path.exists():
#         logger.info("PDF文件不存在")
#         raise HTTPException(status_code=404, detail="PDF文件不存在")
#     if not pdf_path.is_file():
#         logger.info("指定的路径不是有效的PDF文件")
#         raise HTTPException(status_code=400, detail="指定的路径不是有效的PDF文件")

#     if not custom_path.exists():
#         logger.info("不存在文档解析处理结果")
#         raise HTTPException(status_code=400, detail="不存在文档解析处理结果")
    
#     # 检查页面文件夹是否存在
#     if pages_path.exists():
#         logger.info("页面文件夹已存在")
#         raise HTTPException(status_code=400, detail="页面文件夹已存在")
    
#     if Path(str(custom_path)+f"/caption_text_list.json").exists():
#         logger.info("caption_text_list已存在")
#         raise HTTPException(status_code=400, detail="caption_text_list已存在")

#     # 创建页面文件夹
#     pages_path.mkdir()
    
#     #pdf转为页面图片存储
#     logger.info("pdf转为页面图片存储...")
#     pdfToImage(pdf_path,pages_path)
    
#     ImagePaths = [os.path.join(pages_path, name) for name in os.listdir(pages_path)]
    
#     #TODO:获取caption文件,此过程较慢，可后续优化
#     logger.info("获取caption文件...")
#     # caption_text_list_path = getTextList(
#     #                             str(custom_path)+f"/auto/{uniqueId}_content_list.json",
#     #                             "english",
#     #                             str(custom_path)+f"/auto/",
#     #                             str(custom_path)+f"/caption_text_list.json",
#     #                             AIclient
#     #                             )
    
#     caption_text_list_path = getTextList_local(
#                                 str(custom_path)+f"/auto/{uniqueId}_content_list.json",
#                                 str(custom_path)+f"/auto/",
#                                 str(custom_path)+f"/caption_text_list.json",
#                                 img_captioner
#                                 )
    
#     #存入milvus
    
#     #获取图片向量组
#     logger.info("获取图片向量组和单向量...")
#     ds = processImg(ImagePaths,model,processor,device)
#     single_img_vecs = processImg_single(ImagePaths,model_2)
    
#     # 初始化Milvus
#     if(client.has_collection(collection_name=username)):
#         logger.info("用户已存在向量数据库")
#         retriever = MilvusColbertRetriever(collection_name=username, milvus_client=client)
#     else:
#         retriever = MilvusColbertRetriever(collection_name=username, milvus_client=client)
#         retriever.create_collection()
#         retriever.create_index()
        
#     if(client_img.has_collection(collection_name=username+"_img")):
#         logger.info("用户已存在向量数据库(纯图像RAG版本)")
#         retriever_img = MilvusColbertRetriever_img(collection_name=username+"_img", milvus_client=client_img)
#     else:
#         retriever_img = MilvusColbertRetriever_img(collection_name=username+"_img", milvus_client=client_img)
#         retriever_img.create_collection()
#         retriever_img.create_index()
    
#     logger.info("开始写入向量数据库...")    
#     for i, (Imgpath, embedding, single_img_embedding) in enumerate(zip(ImagePaths, ds, single_img_vecs)):
#         text = getTextByPath(Imgpath,caption_text_list_path)
#         # 判断 text 是否为空（None 或空字符串）
#         if text is None or text.strip() == "":
#             text_dense_value = [0.0] * 1024
#         else:
#             text_dense_value = embeder.getTextEmbeddings(text)
#         data = {
#             "colbert_vecs": embedding.float().cpu().numpy(),
#             "doc_id": i,
#             "filepath": Imgpath,
#             "text": text,
#             "customName": customName,
#             "text_dense": text_dense_value
#             }
#         retriever.insert(data)
        
#         data_img = {
#             "multiple_image_dense": embedding.float().cpu().numpy(),
#             "doc_id": i,
#             "filepath": Imgpath,
#             "single_image_dense":single_img_embedding.float().cpu().numpy(),
#             "customName": customName
#             }
#         retriever_img.insert(data_img)

#     return {"message": "RAG知识库搭建成功"}


# @app.post("/delete_RAG/")
# async def delete_RAG(
#     uniqueId: str = Form(...), 
#     username: str = Form(...),
#     customName: str = Form(...)
# ):
#     # 初始化Milvus
#     if(client.has_collection(collection_name=username)):
#         retriever = MilvusColbertRetriever(collection_name=username, milvus_client=client)
#     else:
#         logger.info("用户无RAG向量库")
#         raise HTTPException(status_code=404, detail="用户无RAG向量库")
    
#     if(client_img.has_collection(collection_name=username+"_img")):
#         retriever_img = MilvusColbertRetriever_img(collection_name=username+"_img", milvus_client=client_img)
#     else:
#         logger.info("用户无RAG_img向量库")
#         raise HTTPException(status_code=404, detail="用户无RAG_img向量库")
    
#     logger.info(f"删除知识库为{customName}的向量实体...")
#     retriever.delete_entity(customName)
#     retriever_img.delete_entity(customName)
    
#     dir_deleted = delete_directory_and_contents(f"./pdfs/{username}/{uniqueId}/pages")
#     file_deleted = delete_file_if_exists(f"./pdfs/{username}/{uniqueId}/caption_text_list.json")
    
#     # 判断操作是否成功
#     if dir_deleted and file_deleted:
#         return JSONResponse(
#             status_code=200,
#             content={"status": "success", "message": "文件及目录已成功删除"}
#         )
#     else:
#         # 至少有一个操作失败
#         error_message = []
#         if not dir_deleted:
#             error_message.append("目录删除失败")
#         if not file_deleted:
#             error_message.append("文件删除失败")
#         return JSONResponse(
#             status_code=400,
#             content={"status": "error", "message": ", ".join(error_message)}
#         )
        
# #调试用，平时禁用
# @app.post("/delete_ALL/")
# async def delete_ALL(
#     username: str = Form(...)
# ):
#     # 初始化Milvus
#     if(client.has_collection(collection_name=username)):
#         retriever = MilvusColbertRetriever(collection_name=username, milvus_client=client)
#     else:
#         logger.info("用户无RAG向量库")
#         raise HTTPException(status_code=404, detail="用户无RAG向量库")
    
#     logger.info(f"删除{username}的所有向量实体及向量库...")
#     retriever.delete_entity_all()

# @app.post("/count_entity/")
# async def count_entity(
#     username: str = Form(...)
# ):
#     # 初始化Milvus
#     if(client.has_collection(collection_name=username)):
#         retriever = MilvusColbertRetriever(collection_name=username, milvus_client=client)
#     else:
#         logger.info("用户无RAG向量库")
#         raise HTTPException(status_code=404, detail="用户无RAG向量库")
    
#     res=retriever.count_entity()
#     logger.info(f"统计结果：\n{username}的向量集合中：\n{res}")
#     return f"统计结果：\n{username}的向量集合中：\n{res}"
    
    
        
@app.post("/search_all_customName/")
async def search_all_customName(
    username: str = Form(...),
):
    # 初始化Milvus
    if(milvus_client.has_collection(collection_name=username)):
        retriever_img = MilvusColbertRetriever(collection_name=username, milvus_client=milvus_client)
    else:
        logger.info("用户无RAG向量库")
        raise HTTPException(status_code=404, detail="用户无RAG向量库")
    res = retriever_img.search_all_customName()
    milvus_client.release_collection(
        collection_name=username
    )
    return res




def process_queries_hybrid(username: str ,queries: List[str], fileNames: List[str], topk: int,searchMethod:str):
    hybrid_text_collection_name = username+"_text"
    multi_img_collection_name = username
    
    hybrid_text_retriever = MilvusColbertRetriever(collection_name=hybrid_text_collection_name, milvus_client=milvus_client)
    multi_img_retriever = MilvusColbertRetriever(collection_name=multi_img_collection_name, milvus_client=milvus_client)
    
    search_results_list = []
    if searchMethod == "hybridTextSearch":
        _, search_results_list = hybrid_text_search(queries, fileNames, topk, hybrid_text_retriever, embeder)
    if searchMethod == "Chunk to Patch":
        search_results_list = chunk_to_patch_search(queries, fileNames, topk, multi_img_retriever, hybrid_text_retriever, embeder, coarse = topk*10)
    if searchMethod == "Patch to Patch":
        search_results_list = patch_to_patch_search(queries, fileNames, topk, embeder, topNmodel, scaler, multi_img_retriever )
    
    milvus_client.release_collection(
        collection_name=hybrid_text_collection_name
    )
    milvus_client.release_collection(
        collection_name=multi_img_collection_name
    )
    return search_results_list


@app.post("/hybridSearch/")
async def hybridSearch(
    request: Request,
    username: str = Body(..., description="string of username",embed=True),
    queries: List[str] = Body(..., description="List of search queries",embed=True),
    uniqueIds: List[str] = Body(..., description="List of uniqueIds",embed=True),
    customNames: List[str] = Body(..., description="List of customName",embed=True),
    topk: int = Body(5, ge=1, le=100, description="Number of results to return",embed=True),
    searchMethod: str = Body(..., description="Search method", embed=True)
) -> StreamingResponse:
    """
    执行多模态混合检索查询
    
    - **queries**: 搜索查询列表
    - **uniqueIds**: 知识库对应的uid列表
    - **customNames**: 查找知识库列表
    - **topk**: 返回的结果数量
    """
    logger.info(f"Received search request with {len(queries)} queries, topk={topk}")
    async def generate_stream():
        if(username not in searching_user):
            searching_user.append(username)
        else:
            error_block = {
                    "type": "error",
                    "content": f"{username}已有一个请求在队列中"
                }
            yield json.dumps(error_block) + "\n\n"
            return
        
        try:
            if(searchMethod not in ["hybridTextSearch","Patch to Patch","Chunk to Patch"]):
                logger.warning("非法的检索方法")
                error_block = {
                    "type": "error",
                    "content": f"非法的检索方法: {searchMethod}"
                }
                yield json.dumps(error_block) + "\n\n"
                return
            if await request.is_disconnected():
                logger.warning("客户端已断开，终止流式响应")
                return
            
            global processing_requests
             # 使用锁确保修改 processing_requests 是原子的
            async with processing_lock:
                processing_requests += 1
                logger.info(f"待处理请求+1,当前{processing_requests}个")
            
            logger.info(f"使用{searchMethod}方法")
            print(f"使用{searchMethod}方法")
            # 调用同步函数，处理第一个 for 循环
            search_results_list = await asyncio.to_thread(
                process_queries_hybrid, username, queries, uniqueIds, topk, searchMethod
            )
            
            async with processing_lock:
                processing_requests -= 1
                logger.info(f"待处理请求-1,当前{processing_requests}个")
            
            if await request.is_disconnected():
                logger.warning("客户端已断开，终止流式响应")
                return
            #--------------------------------------------------------------------------------------------------
            for j in range(len(search_results_list)):
                search_results = search_results_list[j]
                if searchMethod == "hybridTextSearch":
                    search_results = list(dict.fromkeys(search_results))
                page_ans="最相关的页码如下：\n"
                first_page = 0
                first_customName = ""
                for m in range(len(search_results)):
                    for n in range(len(uniqueIds)):
                        if uniqueIds[n] in search_results[m]:
                            page_number = str(Path(search_results[m]).stem)
                            page_ans = page_ans + f"元知识库：{customNames[n]},页数：{page_number}\n"
                            if(m == 0):
                                first_page = page_number
                                first_customName = customNames[n]
                                

                first_block = {
                    "type": "pages",
                    "content": page_ans,
                    "first_page": first_page,
                    "first_customName": first_customName
                }
                yield json.dumps(first_block) + "\n\n"
                                          
                # 准备图片用于RAG   
                base64_images=[]
                try:
                    for image_path in search_results:
                        base64_str = image_to_base64(image_path) 
                        base64_images.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_str}"}
                        })
                except Exception as e:
                    logger.error(f"准备图片时出错: {str(e)}")
                    continue  # 跳过当前查询
                   
                try:
                     response_stream  = AIclient.chat.completions.create(
                        model="qwen-vl-max", 
                        messages=[
                        {"role":"system","content":[{"type": "text", "text": "You need to combine the image information provided by the user's document page with your own knowledge base to answer the user's query. Your answer should be in Chinese."}]},
                        {
                            "role": "user",
                            "content": base64_images + [{"type": "text", "text": queries[j]}]
                        }
                        ],
                        stream=True
                    )
                except Exception as e:
                    logger.error(f"调用阿里云 API 失败: {str(e)}")
                    error_block = {
                        "type": "error",
                        "content": f"调用 VLM 失败: {str(e)}"
                    }
                    yield json.dumps(error_block) + "\n\n"
                    continue
                
                # 流式传输RAG结果
                try:
                    logger.info("流式响应同步迭代")
                    # 如果不是异步迭代器，使用同步迭代
                    for chunk in await asyncio.to_thread(lambda: response_stream):
                        if await request.is_disconnected():
                            logger.warning("客户端已断开，终止流式响应")
                            break
                        if chunk.choices and chunk.choices[0].delta.content:
                            token = chunk.choices[0].delta.content
                            rag_block = {
                                "type": "rag",
                                "content": token
                            }
                            yield json.dumps(rag_block) + "\n\n"
                
                except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError) as e:
                    logger.warning(f"客户端断开或网络异常: {e}")    
                    break  # 终止当前查询的流式传输   
                except Exception as e:
                    logger.error(f"Error in streaming RAG: {str(e)}")
                    error_block = {
                        "type": "error",
                        "content": f"流式传输错误: {str(e)}"
                    }
                    yield json.dumps(error_block) + "\n\n"
                finally:
                    if(username in searching_user): 
                        searching_user.remove(username)
                    # 发送结束标记
                    yield json.dumps({"type": "end"}) + "\n\n"
                    if response_stream and hasattr(response_stream, 'close'):
                        response_stream.close()
            
            
                
        except Exception as e:
            if(username in searching_user): 
                        searching_user.remove(username)
            logger.error(f"Error during search: {str(e)}")
            error_block = {
                "type": "error",
                "content": f"处理请求时出错: {str(e)}"
            }
            yield json.dumps(error_block) + "\n\n"
        
    return StreamingResponse(
        generate_stream(),
        media_type="application/x-ndjson",
        headers={
            "Connection": "keep-alive",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

# def checkQA(query,answer,username,customNames,topk,searchMethod):
#     search_results_list = process_queries_hybrid(username,[query],customNames,topk,searchMethod,processor,model,model_2,device,embeder)
#     #默认只有一个查询
#     search_results = search_results_list[0]
#     base64_images=[]
#     for image_path in search_results:
#         base64_str = image_to_base64(image_path) 
#         base64_images.append({
#             "type": "image_url",
#             "image_url": {"url": f"data:image/png;base64,{base64_str}"}
#         })
#     checkAns = getQAcheck(query,answer,base64_images)
#     if(checkAns.lower() == "yes"):
#         return "yes"
#     else:
#         return base64_images
    
    
# @app.post("/hybridSearch_MultiHop/")
# async def hybridSearch_MultiHop(
#     request: Request,
#     username: str = Body(..., description="string of username",embed=True),
#     queries: List[str] = Body(..., description="List of search queries",embed=True),
#     uniqueIds: List[str] = Body(..., description="List of uniqueIds",embed=True),
#     customNames: List[str] = Body(..., description="List of customName",embed=True),
#     topk: int = Body(5, ge=1, le=100, description="Number of results to return",embed=True),
#     searchMethod: str = Body(..., description="Search method", embed=True),
#     Q_count_max: int = Body(5, ge=1, le=100, description="Number of CoQ_nodes_max",embed=True)
# ) -> StreamingResponse:
#     """
#     执行多模态混合多跳检索查询
    
#     """
#     logger.info(f"Received search request with {len(queries)} queries, topk={topk}")
#     async def generate_stream():
#         if(username not in searching_user):
#             searching_user.append(username)
#         else:
#             error_block = {
#                     "type": "error",
#                     "content": f"该用户已有一个请求在队列中: {username}"
#                 }
#             yield json.dumps(error_block) + "\n\n"
#             return
        
#         try:
#             if(searchMethod not in ["Muti_hybrid_search","Muti_hybrid_search_intersection","Muti_hybrid_search_img_in_text","Muti_hybrid_search_text_in_img","Muti_hybrid_search_multiple_in_single","Muti_hybrid_search_single_in_multiple"]):
#                 logger.warning("非法的检索方法")
#                 error_block = {
#                     "type": "error",
#                     "content": f"非法的检索方法: {searchMethod}"
#                 }
#                 yield json.dumps(error_block) + "\n\n"
#                 return
#             if await request.is_disconnected():
#                 logger.warning("客户端已断开，终止流式响应")
#                 return
            
#             global processing_requests
            
            
#              # 使用锁确保修改 processing_requests 是原子的
#             async with processing_lock:
#                 processing_requests += 1
#                 logger.info(f"待处理请求+1,当前{processing_requests}个")
            
#             logger.info(f"使用{searchMethod}方法")
#             print(f"使用{searchMethod}方法")
            
#             #进行多跳逻辑,默认queries列表中一次查询里只有一个query
#             #-----------------------------------------------------------------------
#             isMultiHop = True
#             isExceed = False
#             history=""
#             base64_images=[]
#             CoQ=None
#             historical_right_CoQ=[]
#             while(isMultiHop):
#                 print(f"history:{history}")
#                 CoQ = getQueries(queries[0],history,Q_count_max,base64_images)
#                 print(CoQ)
#                 chain = CoQ["chain"]
#                 for i in range(0,len(chain)):
#                     if(i+1 <= len(historical_right_CoQ)):
#                         continue
#                     if(chain[i]["unsolved_query"] != ""):
#                         multi_search_results_list = await asyncio.to_thread(
#                             process_queries_hybrid, 
#                             username, 
#                             [chain[i]["unsolved_query"]], 
#                             customNames, 
#                             topk,
#                             searchMethod,
#                             processor,
#                             model,
#                             model_2,
#                             device,
#                             embeder
#                         )
#                         for image_path in multi_search_results_list[0]:
#                             base64_str = image_to_base64(image_path) 
#                             base64_images.append({
#                                 "type": "image_url",
#                                 "image_url": {"url": f"data:image/png;base64,{base64_str}"}
#                             })
#                         break
#                     #节点检查方法判断节点答案正确性
#                     checkQAans = checkQA(chain[i]["query"],chain[i]["answer"],username,customNames,topk,searchMethod)
#                     if(not isinstance(checkQAans, str)):
#                         base64_images = checkQAans
#                         break
                        
#                     if(len(historical_right_CoQ) < Q_count_max):   
#                         historical_right_CoQ.append(chain[i])
#                     if(len(historical_right_CoQ) >= Q_count_max):
#                         isMultiHop = False
#                         isExceed = True
#                     if(i == len(chain) - 1):
#                         isMultiHop = False
#                 history = getHistoricalAns(historical_right_CoQ)        


#             #------------------------------------------------------------------------
#             async with processing_lock:
#                 processing_requests -= 1
#                 logger.info(f"待处理请求-1,当前{processing_requests}个")
            
#             if await request.is_disconnected():
#                 logger.warning("客户端已断开，终止流式响应")
#                 return
#             #--------------------------------------------------------------------------------------------------
#             if(isExceed):
#                 print("超出多跳推理最大节点数")
#             else:
#                 print(CoQ)  
                
#         except Exception as e:
#             if(username in searching_user): 
#                         searching_user.remove(username)
#             logger.error(f"Error during search: {str(e)}")
#             error_block = {
#                 "type": "error",
#                 "content": f"处理请求时出错: {str(e)}"
#             }
#             yield json.dumps(error_block) + "\n\n"
        
#     return StreamingResponse(
#         generate_stream(),
#         media_type="application/x-ndjson",
#         headers={
#             "Connection": "keep-alive",
#             "Cache-Control": "no-cache",
#             "X-Accel-Buffering": "no"
#         }
#     )
    
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting service...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")