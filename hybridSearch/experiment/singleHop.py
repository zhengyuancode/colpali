from typing import List
from fastapi_server import logger,processing_lock,process_queries_hybrid,image_to_base64,AIclient
import asyncio
import json

async def hybridSearch(
    queries: List[str],
    topk: int,
    searchMethod: str
):
    """
    执行多模态混合检索查询
    
    - **queries**: 搜索查询列表
    - **uniqueIds**: 知识库对应的uid列表
    - **customNames**: 查找知识库列表
    - **topk**: 返回的结果数量
    """ 
    collection_name = "vidoseek"
    
    customNames = []
    with open("/home/gpu/milvus/backend/colpali/ViDoSeek/subfolders.json", 'r', encoding='utf-8') as file:
        subfolders = json.load(file)
    subfolders_list = subfolders["subfolders"]
    for pdf in subfolders_list:
        customNames.append(pdf["pdfId"])
    
    try:
        if(searchMethod not in ["Muti_hybrid_search","Muti_hybrid_search_intersection","Muti_hybrid_search_img_in_text","Muti_hybrid_search_text_in_img","Muti_hybrid_search_multiple_in_single","Muti_hybrid_search_single_in_multiple"]):
            logger.warning("非法的检索方法")
            return
        
        global processing_requests 
            # 使用锁确保修改 processing_requests 是原子的
        async with processing_lock:
            processing_requests += 1
            logger.info(f"待处理检索+1,当前{processing_requests}个")
        
        logger.info(f"使用{searchMethod}方法")
        print(f"使用{searchMethod}方法")
        # 调用同步函数，处理第一个 for 循环
        search_results_list = await asyncio.to_thread(
            process_queries_hybrid, collection_name, queries, customNames, topk,searchMethod
        )
        
        async with processing_lock:
            processing_requests -= 1
            logger.info(f"待处理请求-1,当前{processing_requests}个")
        
        for j in range(len(search_results_list)):
            search_results = search_results_list[j]
                                        
            # 准备图片用于生成器   
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
                response  = AIclient.chat.completions.create(
                    model="qwen-vl-max-2025-08-13", 
                    messages=[
                    {"role":"system","content":[{"type": "text", "text": "You need to combine the image information provided by the user's document page with your own knowledge base to answer the user's query. Your answer should be in English.Your answer needs to be as concise as possible."}]},
                    {
                        "role": "user",
                        "content": base64_images + [{"type": "text", "text": queries[j]}]
                    }
                    ]
                )
                print(response)
            except Exception as e:
                logger.error(f"调用阿里云 API 失败: {str(e)}")
                continue
            
                  
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        return
    
def main():
    return

if __name__ == "__main__":
    main()