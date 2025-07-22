from openai import OpenAI
import json
import base64
import requests
from tqdm import tqdm
import os
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

# 初始化图像描述模型
img_captioning = pipeline(
    task=Tasks.image_captioning,
    model='iic/ofa_image-caption_coco_large_en',
    model_revision='master'
)

def getTextList(block_path,image_dir,output_path):
    # image_path="./hybridSearch/images/4eaaa6f1d3db02e8cc5ce0447fb15c5e5adb65a5e616efa364e858ac074b81e1.jpg"
    # print(getCaption(image_path,"english"))
    
    textList=[]
    with open(block_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
    # 找到最大的 page_idx
    max_page_idx = max(item["page_idx"] for item in data)
    
    # 初始化 textList，每个元素初始为空字符串
    textList = [""] * (max_page_idx + 1)
    
    for item in tqdm(data, desc="Processing items", unit="item"):
        page_idx = item["page_idx"]

        if item["type"] == "text":
            textList[page_idx] += item["text"]
            
        elif item["type"] == "image":
            img_path=image_dir + item["img_path"]
            caption = ""
            if img_path and os.path.isfile(img_path):
                result = img_captioning(img_path)
                # print("\n图像描述："+result[OutputKeys.CAPTION][0])  # 输出图像描述
                caption = result[OutputKeys.CAPTION][0]
            textList[page_idx] += caption if caption is not None else ""
            
        elif item["type"] == "table":
            img_path=image_dir + item["img_path"]
            caption = ""
            if img_path and os.path.isfile(img_path):
                result = img_captioning(img_path)
                # print("\n图像描述："+result[OutputKeys.CAPTION][0])  # 输出图像描述
                caption = result[OutputKeys.CAPTION][0]
            if (str(item["table_caption"]) != "[]"):
                caption += str(item["table_caption"])
            textList[page_idx] += caption if caption is not None else ""
            
        elif item["type"] == "equation":
            img_path=image_dir + item["img_path"]
            caption = ""
            if img_path and os.path.isfile(img_path):
                result = img_captioning(img_path)
                # print("\n图像描述："+result[OutputKeys.CAPTION][0])  # 输出图像描述
                caption = result[OutputKeys.CAPTION][0]
                
            caption += item["text"]
            textList[page_idx] += caption if caption is not None else ""
        else:
            print(item)
            
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(textList, file, indent=2)
        
    return output_path

def get_all_files(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.join(root, file)
            file_list.append(full_path)
    return file_list

def extract_pdf_names(file_paths):
    return [os.path.splitext(os.path.basename(path))[0] for path in file_paths]
    
if __name__ == "__main__":
    language="english"
    pdf_dir = "./pdf"
    pdf_paths = get_all_files(pdf_dir)
    pdf_names = extract_pdf_names(pdf_paths)
    # print(pdf_names)
    for pdf_name in pdf_names:
        block_path=f"./minerU_pdf/{pdf_name}/auto/{pdf_name}_content_list.json"   
        getTextList(block_path,
                    f"./minerU_pdf/{pdf_name}/auto/",
                    f"./minerU_pdf/{pdf_name}/caption_text_list.json")
    
    # block_path="/home/gpu/milvus/backend/colpali/ViDoSeek/testmineru/KAG-Thinker/auto/KAG-Thinker_content_list.json"
    # getTextList(block_path,
    #             f"/home/gpu/milvus/backend/colpali/ViDoSeek/testmineru/KAG-Thinker/auto/",
    #             f"/home/gpu/milvus/backend/colpali/ViDoSeek/testmineru/KAG-Thinker/caption_text_list.json")
        


