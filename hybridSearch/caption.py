from openai import OpenAI
import json
import base64
from tqdm import tqdm
import os

# QWENAPIKEY="sk-f78b07615c8a45128d760579e6d42e1f"

# AIclient = OpenAI(
#     # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
#     api_key=QWENAPIKEY,
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )

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
                            {"type": "text", "text": f"Please perform caption extraction on this image or table.Your answer can only be in {language}."},
                        ]
                }
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"调用阿里云 API 失败: {str(e)}")
        return ""



def getTextList(block_path,language,image_dir,output_path,client):
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
                caption += getCaption(image_dir + item["img_path"], language,client)
            textList[page_idx] += caption if caption is not None else ""
            
        elif item["type"] == "table":
            img_path=image_dir + item["img_path"]
            caption = ""
            if img_path and os.path.isfile(img_path):
                caption += getCaption(image_dir + item["img_path"], language,client)
            if (str(item["table_caption"]) != "[]"):
                caption += str(item["table_caption"])
            textList[page_idx] += caption if caption is not None else ""
            
        elif item["type"] == "equation":
            img_path=image_dir + item["img_path"]
            caption = ""
            if img_path and os.path.isfile(img_path):
                caption += getCaption(image_dir + item["img_path"], language,client)
            caption += item["text"]
            textList[page_idx] += caption if caption is not None else ""    
            
        else:
            print(item)
            
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(textList, file, indent=2)
        
    return output_path
    
if __name__ == "__main__":
    language="english"
    block_path="hybridSearch\\66b6191f-78ac-4132-b182-3f490cf1b63a_content_list.json"
    # getTextList(block_path,language)