
# Embedding_client = OpenAI(
#             api_key="sk-f78b07615c8a45128d760579e6d42e1f",  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
#             base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
#         )

# class QwenEmbeder:
#     def __init__(self,client):
#         self.client=client

#     def getTextEmbeddings(self,text,dim):
#         completion = self.client.embeddings.create(
#             model="text-embedding-v4",
#             input=text,
#             dimensions=dim, # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
#             encoding_format="float"
#         )
#         return(json.loads(completion.model_dump_json())["data"][0]["embedding"])

import os
from openai import OpenAI
import json
import requests



class QwenEmbeder:
    def __init__(self,url):
        self.url=url
        
    def getTextEmbeddings(self,text):
        payload = {
            "model": "BAAI/bge-m3",
            "input":text,
            "encoding_format":"float"
        }
        headers = {
            "Authorization": "Bearer sk-splwhbmdyruezezpmskzubvdsrvgvufmrnyhcsoedfcgqwoh",
            "Content-Type": "application/json"
        }
        
        response = requests.request("POST", self.url, json=payload, headers=headers).json()
        
        return(response["data"][0]["embedding"])

