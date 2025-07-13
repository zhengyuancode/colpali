import os
from openai import OpenAI
import json

Embedding_client = OpenAI(
            api_key="sk-f78b07615c8a45128d760579e6d42e1f",  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
        )

class QwenEmbeder:
    def __init__(self,client):
        self.client=client

    def getTextEmbeddings(self,text,dim):
        completion = self.client.embeddings.create(
            model="text-embedding-v4",
            input=text,
            dimensions=dim, # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
            encoding_format="float"
        )
        return(json.loads(completion.model_dump_json())["data"][0]["embedding"])

