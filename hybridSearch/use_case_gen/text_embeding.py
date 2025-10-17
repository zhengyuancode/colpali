import os
from openai import OpenAI
import json
import requests
import numpy as np




class QwenEmbeder:
    def __init__(self,url,model):
        self.url=url
        self.model=model
        
    def getTextEmbeddings(self,text):
        payload = {
            "model": "BAAI/bge-m3",
            "input":text,
            "encoding_format":"float"
        }
        headers = {
            "Authorization": "Bearer sk-xlgtscjecqqgawhkzfzdhqgosmoywwszhufwqoqlmnzgnvxp",
            "Content-Type": "application/json"
        }
        
        response = requests.request("POST", self.url, json=payload, headers=headers).json()
        
        return(response["data"][0]["embedding"])

    def getTextEmbeddingsByLocal(self,text):

        embeddings = self.model.encode([text], 
                                    batch_size=12, 
                                    max_length=512,
                                    )['dense_vecs'][0]

        return embeddings.astype(np.float32)