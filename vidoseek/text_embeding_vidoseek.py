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