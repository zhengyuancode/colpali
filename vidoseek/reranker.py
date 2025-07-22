import requests
url = "https://api.siliconflow.cn/v1/rerank"

def text_rerank(documents,query,topk):
    
    payload = {
        "model": "BAAI/bge-reranker-v2-m3",
        "query": query,
        "documents": documents,
        "top_n": topk,
        "return_documents":False
    }
    headers = {
        "Authorization": "Bearer sk-splwhbmdyruezezpmskzubvdsrvgvufmrnyhcsoedfcgqwoh",
        "Content-Type": "application/json"
    }
    
    response = requests.request("POST", url, json=payload, headers=headers).json()
    return response["results"]


if __name__ == '__main__':
    print(text_rerank(["abananapple", "banana", "fruit", "apple"],"Apple",2))