import dashscope
from http import HTTPStatus

dashscope.api_key = "sk-f78b07615c8a45128d760579e6d42e1f"
def text_rerank(documents,query,topk):
    resp = dashscope.TextReRank.call(
        model="gte-rerank-v2",
        query=query,
        documents=documents,
        top_n=topk,
        return_documents=False
    )
    if resp.status_code == HTTPStatus.OK:
        return(resp)
    else:
        return(resp)


if __name__ == '__main__':
    text_rerank()