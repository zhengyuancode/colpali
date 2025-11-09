from milvus_conf import MilvusColbertRetriever, client as milvus_client


# Please ensure that the jobId is obtained through upload_ilvusmulti_img first
def check_job(jobId,collection_name):
    jobId=jobId
    collection_name = collection_name
    
    retriever = MilvusColbertRetriever(collection_name = collection_name, milvus_client = milvus_client)
    resp = retriever.search_import_progress(jobId)
    milvus_client.release_collection(
        collection_name=collection_name
    )
    return resp["data"]["state"]
    
if __name__ == "__main__":
    print(check_job("462024045693351478","admin_text"))
    
