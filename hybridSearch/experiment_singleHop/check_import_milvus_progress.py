from milvus_conf import MilvusColbertRetriever, client as milvus_client


# Please ensure that the jobId is obtained through upload_ilvusmulti_img first
def main():
    jobId="461716242494794340"
    collection_name = "tatdqa"
    
    retriever = MilvusColbertRetriever(collection_name = collection_name, milvus_client = milvus_client)
    resp = retriever.search_import_progress(jobId)
    print(len(resp["data"]["details"]))
    
if __name__ == "__main__":
    main()
    
