from milvus_conf import MilvusColbertRetriever, client as milvus_client


# Please ensure that the jobId is obtained through upload_ilvusmulti_img first
def main():
    jobId="460918093663852799"
    collection_name = "vidore_tatdqa_text"
    
    retriever = MilvusColbertRetriever(collection_name = collection_name, milvus_client = milvus_client)
    resp = retriever.search_import_progress(jobId)
    print(len(resp["data"]["details"]))
    
if __name__ == "__main__":
    main()