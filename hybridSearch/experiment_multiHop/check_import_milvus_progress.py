from milvus_conf import MilvusColbertRetriever, client as milvus_client


# Please ensure that the jobId is obtained through upload_ilvusmulti_img first
def main():
    jobId="461130397385547698"
    collection_name = "MMLongDoc"
    
    retriever = MilvusColbertRetriever(collection_name = collection_name, milvus_client = milvus_client)
    resp = retriever.search_import_progress(jobId)
    print(len(resp["data"]["details"]))
    
if __name__ == "__main__":
    main()
    
