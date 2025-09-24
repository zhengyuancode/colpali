from milvus_conf import MilvusColbertRetriever, client as milvus_client
from minio import Minio
from minio.error import S3Error


# This is the main function for batch insertion. Please ensure that 'preprocess_minio_multi_img.py' has been executed first
def main(prefix,collection_name):
    # Specify bucket and prefix
    bucket_name = "a-bucket"
    prefix = prefix
    parquet_files = []
    collection_name = collection_name
    
    
    # Initialize MinIO client
    minio_client = Minio(
        "localhost:9000",  
        access_key="minioadmin",  
        secret_key="minioadmin", 
        secure=False
    )
    
    
    try:
        objects = minio_client.list_objects(bucket_name, prefix=prefix, recursive=True)
        for obj in objects:
            if obj.object_name.endswith(".parquet"):
                parquet_files.append([obj.object_name])
    except S3Error as e:
        print(f"MinIO error: {e}")
    
    retriever = MilvusColbertRetriever(collection_name = collection_name, milvus_client=milvus_client)
    retriever.bulk_minio_insert_milvus(collection_name,parquet_files)
    
if __name__ == "__main__":
    main(prefix="vidore_tatdqa_text/954605b6-77bb-46d7-997c-aacf7bd0b3aa/",collection_name="vidore_tatdqa_text")
    