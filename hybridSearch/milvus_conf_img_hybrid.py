from pymilvus import MilvusClient, DataType, Function, FunctionType
import numpy as np
import concurrent.futures
from pymilvus import AnnSearchRequest
from pymilvus import RRFRanker
import torch
from reranker import text_rerank
import json

client = MilvusClient(uri="http://127.0.0.1:19530")

class MilvusColbertRetriever:
    def __init__(self, milvus_client, collection_name, dim=128):
        # Initialize the retriever with a Milvus client, collection name, and dimensionality of the vector embeddings.
        # If the collection exists, load it.
        self.collection_name = collection_name
        self.client = milvus_client
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.load_collection(collection_name)
        self.dim = dim

    def create_collection(self):
        # Create a new collection in Milvus for storing embeddings.
        # Drop the existing collection if it already exists and define the schema for the collection.
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)
        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_fields=True,
        )
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)

        schema.add_field(
            field_name="single_image_dense", 
            datatype=DataType.FLOAT_VECTOR, 
            dim=2048, 
            description="single_image_dense"
        )

        schema.add_field(
            field_name="multiple_image_dense", 
            datatype=DataType.FLOAT_VECTOR, 
            dim=128,
            description="multiple_image_dense"
        )
        schema.add_field(field_name="seq_id", datatype=DataType.INT16)
        schema.add_field(field_name="doc_id", datatype=DataType.INT64)
        schema.add_field(field_name="doc", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="customName", datatype=DataType.VARCHAR, max_length=65535)

        
        self.client.create_collection(
            collection_name=self.collection_name, schema=schema
        )

    def create_index(self):
        # Create an index on the vector field to enable fast similarity search.
        # Releases and drops any existing index before creating a new one with specified parameters.
        self.client.release_collection(collection_name=self.collection_name)
        self.client.drop_index(
            collection_name=self.collection_name, index_name="vector"
        )
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="single_image_dense",
            index_name="single_image_dense_index",
            index_type="AUTOINDEX",
            metric_type="IP"
        )

        index_params.add_index(
            field_name="multiple_image_dense",
            index_name="multiple_image_dense_index",
            index_type="HNSW",  # or any other index type you want
            metric_type="IP",  # or the appropriate metric type
            params={
                "M": 16,
                "efConstruction": 500,
            },  # adjust these parameters as needed
        )

        self.client.create_index(
            collection_name=self.collection_name, index_params=index_params, sync=True
        )


    def Img_search(self, data,customNames, topk,doc_id=[]):
        # Perform a vector search on the collection to find the top-k most similar documents.
        # data是一个向量组，这里在进行批量检索
        limit_num = int(topk*1023)
        if (limit_num >= 16384):
            limit_num = 16383
        try: 
            if(doc_id != []):
                results = self.client.search(
                    self.collection_name,
                    data,
                    limit=limit_num,
                    anns_field="multiple_image_dense",
                    filter=f'customName in {customNames} and doc_id in {doc_id}',
                    output_fields=["multiple_image_dense", "seq_id", "doc_id","customName"],
                    search_params={"metric_type": "IP"}
                )
            else:
                results = self.client.search(
                    self.collection_name,
                    data,
                    limit=limit_num,
                    anns_field="multiple_image_dense",
                    filter=f'customName in {customNames}',
                    output_fields=["multiple_image_dense", "seq_id", "doc_id","customName"],
                    search_params={"metric_type": "IP"}
                )
        except Exception as e:
            print("colpali检索失败：\n")
            print(str(e))
                 
        docs = []
        seen = set()  # 用于跟踪已见的唯一标识

        for r_id in range(len(results)):
            for r in range(len(results[r_id])):
                # 提取文档的唯一标识组合
                doc_id = results[r_id][r]["entity"]["doc_id"]
                custom_name = results[r_id][r]["entity"]["customName"]
                unique_key = (doc_id, custom_name)  # 创建不可变的唯一键
                
                # 仅当未出现过时才添加到列表
                if unique_key not in seen:
                    seen.add(unique_key)
                    docs.append({
                        "doc_id": doc_id,
                        "customName": custom_name
                    })
        scores = []
        def rerank_single_doc(doc, data, client, collection_name):
            # Rerank a single document by retrieving its embeddings and calculating the similarity with the query.
            doc_id = doc["doc_id"]
            customName = doc["customName"]
            doc_colbert_vecs = client.query(
                collection_name=collection_name,
                filter=f'doc_id == {doc_id} and customName == "{customName}"',
                output_fields=["seq_id", "multiple_image_dense", "doc"]
            )
            doc_vecs = np.vstack(
                [doc_colbert_vecs[i]["multiple_image_dense"] for i in range(len(doc_colbert_vecs))]
            )
            score = np.dot(data, doc_vecs.T).max(1).sum()
            docPath=""
            for item in doc_colbert_vecs:
                if item["seq_id"] == 0:
                    docPath = item["doc"]
                    break 
            return (score, doc_id, docPath)

        with concurrent.futures.ThreadPoolExecutor(max_workers=300) as executor:
            futures = {
                executor.submit(
                    rerank_single_doc, doc, data, client, self.collection_name
                ): doc
                for doc in docs
            }
            for future in concurrent.futures.as_completed(futures):
                score, doc_id, doc = future.result()
                scores.append((score, doc_id, doc))
 
        scores.sort(key=lambda x: x[0], reverse=True)
        # return scores
        if len(scores) >= topk:
            return scores[:topk]
        else:
            return scores
        
    
    
    # 单图片向量与多向量组向量的多阶段混合检索
    def Muti_hybrid_search_multiple_in_single(self,query_param, topk, rerank_topn=50):
        customNames = query_param["customNames"]
        
        count = self.count_entity_customNames(customNames)
        if(count >= rerank_topn*2):
            rerank_topn = rerank_topn
        elif(count < rerank_topn*2 and count > topk*2):
            rerank_topn = count//2
        elif(count < topk*2 and count >= topk):
            rerank_topn = topk
        else:
            topk = count
            rerank_topn =count
        print(f"topk:{topk}\nrerank_topn:{rerank_topn}\ncount:{count}\n")       
        
        res = client.search(
            collection_name=self.collection_name,
            anns_field="single_image_dense",
            data=[query_param["single_img_qs"]],
            limit=int(rerank_topn),
            search_params={"nprobe": 10,"metric_type": "IP"},
            filter=f"seq_id == 0 and customName in {customNames}",
            output_fields=["doc","doc_id"]
        )
        
        doc_id=[]
        for resItem in res:
            for item in resItem:
                doc_id.append(item["doc_id"])
        request_3 = self.Img_search(query_param["image_query"],customNames,topk,doc_id)
        search_output = []
        for sitem in request_3:
            search_output.append(sitem[2])
        
        return search_output
        
    def Muti_hybrid_search_single_in_multiple(self,query_param, topk, rerank_topn=50):
        customNames = query_param["customNames"]
        
        count = self.count_entity_customNames(customNames)
        if(count >= rerank_topn*2):
            rerank_topn = rerank_topn
        elif(count < rerank_topn*2 and count > topk*2):
            rerank_topn = count//2
        elif(count < topk*2 and count >= topk):
            rerank_topn = topk
        else:
            topk = count
            rerank_topn =count
        print(f"topk:{topk}\nrerank_topn:{rerank_topn}\ncount:{count}\n")        
        request_3 = self.Img_search(query_param["image_query"],customNames,rerank_topn)
        doc = []
        for sitem in request_3:
            doc.append(sitem[2])
            
        res = client.search(
            collection_name=self.collection_name,
            anns_field="single_image_dense",
            data=[query_param["single_img_qs"]],
            limit=topk,
            search_params={"nprobe": 10,"metric_type": "IP"},
            filter=f"seq_id == 0 and customName in {customNames} and doc in {doc}",
            output_fields=["doc"]
        )
        
        search_output=[]
        for resItem in res:
            for item in resItem:
                search_output.append(item["doc"])
                     
        return search_output

    def insert(self, data):
        multiple_image_dense = data["multiple_image_dense"]
        seq_length = len(multiple_image_dense)

       
        self.client.insert(
            self.collection_name,
            [
                {
                    "multiple_image_dense": multiple_image_dense[i],
                    "seq_id": i,
                    "doc_id": data["doc_id"],
                    "doc": data["filepath"] if i == 0 else "",
                    "customName": data["customName"],
                    "single_image_dense": data["single_image_dense"] if i == 0 else ([0.0] * 2048)
                }
                for i in range(seq_length)
            ],
        )
        
    def delete_entity(self,customName):
        res = client.delete(
            collection_name=self.collection_name,
            filter=f"customName in ['{customName}']"
        )
        print(res)
        
    def delete_entity_all(self):
        client.drop_collection(collection_name=self.collection_name)
        
    def search_all_customName(self):
        res = client.query(
            collection_name=self.collection_name,
            filter="seq_id == 0 and doc_id == 0",
            output_fields=["customName"]
        )
        return res
    
    
    def count_entity(self):
        res = client.query(
            collection_name=self.collection_name,
            output_fields=["count(*)"]
        )
        return res
    
    def count_entity_customNames(self,customNames):
        res = client.query(
            collection_name=self.collection_name,
            filter=f"seq_id == 0 and customName in {customNames}",
            output_fields=["count(*)"]
        )
        count_value = res[0]['count(*)']
        return count_value