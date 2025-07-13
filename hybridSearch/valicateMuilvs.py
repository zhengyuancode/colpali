from pymilvus import connections, Collection

connections.connect(host='localhost', port='19530')  # 根据你的配置调整
collection = Collection("colpali")
print(f"Milvus 中的记录数: {collection.num_entities}")
