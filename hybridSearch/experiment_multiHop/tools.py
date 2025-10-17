from milvus_conf import MilvusColbertRetriever, client as milvus_client

            
# from pdf2image import convert_from_path
# import os
# from tqdm import tqdm

# # 指定PDF文件所在的目录
# pdf_dir = "/root/autodl-tmp/cpdfr-data/cpdfr/hybridSearch/experiment_multiHop/MMLongBench-Doc/documents"

# # 遍历目录下的所有文件
# for filename in tqdm(os.listdir(pdf_dir), desc="Processing PDFs",total=len(os.listdir(pdf_dir))):
#     # 只处理PDF文件
#     if filename.lower().endswith('.pdf'):
#         # 获取PDF文件的完整路径
#         pdf_path = os.path.join(pdf_dir, filename)
        
#         # 创建子文件夹，以PDF文件名命名（不带扩展名）
#         pdf_name = os.path.splitext(filename)[0]
#         output_dir = os.path.join(pdf_dir, pdf_name)
        
#         # 如果子文件夹不存在，则创建
#         # if not os.path.exists(output_dir):
#         #     os.makedirs(output_dir)
        
#         # 将PDF转换为图片
#         # try:
#         #     images = convert_from_path(pdf_path, dpi=200)
#         #     for i, image in enumerate(images):
#         #         image_path = os.path.join(output_dir, f"{i+1}.png")
#         #         image.save(image_path, "PNG")
#         # except Exception as e:
#         #     print(f"Error processing {filename}: {str(e)}")
#         chunk_path = output_dir + "/chunk.json"
#         if os.path.exists(chunk_path):
#             os.remove(chunk_path)
#         else:
#             print("文件不存在，无需删除。")
# print("PDF转换完成。")

# from transformers import AutoModel
# import torch
# model = AutoModel.from_pretrained(
#     "jinaai/jina-embeddings-v4", 
#     trust_remote_code=True, 
#     torch_dtype=torch.float16)
# model.to("cuda")

# # Encode query
# query_embeddings = model.encode_text(
#     texts=["Overview of climate change impacts on coastal cities"],
#     task="retrieval",
#     prompt_name="query"
# )
# output = query_embeddings[0].float().cpu().numpy()
# print(output)
# print(len(output))

# # Encode passage (text)
# passage_embeddings = model.encode_text(
#     texts=[
#         "Climate change has led to rising sea levels, increased frequency of extreme weather events..."
#     ],
#     task="retrieval",
#     prompt_name="passage",
# )

# # Encode image/document
# image_embeddings = model.encode_image(
#     images=["https://i.ibb.co/nQNGqL0/beach1.jpg"],
#     task="retrieval",
# )

# ========================
# Use multivectors
# ========================

# texts = [
#     # "غروب جميل على الشاطئ"  # Arabic
#     "海滩上美丽的日落"  # Chinese
#     # "Un beau coucher de soleil sur la plage",  # French
#     # "Ein wunderschöner Sonnenuntergang am Strand",  # German
#     # "Ένα όμορφο ηλιοβασίλεμα πάνω από την παραλία",  # Greek
#     # "समुद्र तट पर एक खूबसूरत सूर्यास्त",  # Hindi
#     # "Un bellissimo tramonto sulla spiaggia",  # Italian
#     # "浜辺に沈む美しい夕日",  # Japanese
#     # "해변 위로 아름다운 일몰",  # Korean
# ]

# multivector_embeddings = model.encode_text(
#     texts=texts,
#     task="retrieval",
#     prompt_name="query",
#     return_multivector=True,
# )[0].float().cpu().numpy()

# print(len(multivector_embeddings))
# print(len(multivector_embeddings[0]))

# images = ["/root/autodl-tmp/cpdfr-data/cpdfr/hybridSearch/experiment_multiHop/MMLongBench-Doc/documents/0b85477387a9d0cc33fca0f4becaa0e5/1.png", 
#           "/root/autodl-tmp/cpdfr-data/cpdfr/hybridSearch/experiment_multiHop/MMLongBench-Doc/documents/0b85477387a9d0cc33fca0f4becaa0e5/1.png"]
# multivector_image_embeddings = model.encode_image(
#     images=images,
#     task="retrieval",
#     return_multivector=True,
# )
# print(multivector_image_embeddings)

# import pandas as pd
# from pathlib import Path
# import json
# import ast

# # 读取整个文件到内存（适合中小数据集）
# df = pd.read_parquet('/root/autodl-tmp/cpdfr-data/cpdfr/hybridSearch/experiment_multiHop/MMLongBench-Doc/data/train-00000-of-00001.parquet')


# MMlongDoc = {"examples": []}
# for index, row in df.iterrows():
#     MMlongDoc["examples"].append({
#         "doc_id": str(Path(row['doc_id']).stem),
#         "question": row['question'],
#         "answer": row['answer'],
#         "evidence_pages": ast.literal_eval(row['evidence_pages'])
#     })
    
# with open("/root/autodl-tmp/cpdfr-data/cpdfr/hybridSearch/experiment_multiHop/MMLongBench-Doc/data/MMlongDoc.json", 'w', encoding='utf-8') as file:
#     json.dump(MMlongDoc, file, indent=4, ensure_ascii=False)

# from mem0 import Memory
# memory = Memory()

# # For a user
# messages = [
#     {
#         "role": "user",
#         "content": "I like to drink coffee in the morning and go for a walk"
#     }
# ]
# result = memory.add(messages, user_id="default", metadata={"category": "preferences"})

retriever = MilvusColbertRetriever(collection_name="MMLongDoc", milvus_client=milvus_client)
print(retriever.count_page_by_file(["0b85477387a9d0cc33fca0f4becaa0e5"]))