# !pip install transformers>=4.52.0 torch>=2.6.0 peft>=0.15.2 torchvision pillow
# !pip install
from transformers import AutoModel
import torch
import numpy as np

# Initialize the model
model_name = "/home/gpu/milvus/backend/colpali/modelcache/models--jinaai--jina-embeddings-v4/snapshots/50cb06ee0b17a7257c8caf4417c2a7596eb7e5d2"
cachedir = "/home/gpu/milvus/backend/colpali/modelcache/"
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    cache_dir=cachedir,                # 指定缓存路径
    local_files_only=True,              # 强制离线加载
)


model.to("cuda")

# ========================
# 1. Retrieval Task
# ========================
# Configure truncate_dim, max_length (for texts), max_pixels (for images), vector_type, batch_size in the encode function if needed

# Encode query
# query_embeddings = model.encode_text(
#     texts=["Overview of climate change impacts on coastal cities"],
#     task="retrieval",
#     prompt_name="query",
# )

# Encode passage (text)
# passage_embeddings = model.encode_text(
#     texts=[
#         "Climate change has led to rising sea levels, increased frequency of extreme weather events..."
#     ],
#     task="retrieval",
#     prompt_name="passage",
# )

# Encode image/document，single dense vector dim:2048，miltivector vector dim:747*128
image_embeddings = model.encode_image(
    images=["/home/gpu/milvus/backend/colpali/pdfs/dzy/c8ecadbb-f3b9-4d28-97e8-fb70f01223f4/pages/page_2.png"],
    task="retrieval",
    return_multivector=True
)


print(len(image_embeddings))
for item in image_embeddings:
    print(len(item))
    print(len(item[0]))
        
# ========================
# 2. Text Matching Task
# ========================
# texts = [
#     "غروب جميل على الشاطئ",  # Arabic
#     "海滩上美丽的日落",  # Chinese
#     "Un beau coucher de soleil sur la plage",  # French
#     "Ein wunderschöner Sonnenuntergang am Strand",  # German
#     "Ένα όμορφο ηλιοβασίλεμα πάνω από την παραλία",  # Greek
#     "समुद्र तट पर एक खूबसूरत सूर्यास्त",  # Hindi
#     "Un bellissimo tramonto sulla spiaggia",  # Italian
#     "浜辺に沈む美しい夕日",  # Japanese
#     "해변 위로 아름다운 일몰",  # Korean
# ]

# text_embeddings = model.encode_text(texts=texts, task="text-matching")

# ========================
# 3. Code Understanding Task
# ========================

# Encode query
# query_embedding = model.encode_text(
#     texts=["Find a function that prints a greeting message to the console"],
#     task="code",
#     prompt_name="query",
# )

# # Encode code
# code_embeddings = model.encode_text(
#     texts=["def hello_world():\n    print('Hello, World!')"],
#     task="code",
#     prompt_name="passage",
# )

# ========================
# 4. Use multivectors
# ========================

# multivector_embeddings = model.encode_text(
#     texts=texts,
#     task="retrieval",
#     prompt_name="query",
#     return_multivector=True,
# )

# images = ["https://i.ibb.co/nQNGqL0/beach1.jpg", "https://i.ibb.co/r5w8hG8/beach2.jpg"]
# multivector_image_embeddings = model.encode_image(
#     images=images,
#     task="retrieval",
#     return_multivector=True,
# )
