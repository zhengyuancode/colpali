from milvus_conf import MilvusColbertRetriever, client as milvus_client
from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

def delete_collection(collection_name):
    return client.drop_collection(collection_name=collection_name)

print(client.list_collections())
# print(delete_collection("colpali"))


# !pip install transformers>=4.52.0 torch>=2.6.0 peft>=0.15.2 torchvision pillow
# !pip install
# from transformers import AutoModel
# import torch

# # Initialize the model
# model_path = "/home/gpu/.cache/huggingface/hub/models--jinaai--jina-embeddings-v4/snapshots/737fa5c46f0262ceba4a462ffa1c5bcf01da416f"
    
# model = AutoModel.from_pretrained(
#     model_path,
#     trust_remote_code=True, 
#     torch_dtype=torch.float16,
#     local_files_only=True
#     )
# model.to("cuda")


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

# chunk_texts = [texts[i:i + 2] for i in range(0, len(texts), 2)]
# print(chunk_texts)
# results = []
# for ct in chunk_texts:
#     with torch.no_grad():
#         multivector_text_embeddings = model.encode_text(
#             texts=ct,
#             task="retrieval",
#             prompt_name="passage",
#             return_multivector=True,
#         )
#     if isinstance(multivector_text_embeddings, torch.Tensor):
#         result = multivector_text_embeddings.float().cpu()
#     elif isinstance(multivector_text_embeddings, (list, tuple)):
#         result = [
#             emb.float().cpu() if isinstance(emb, torch.Tensor) else emb
#             for emb in multivector_text_embeddings
#         ]
#     else:
#         result = multivector_text_embeddings  # 假设已经是 CPU 数据

#     # 显式删除 GPU 张量（虽然可能已被引用清除，但保险起见）
#     del multivector_text_embeddings
#     torch.cuda.empty_cache()  
#     results.extend(result)
#     print(len(result))
#     print(len(result[0].numpy()))
#     print(len(result[0].numpy()[0]))

# print(len(results))
# print(results)

# from openai import OpenAI
# import json
# import ast
# MODELNAME="qwen3-vl-235b-a22b-instruct"
# QWENAPIKEY="sk-f78b07615c8a45128d760579e6d42e1f"
# DMXAPIKEY="sk-gWMA9DJgGb2QzeIa7L7nOvWeXpeESrBAB6SXVflIjnafbonl"

# QWENclient = OpenAI(
#     api_key=QWENAPIKEY,
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     max_retries=5
# )
# completion = QWENclient.chat.completions.create(
#     model="qwen3-max",
#     messages=[
#         {"role": "system", "content": "You can only answer one number"},
#         {"role": "user", "content": "你好"},
#     ],
# )  
# print(json.loads(completion.model_dump_json())["choices"][0]["message"]["content"])




# 对不同类型的答案依据类型来计算检索准确率
# with open("/home/gpu/dzy/M3-CaseRAG/experiment_multiHop/MMLongBench-Doc/data/MMlongDoc.json", 'r', encoding='utf-8') as file:
#     data = json.load(file)
    
# with open("/home/gpu/dzy/M3-CaseRAG/experiment_multiHop/efficiency_results/topkx1/chunk_to_patch_MMLongDoc_1_topkx1_results.json", 'r', encoding='utf-8') as file1:
#     result1 = json.load(file1)

# with open("/home/gpu/dzy/M3-CaseRAG/experiment_multiHop/efficiency_results/topkx1/chunk_to_patch_MMLongDoc_2_topkx1_results.json", 'r', encoding='utf-8') as file2:
#     result2 = json.load(file2)

    
# results = result1["singleHop"] + result2["singleHop"]


# total = 0
# acc = 0
# total_txt = 0
# total_lay = 0
# total_cha = 0
# total_tab = 0
# total_fig = 0
# acc_txt = 0
# acc_lay = 0
# acc_cha = 0
# acc_tab = 0
# acc_fig = 0
# for res in results:
#     total += 1
#     if res["page_judge"] == 1:
#         acc += 1
#     for item in data["examples"]:
#         if item["doc_id"] == res["doc_id"] and item["question"] == res["query"]:
#             evidence_sources = lst = ast.literal_eval(item["evidence_sources"])
#             if "Pure-text (Plain-text)" in evidence_sources:
#                 evidence_sources.remove("Pure-text (Plain-text)")
#                 total_txt += 1
#                 if res["page_judge"] == 1:
#                     acc_txt += 1
#             if "Generalized-text (Layout)" in evidence_sources:
#                 evidence_sources.remove("Generalized-text (Layout)")
#                 total_lay += 1
#                 if res["page_judge"] == 1:
#                     acc_lay += 1
#             if "Chart" in evidence_sources:
#                 evidence_sources.remove("Chart")
#                 total_cha += 1
#                 if res["page_judge"] == 1:
#                     acc_cha += 1
#             if "Table" in evidence_sources:
#                 evidence_sources.remove("Table")
#                 total_tab += 1
#                 if res["page_judge"] == 1:
#                     acc_tab += 1
#             if "Figure" in evidence_sources:
#                 evidence_sources.remove("Figure")
#                 total_fig += 1
#                 if res["page_judge"] == 1:
#                     acc_fig += 1
#             if len(evidence_sources) != 0:
#                 print(f"存在未知的答案类型：{evidence_sources}")
#             break

# statistics = {
#     "all":{
#         "total":total,
#         "acc_txt":acc
#     },
#     "txt":{
#       "total_txt":total_txt,
#       "acc_txt":acc_txt
#     },
#     "lay":{
#       "total_lay":total_lay,
#       "acc_lay":acc_lay
#     },
#     "cha":{
#       "total_cha":total_cha,
#       "acc_cha":acc_cha
#     },
#     "tab":{
#       "total_tab":total_tab,
#       "acc_tab":acc_tab
#     },
#     "fig":{
#       "total_fig":total_fig,
#       "acc_fig":acc_fig
#     },
# }

# with open("/home/gpu/dzy/M3-CaseRAG/experiment_multiHop/efficiency_results/topkx1/chunk_to_patch_MMLongDoc_topkx1_statistics.json", 'w', encoding='utf-8') as f:
#     json.dump(statistics, f, ensure_ascii=False, indent=4) 



# 对不同类型的答案依据类型来计算推理准确率
# with open("/home/gpu/dzy/M3-CaseRAG/experiment_multiHop/MMLongBench-Doc/data/MMlongDoc.json", 'r', encoding='utf-8') as file:
#     data = json.load(file)
    
# with open("/home/gpu/dzy/M3-CaseRAG/experiment_multiHop/qwen3-vl-235b-a22b-instruct/qwen3-vl-235b-a22b-instruct_results.json", 'r', encoding='utf-8') as file1:
#     result1 = json.load(file1)

# # with open("/home/gpu/dzy/M3-CaseRAG/experiment_multiHop/qwen3-vl-235b-a22b-instruct/multi_img_MMLongDoc_2_topkx10_results.json", 'r', encoding='utf-8') as file2:
# #     result2 = json.load(file2)

# # with open("/home/gpu/dzy/M3-CaseRAG/experiment_multiHop/qwen3-vl-235b-a22b-instruct/multi_img_MMLongDoc_3_topkx10_results.json", 'r', encoding='utf-8') as file3:
# #     result3 = json.load(file3)
    
# results = result1["singleHop"]

# def judge_ans(score):
#     if score == "4" or score == "5":
#         return 1
#     elif score == "1" or score == "2" or score == "3":
#         return 0
#     else:
#         print(f"score error:{score}")

# total = 0
# acc = 0
# total_txt = 0
# total_lay = 0
# total_cha = 0
# total_tab = 0
# total_fig = 0
# acc_txt = 0
# acc_lay = 0
# acc_cha = 0
# acc_tab = 0
# acc_fig = 0
# for res in results:
#     if res["answer"] != "qwen api error":
#         total += 1
#         if judge_ans(res["answer_score"]) == 1:
#             acc += 1
#         for item in data["examples"]:
#             if item["doc_id"] == res["doc_id"] and item["question"] == res["query"]:
#                 evidence_sources = ast.literal_eval(item["evidence_sources"])
#                 if "Pure-text (Plain-text)" in evidence_sources:
#                     evidence_sources.remove("Pure-text (Plain-text)")
#                     total_txt += 1
#                     if judge_ans(res["answer_score"]) == 1:
#                         acc_txt += 1
#                 if "Generalized-text (Layout)" in evidence_sources:
#                     evidence_sources.remove("Generalized-text (Layout)")
#                     total_lay += 1
#                     if judge_ans(res["answer_score"]) == 1:
#                         acc_lay += 1
#                 if "Chart" in evidence_sources:
#                     evidence_sources.remove("Chart")
#                     total_cha += 1
#                     if judge_ans(res["answer_score"]) == 1:
#                         acc_cha += 1
#                 if "Table" in evidence_sources:
#                     evidence_sources.remove("Table")
#                     total_tab += 1
#                     if judge_ans(res["answer_score"]) == 1:
#                         acc_tab += 1
#                 if "Figure" in evidence_sources:
#                     evidence_sources.remove("Figure")
#                     total_fig += 1
#                     if judge_ans(res["answer_score"]) == 1:
#                         acc_fig += 1
#                 if len(evidence_sources) != 0:
#                     print(f"存在未知的答案类型：{evidence_sources}")
#                 break

# statistics = {
#     "all":{
#         "total":total,
#         "acc_txt":acc
#     },
#     "txt":{
#       "total_txt":total_txt,
#       "acc_txt":acc_txt
#     },
#     "lay":{
#       "total_lay":total_lay,
#       "acc_lay":acc_lay
#     },
#     "cha":{
#       "total_cha":total_cha,
#       "acc_cha":acc_cha
#     },
#     "tab":{
#       "total_tab":total_tab,
#       "acc_tab":acc_tab
#     },
#     "fig":{
#       "total_fig":total_fig,
#       "acc_fig":acc_fig
#     },
# }

# with open("/home/gpu/dzy/M3-CaseRAG/experiment_multiHop/qwen3-vl-235b-a22b-instruct/qwen3-vl-235b-a22b-instruct_statistics.json", 'w', encoding='utf-8') as f:
#     json.dump(statistics, f, ensure_ascii=False, indent=4)





