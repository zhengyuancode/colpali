import json
from tqdm import tqdm


def getMultiHopExamples(orgin_path,output_path):
    with open(orgin_path, 'r', encoding='utf-8') as file1:
        data = json.load(file1)
    examples=data["examples"]
    multihopexamples={"examples":[]}
    for item in examples:
        if item["meta_info"]["query_type"] == "multi_hop":
            multihopexamples["examples"].append(item)
    with open(output_path, 'w', encoding='utf-8') as file2:
        json.dump(multihopexamples, file2, indent=4, ensure_ascii=False)
        
# getMultiHopExamples("/root/autodl-tmp/cpdfr-data/cpdfr/hybridSearch/experiment_multiHop/slidevqa_refined.json","/root/autodl-tmp/cpdfr-data/cpdfr/hybridSearch/experiment_multiHop/slidevqa_refined_multiHop.json")

            
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
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
        
#         # 将PDF转换为图片
#         try:
#             images = convert_from_path(pdf_path, dpi=200)
#             for i, image in enumerate(images):
#                 image_path = os.path.join(output_dir, f"{i+1}.png")
#                 image.save(image_path, "PNG")
#         except Exception as e:
#             print(f"Error processing {filename}: {str(e)}")

# print("PDF转换完成。")

# Use a pipeline as a high-level helper
# import torch
# from PIL import Image
# from transformers import AutoModel, AutoTokenizer
# import ast

# model = AutoModel.from_pretrained(
#     'openbmb/MiniCPM-V-4_5-int4', 
#     trust_remote_code=True, # or openbmb/MiniCPM-o-2_6
#     attn_implementation='sdpa', 
#     torch_dtype=torch.float16
#     ) # sdpa or flash_attention_2, no eager
# model = model.eval().to('cuda')
# tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-4_5-int4', trust_remote_code=True) # or openbmb/MiniCPM-o-2_6

# # image = Image.open('/root/autodl-tmp/cpdfr-data/cpdfr/hybridSearch/experiment_multiHop/MMLongBench-Doc/documents/0b85477387a9d0cc33fca0f4becaa0e5/1.png').convert('RGB')


# # First round chat 
# question = f"You are an assistant capable of extracting captions from images or tables.Please perform caption extraction on this image or table, etc.Your answer can only be in english."
# # msgs = [{'role': 'user', 'content': [image, question]}]
# # msgs = [{'role': 'user', 'content': "who are you?"}]


# SEP = "<S-E-P>"
# caption_prompt = f"""You are a preprocessing module for an information retrieval system. Your task is to convert a single document image page—of any type (e.g., report, invoice, academic paper, form, etc.)—into an article described in natural language. Then, use separators({SEP}) to split this article into uniformly sized, semantically coherent text chunks.

# Perform the following steps:

# 1. Comprehensively analyze the entire visible content of the document image.
# 2. Extract and represent all elements faithfully:
#     - Text: Transcribe all visible text verbatim, preserving original wording, line breaks, and logical groupings.
#     - Tables: Convert each table into natural-language descriptions, clearly stating row and column relationships (e.g., 'Row 1: Name – John, Age – 32; Row 2: Name – Lisa, Age – 28').
#     - Mathematical or scientific formulas: Render them accurately using plain-text notation that preserves structure and symbols (e.g., 'E equals m times c squared' or '∫₀¹ x² dx').
#     - Figures, charts, or diagrams: Describe their visual content, key labels, data trends, and spatial context (e.g., 'A bar chart on the right shows sales increasing from Q1 to Q4, with Q4 peaking at $1.2M').
#     - Other: A reasonable description can be in natural language.
# 3. Synthesize all extracted content into fluent, natural-language prose, maintaining logical flow and contextual coherence. Do not add interpretations, summaries, or external knowledge. Then, use separators({SEP}) to split this prose into uniformly sized, semantically coherent text chunks.
# 4. Insert separators({SEP}) into the complete description and return it, where each segmented string (chunk) has approximately the same length (e.g. 150-250 characters or 25-40 words), ensuring semantic boundaries (e.g. avoiding splitting table descriptions into middle rows).
# 5. If there is a number representing the page number, describe this number as Page X.
# 6. Ensure that each block size does not exceed 1000 plain text, unless there are special circumstances such as long tables.
# Example output:
# chunk1 {SEP} chunk2 {SEP} ... {SEP} chunkN
# """
# example_answer = f"""NAVAL MEDICAL RESEARCH AND DEVELOPMENT NEWS Volume IV, Issue 12 December 2012 {SEP} In this issue... CO's Messages 2 Building Afghan Medical Capacity 3 USNS Mercy Pacific Partnership 4 DoD Bone Marrow Donor Program 5 ID Joint Planning Group 6 Capacity Building in Liberia 7 Kazakh Scientists Train at NMRC 8 Patient Condition Occurrence Tool 9 Combat Casualty Research Team 10 Accelerating Technology Transfer 11 NMRC Hosts Dining Out 12 Villasante Speaks at Notre Dame 13 Keane-Myers Speaks at Hopkins 14 Cub Scouts Learn Flag Etiquette 14 NMRC High School Outreach 15 NMRC Officers Teach Science 15 2012 Combined Federal Campaign 16 Ombudsman's Note 16 {SEP} NMRC Hosts Visit from U.S. Global Malaria Coordinator, President's Malaria Initiative SILVER SPRING, Md. - Rear Adm. (Ret.) Tim Ziemer, the U.S. Global Malaria Coordinator, President's Malaria Initiative, visited the Naval Medical Research Center (NMRC), November 29, for a brief on the current malaria vaccine research efforts and to tour the facility. He was interested in learning more about the malaria program at the laboratory. {SEP} Capt. John Sanders, NMRC commanding officer, provided a general overview of the NMRC enterprise with emphasis on the infectious diseases research efforts, specifically in the area of malaria. "The NMRC malaria research program is at the forefront of malaria research worldwide," Sanders pointed out and added, "Researchers here have been investigating methods to control and conquer malaria for more than two decades and have made some exciting discoveries in the last few years." {SEP} Ziemer visited a laboratory focused on investigating the liver stage of infection as a vaccine target, spoke with a researcher about antigen discovery and another researcher on the humanized mouse model developed at NMRC. He also had the opportunity to visit the insectary and hear about clinical immunology and current malaria (Continued on page 14) {SEP} A photograph shows Dr. Xiaoyan "Cathy" Zou, staff scientist from the Henry Jackson Foundation, discussing research on malaria with Rear Adm. (Ret.) Tim Ziemer, the U.S. Global Malaria Coordinator, President, Malaria Initiative. The setting is a laboratory environment with visible equipment including a microscope, biosafety cabinet, and pipettes. Both individuals are standing and engaged in conversation. {SEP} NM&D News is an authorized publication of the Naval Medical Research Center, 503 Robert Grant Avenue, Silver Spring, MD 20910. NM&D News is published monthly by the NMRC Public Affairs Office, 301-319-9378 or svc.pao.nmrc@med.navy.mil. {SEP} Commanding Officer Capt. John W. Sanders Executive Officer Capt. Elizabeth Montcalm-Smith Director for Administration Lt. Cmdr. Nathaniel Smith Public Affairs Officer Doris Ryan Editors Jan Helman Makeda Knott {SEP} http://www.facebook.com/navalmedicalresearchcenter {SEP} Use your smart-phone to access our website! {SEP} Page 1"""

# example_image = Image.open("/root/autodl-tmp/cpdfr-data/cpdfr/hybridSearch/experiment_multiHop/MMLongBench-Doc/documents/0b85477387a9d0cc33fca0f4becaa0e5/1.png").convert('RGB')

# image = Image.open("/root/autodl-tmp/cpdfr-data/cpdfr/hybridSearch/experiment_multiHop/MMLongBench-Doc/documents/0b85477387a9d0cc33fca0f4becaa0e5/5.png").convert('RGB')
# msgs = [
#     {'role': 'user', 'content': [example_image, caption_prompt]},
#     {'role': 'assistant', 'content': [example_answer]},
#     {'role': 'user', 'content': [image, caption_prompt]}
#     ]
# answer = model.chat(
#     msgs=msgs,
#     tokenizer=tokenizer,
#     enable_thinking=False,
#     stream=False
# )
# chunks = answer.split(SEP)
# chunks = [chunk.replace('\n', ' ').replace('\r', ' ') for chunk in chunks]
# for item in chunks:
#     print(item)
#     print("-------")
# chunks = ast.literal_eval(example_answer)
# print(chunks)
# print(len(chunks))

# answer = model.chat(
#     msgs=msgs,
#     tokenizer=tokenizer,
#     enable_thinking=False,
#     stream=False
#     )

# print(answer)

# generated_text = ""
# for new_text in answer:
#     generated_text += new_text
#     print(new_text, flush=True, end='')

# Second round chat, pass history context of multi-turn conversation
# msgs.append({"role": "assistant", "content": [generated_text]})
# msgs.append({"role": "user", "content": ["What should I pay attention to when traveling here?"]})

# answer = model.chat(
#     msgs=msgs,
#     tokenizer=tokenizer,
#     stream=True
# )

# generated_text = ""
# for new_text in answer:
#     generated_text += new_text
#     print(new_text, flush=True, end='')


# from FlagEmbedding import BGEM3FlagModel

# model = BGEM3FlagModel('BAAI/bge-m3',  
#                        use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

# sentences_1 = ["What is BGE M3?"]


# embeddings_1 = model.encode(sentences_1, 
#                             batch_size=12, 
#                             max_length=512,
#                             )['dense_vecs'][0]

# print(embeddings_1)
# print(len(embeddings_1))

# [[0.6265, 0.3477], [0.3499, 0.678 ]]

from transformers import AutoModel
import torch
model = AutoModel.from_pretrained(
    "jinaai/jina-embeddings-v4", 
    trust_remote_code=True, 
    torch_dtype=torch.float16)
model.to("cuda")

# # Encode query
# query_embeddings = model.encode_text(
#     texts=["Overview of climate change impacts on coastal cities"],
#     task="retrieval",
#     prompt_name="query",
# )

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

texts = [
    "غروب جميل على الشاطئ",  # Arabic
    # "海滩上美丽的日落",  # Chinese
    # "Un beau coucher de soleil sur la plage",  # French
    # "Ein wunderschöner Sonnenuntergang am Strand",  # German
    # "Ένα όμορφο ηλιοβασίλεμα πάνω από την παραλία",  # Greek
    # "समुद्र तट पर एक खूबसूरत सूर्यास्त",  # Hindi
    # "Un bellissimo tramonto sulla spiaggia",  # Italian
    # "浜辺に沈む美しい夕日",  # Japanese
    # "해변 위로 아름다운 일몰",  # Korean
]

multivector_embeddings = model.encode_text(
    texts=texts,
    task="retrieval",
    prompt_name="query",
    return_multivector=True,
)

print(multivector_embeddings)

# images = ["/root/autodl-tmp/cpdfr-data/cpdfr/hybridSearch/experiment_multiHop/MMLongBench-Doc/documents/0b85477387a9d0cc33fca0f4becaa0e5/1.png", 
#           "/root/autodl-tmp/cpdfr-data/cpdfr/hybridSearch/experiment_multiHop/MMLongBench-Doc/documents/0b85477387a9d0cc33fca0f4becaa0e5/1.png"]
# multivector_image_embeddings = model.encode_image(
#     images=images,
#     task="retrieval",
#     return_multivector=True,
# )
# print(multivector_image_embeddings)