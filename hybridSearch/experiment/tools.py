import json
import os

def getSingleHopExamples(orgin_path,output_path):
    with open(orgin_path, 'r', encoding='utf-8') as file1:
        data = json.load(file1)
    examples=data["examples"]
    singlehopexamples={"examples":[]}
    for item in examples:
        if item["meta_info"]["query_type"] == "single_hop":
            singlehopexamples["examples"].append(item)
    with open(output_path, 'w', encoding='utf-8') as file2:
        json.dump(singlehopexamples, file2, indent=4, ensure_ascii=False)
        
def getParseList():
    parseDir = "/home/gpu/milvus/backend/colpali/ViDoSeek/minerU_pdf"
    subfolders = {"subfolders":[]}
    # 遍历 parseDir 下的所有条目
    for name in os.listdir(parseDir):
        path = os.path.join(parseDir, name)
        if os.path.isdir(path):  # 判断是否为目录
            subfolders["subfolders"].append({"path":path,"pdfId":name})
    with open("subfolders.json", 'w', encoding='utf-8') as file:
        json.dump(subfolders, file, indent=4, ensure_ascii=False)

def setId():
    with open("subfolders.json", 'r', encoding='utf-8') as file1:
        data = json.load(file1)
    subfolders=data["subfolders"]
    for i in range(len(subfolders)):
        subfolders[i]["sort"] = i+1
    new_subfolders={"subfolders":subfolders}
    with open("subfolders.json", 'w', encoding='utf-8') as file:
        json.dump(new_subfolders, file, indent=4, ensure_ascii=False)
        
# getSingleHopExamples("experiment/vidoseek.json","experiment/vidoseek_singleHop.json")
# getParseList()
setId()
        