import json
import os
from milvus_conf_hybrid import MilvusColbertRetriever, client
from pathlib import Path

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
        
def createVidoseekCollection():
    # 初始化Milvus
    if(client.has_collection(collection_name="vidoseek")):
        # logger.info("已存在该向量数据库")
        print("已存在vidoseek向量数据库")
    else:
        retriever = MilvusColbertRetriever(collection_name="vidoseek", milvus_client=client)
        retriever.create_collection()
        retriever.create_index()
        print("已创建vidoseek向量数据库")
        

# getSingleHopExamples("experiment/vidoseek.json","experiment/vidoseek_singleHop.json")
# getParseList()
# setId()
# createVidoseekCollection()
        


# retriever = MilvusColbertRetriever(collection_name="vidoseek", milvus_client=client)
# 上传本地的向量数据到minio
# remote_files = retriever.bulk_LocalData_upload("/home/gpu/milvus/backend/colpali/ViDoSeek/bulkInsert","c494711a-1dbe-43f2-9d18-994eb651957d")

#将minio的向量数据插入milvus
# remote_files=['milvus_bulkinsert/parquet/120.parquet', 'milvus_bulkinsert/parquet/4.parquet', 'milvus_bulkinsert/parquet/112.parquet', 'milvus_bulkinsert/parquet/195.parquet', 'milvus_bulkinsert/parquet/14.parquet', 'milvus_bulkinsert/parquet/255.parquet', 'milvus_bulkinsert/parquet/154.parquet', 'milvus_bulkinsert/parquet/83.parquet', 'milvus_bulkinsert/parquet/41.parquet', 'milvus_bulkinsert/parquet/81.parquet', 'milvus_bulkinsert/parquet/110.parquet', 'milvus_bulkinsert/parquet/172.parquet', 'milvus_bulkinsert/parquet/132.parquet', 'milvus_bulkinsert/parquet/202.parquet', 'milvus_bulkinsert/parquet/133.parquet', 'milvus_bulkinsert/parquet/36.parquet', 'milvus_bulkinsert/parquet/119.parquet', 'milvus_bulkinsert/parquet/129.parquet', 'milvus_bulkinsert/parquet/272.parquet', 'milvus_bulkinsert/parquet/68.parquet', 'milvus_bulkinsert/parquet/262.parquet', 'milvus_bulkinsert/parquet/213.parquet', 'milvus_bulkinsert/parquet/20.parquet', 'milvus_bulkinsert/parquet/194.parquet', 'milvus_bulkinsert/parquet/54.parquet', 'milvus_bulkinsert/parquet/173.parquet', 'milvus_bulkinsert/parquet/178.parquet', 'milvus_bulkinsert/parquet/206.parquet', 'milvus_bulkinsert/parquet/284.parquet', 'milvus_bulkinsert/parquet/165.parquet', 'milvus_bulkinsert/parquet/59.parquet', 'milvus_bulkinsert/parquet/200.parquet', 'milvus_bulkinsert/parquet/106.parquet', 'milvus_bulkinsert/parquet/146.parquet', 'milvus_bulkinsert/parquet/45.parquet', 'milvus_bulkinsert/parquet/46.parquet', 'milvus_bulkinsert/parquet/261.parquet', 'milvus_bulkinsert/parquet/91.parquet', 'milvus_bulkinsert/parquet/124.parquet', 'milvus_bulkinsert/parquet/103.parquet', 'milvus_bulkinsert/parquet/15.parquet', 'milvus_bulkinsert/parquet/234.parquet', 'milvus_bulkinsert/parquet/113.parquet', 'milvus_bulkinsert/parquet/221.parquet', 'milvus_bulkinsert/parquet/55.parquet', 'milvus_bulkinsert/parquet/208.parquet', 'milvus_bulkinsert/parquet/184.parquet', 'milvus_bulkinsert/parquet/252.parquet', 'milvus_bulkinsert/parquet/84.parquet', 'milvus_bulkinsert/parquet/155.parquet', 'milvus_bulkinsert/parquet/283.parquet', 'milvus_bulkinsert/parquet/48.parquet', 'milvus_bulkinsert/parquet/16.parquet', 'milvus_bulkinsert/parquet/57.parquet', 'milvus_bulkinsert/parquet/69.parquet', 'milvus_bulkinsert/parquet/160.parquet', 'milvus_bulkinsert/parquet/90.parquet', 'milvus_bulkinsert/parquet/188.parquet', 'milvus_bulkinsert/parquet/201.parquet', 'milvus_bulkinsert/parquet/214.parquet', 'milvus_bulkinsert/parquet/27.parquet', 'milvus_bulkinsert/parquet/131.parquet', 'milvus_bulkinsert/parquet/243.parquet', 'milvus_bulkinsert/parquet/190.parquet', 'milvus_bulkinsert/parquet/71.parquet', 'milvus_bulkinsert/parquet/76.parquet', 'milvus_bulkinsert/parquet/273.parquet', 'milvus_bulkinsert/parquet/196.parquet', 'milvus_bulkinsert/parquet/97.parquet', 'milvus_bulkinsert/parquet/179.parquet', 'milvus_bulkinsert/parquet/122.parquet', 'milvus_bulkinsert/parquet/288.parquet', 'milvus_bulkinsert/parquet/171.parquet', 'milvus_bulkinsert/parquet/29.parquet', 'milvus_bulkinsert/parquet/225.parquet', 'milvus_bulkinsert/parquet/138.parquet', 'milvus_bulkinsert/parquet/9.parquet', 'milvus_bulkinsert/parquet/87.parquet', 'milvus_bulkinsert/parquet/17.parquet', 'milvus_bulkinsert/parquet/64.parquet', 'milvus_bulkinsert/parquet/108.parquet', 'milvus_bulkinsert/parquet/192.parquet', 'milvus_bulkinsert/parquet/157.parquet', 'milvus_bulkinsert/parquet/43.parquet', 'milvus_bulkinsert/parquet/231.parquet', 'milvus_bulkinsert/parquet/217.parquet', 'milvus_bulkinsert/parquet/118.parquet', 'milvus_bulkinsert/parquet/47.parquet', 'milvus_bulkinsert/parquet/24.parquet', 'milvus_bulkinsert/parquet/265.parquet', 'milvus_bulkinsert/parquet/286.parquet', 'milvus_bulkinsert/parquet/250.parquet', 'milvus_bulkinsert/parquet/203.parquet', 'milvus_bulkinsert/parquet/215.parquet', 'milvus_bulkinsert/parquet/238.parquet', 'milvus_bulkinsert/parquet/79.parquet', 'milvus_bulkinsert/parquet/140.parquet', 'milvus_bulkinsert/parquet/166.parquet', 'milvus_bulkinsert/parquet/7.parquet', 'milvus_bulkinsert/parquet/244.parquet', 'milvus_bulkinsert/parquet/219.parquet', 'milvus_bulkinsert/parquet/189.parquet', 'milvus_bulkinsert/parquet/209.parquet', 'milvus_bulkinsert/parquet/267.parquet', 'milvus_bulkinsert/parquet/77.parquet', 'milvus_bulkinsert/parquet/187.parquet', 'milvus_bulkinsert/parquet/167.parquet', 'milvus_bulkinsert/parquet/181.parquet', 'milvus_bulkinsert/parquet/142.parquet', 'milvus_bulkinsert/parquet/12.parquet', 'milvus_bulkinsert/parquet/139.parquet', 'milvus_bulkinsert/parquet/182.parquet', 'milvus_bulkinsert/parquet/247.parquet', 'milvus_bulkinsert/parquet/100.parquet', 'milvus_bulkinsert/parquet/148.parquet', 'milvus_bulkinsert/parquet/137.parquet', 'milvus_bulkinsert/parquet/96.parquet', 'milvus_bulkinsert/parquet/75.parquet', 'milvus_bulkinsert/parquet/10.parquet', 'milvus_bulkinsert/parquet/281.parquet', 'milvus_bulkinsert/parquet/66.parquet', 'milvus_bulkinsert/parquet/35.parquet', 'milvus_bulkinsert/parquet/161.parquet', 'milvus_bulkinsert/parquet/275.parquet', 'milvus_bulkinsert/parquet/193.parquet', 'milvus_bulkinsert/parquet/88.parquet', 'milvus_bulkinsert/parquet/248.parquet', 'milvus_bulkinsert/parquet/21.parquet', 'milvus_bulkinsert/parquet/177.parquet', 'milvus_bulkinsert/parquet/197.parquet', 'milvus_bulkinsert/parquet/141.parquet', 'milvus_bulkinsert/parquet/147.parquet', 'milvus_bulkinsert/parquet/156.parquet', 'milvus_bulkinsert/parquet/61.parquet', 'milvus_bulkinsert/parquet/291.parquet', 'milvus_bulkinsert/parquet/254.parquet', 'milvus_bulkinsert/parquet/174.parquet', 'milvus_bulkinsert/parquet/26.parquet', 'milvus_bulkinsert/parquet/180.parquet', 'milvus_bulkinsert/parquet/292.parquet', 'milvus_bulkinsert/parquet/223.parquet', 'milvus_bulkinsert/parquet/279.parquet', 'milvus_bulkinsert/parquet/101.parquet', 'milvus_bulkinsert/parquet/150.parquet', 'milvus_bulkinsert/parquet/117.parquet', 'milvus_bulkinsert/parquet/70.parquet', 'milvus_bulkinsert/parquet/224.parquet', 'milvus_bulkinsert/parquet/218.parquet', 'milvus_bulkinsert/parquet/65.parquet', 'milvus_bulkinsert/parquet/18.parquet', 'milvus_bulkinsert/parquet/126.parquet', 'milvus_bulkinsert/parquet/257.parquet', 'milvus_bulkinsert/parquet/3.parquet', 'milvus_bulkinsert/parquet/25.parquet', 'milvus_bulkinsert/parquet/285.parquet', 'milvus_bulkinsert/parquet/229.parquet', 'milvus_bulkinsert/parquet/8.parquet', 'milvus_bulkinsert/parquet/1.parquet', 'milvus_bulkinsert/parquet/80.parquet', 'milvus_bulkinsert/parquet/258.parquet', 'milvus_bulkinsert/parquet/276.parquet', 'milvus_bulkinsert/parquet/158.parquet', 'milvus_bulkinsert/parquet/269.parquet', 'milvus_bulkinsert/parquet/230.parquet', 'milvus_bulkinsert/parquet/93.parquet', 'milvus_bulkinsert/parquet/123.parquet', 'milvus_bulkinsert/parquet/32.parquet', 'milvus_bulkinsert/parquet/63.parquet', 'milvus_bulkinsert/parquet/51.parquet', 'milvus_bulkinsert/parquet/58.parquet', 'milvus_bulkinsert/parquet/170.parquet', 'milvus_bulkinsert/parquet/164.parquet', 'milvus_bulkinsert/parquet/259.parquet', 'milvus_bulkinsert/parquet/99.parquet', 'milvus_bulkinsert/parquet/159.parquet', 'milvus_bulkinsert/parquet/211.parquet', 'milvus_bulkinsert/parquet/114.parquet', 'milvus_bulkinsert/parquet/149.parquet', 'milvus_bulkinsert/parquet/19.parquet', 'milvus_bulkinsert/parquet/95.parquet', 'milvus_bulkinsert/parquet/239.parquet', 'milvus_bulkinsert/parquet/23.parquet', 'milvus_bulkinsert/parquet/121.parquet', 'milvus_bulkinsert/parquet/186.parquet', 'milvus_bulkinsert/parquet/271.parquet', 'milvus_bulkinsert/parquet/28.parquet', 'milvus_bulkinsert/parquet/85.parquet', 'milvus_bulkinsert/parquet/89.parquet', 'milvus_bulkinsert/parquet/111.parquet', 'milvus_bulkinsert/parquet/204.parquet', 'milvus_bulkinsert/parquet/280.parquet', 'milvus_bulkinsert/parquet/241.parquet', 'milvus_bulkinsert/parquet/183.parquet', 'milvus_bulkinsert/parquet/50.parquet', 'milvus_bulkinsert/parquet/30.parquet', 'milvus_bulkinsert/parquet/287.parquet', 'milvus_bulkinsert/parquet/13.parquet', 'milvus_bulkinsert/parquet/228.parquet', 'milvus_bulkinsert/parquet/216.parquet', 'milvus_bulkinsert/parquet/162.parquet', 'milvus_bulkinsert/parquet/163.parquet', 'milvus_bulkinsert/parquet/92.parquet', 'milvus_bulkinsert/parquet/5.parquet', 'milvus_bulkinsert/parquet/125.parquet', 'milvus_bulkinsert/parquet/98.parquet', 'milvus_bulkinsert/parquet/31.parquet', 'milvus_bulkinsert/parquet/153.parquet', 'milvus_bulkinsert/parquet/263.parquet', 'milvus_bulkinsert/parquet/253.parquet', 'milvus_bulkinsert/parquet/246.parquet', 'milvus_bulkinsert/parquet/73.parquet', 'milvus_bulkinsert/parquet/102.parquet', 'milvus_bulkinsert/parquet/38.parquet', 'milvus_bulkinsert/parquet/82.parquet', 'milvus_bulkinsert/parquet/220.parquet', 'milvus_bulkinsert/parquet/249.parquet', 'milvus_bulkinsert/parquet/282.parquet', 'milvus_bulkinsert/parquet/94.parquet', 'milvus_bulkinsert/parquet/233.parquet', 'milvus_bulkinsert/parquet/205.parquet', 'milvus_bulkinsert/parquet/60.parquet', 'milvus_bulkinsert/parquet/134.parquet', 'milvus_bulkinsert/parquet/242.parquet', 'milvus_bulkinsert/parquet/22.parquet', 'milvus_bulkinsert/parquet/210.parquet', 'milvus_bulkinsert/parquet/268.parquet', 'milvus_bulkinsert/parquet/212.parquet', 'milvus_bulkinsert/parquet/11.parquet', 'milvus_bulkinsert/parquet/266.parquet', 'milvus_bulkinsert/parquet/289.parquet', 'milvus_bulkinsert/parquet/6.parquet', 'milvus_bulkinsert/parquet/191.parquet', 'milvus_bulkinsert/parquet/151.parquet', 'milvus_bulkinsert/parquet/39.parquet', 'milvus_bulkinsert/parquet/152.parquet', 'milvus_bulkinsert/parquet/176.parquet', 'milvus_bulkinsert/parquet/168.parquet', 'milvus_bulkinsert/parquet/40.parquet', 'milvus_bulkinsert/parquet/236.parquet', 'milvus_bulkinsert/parquet/232.parquet', 'milvus_bulkinsert/parquet/256.parquet', 'milvus_bulkinsert/parquet/49.parquet', 'milvus_bulkinsert/parquet/136.parquet', 'milvus_bulkinsert/parquet/67.parquet', 'milvus_bulkinsert/parquet/290.parquet', 'milvus_bulkinsert/parquet/143.parquet', 'milvus_bulkinsert/parquet/240.parquet', 'milvus_bulkinsert/parquet/278.parquet', 'milvus_bulkinsert/parquet/175.parquet', 'milvus_bulkinsert/parquet/169.parquet', 'milvus_bulkinsert/parquet/56.parquet', 'milvus_bulkinsert/parquet/109.parquet', 'milvus_bulkinsert/parquet/144.parquet', 'milvus_bulkinsert/parquet/251.parquet', 'milvus_bulkinsert/parquet/128.parquet', 'milvus_bulkinsert/parquet/78.parquet', 'milvus_bulkinsert/parquet/2.parquet', 'milvus_bulkinsert/parquet/245.parquet', 'milvus_bulkinsert/parquet/74.parquet', 'milvus_bulkinsert/parquet/237.parquet', 'milvus_bulkinsert/parquet/42.parquet', 'milvus_bulkinsert/parquet/107.parquet', 'milvus_bulkinsert/parquet/53.parquet', 'milvus_bulkinsert/parquet/235.parquet', 'milvus_bulkinsert/parquet/116.parquet', 'milvus_bulkinsert/parquet/37.parquet', 'milvus_bulkinsert/parquet/277.parquet', 'milvus_bulkinsert/parquet/274.parquet', 'milvus_bulkinsert/parquet/44.parquet', 'milvus_bulkinsert/parquet/104.parquet', 'milvus_bulkinsert/parquet/207.parquet', 'milvus_bulkinsert/parquet/127.parquet', 'milvus_bulkinsert/parquet/135.parquet', 'milvus_bulkinsert/parquet/105.parquet', 'milvus_bulkinsert/parquet/130.parquet', 'milvus_bulkinsert/parquet/185.parquet', 'milvus_bulkinsert/parquet/226.parquet', 'milvus_bulkinsert/parquet/260.parquet', 'milvus_bulkinsert/parquet/222.parquet', 'milvus_bulkinsert/parquet/72.parquet', 'milvus_bulkinsert/parquet/270.parquet', 'milvus_bulkinsert/parquet/145.parquet', 'milvus_bulkinsert/parquet/115.parquet', 'milvus_bulkinsert/parquet/199.parquet', 'milvus_bulkinsert/parquet/198.parquet', 'milvus_bulkinsert/parquet/86.parquet', 'milvus_bulkinsert/parquet/33.parquet', 'milvus_bulkinsert/parquet/62.parquet', 'milvus_bulkinsert/parquet/34.parquet', 'milvus_bulkinsert/parquet/264.parquet', 'milvus_bulkinsert/parquet/227.parquet', 'milvus_bulkinsert/parquet/52.parquet']

# files = []
# for path in remote_files:
#     files.append([path])
# retriever.bulk_minio_insert_milvus("vidoseek",files)


#检查导入进度
# jobId="460774211820490079"
# retriever.search_import_progress(jobId)


#检查导入记录有无异常
# resp = retriever.search_import_progress(jobId)
# with open("vidoseek_import_record.json", 'w', encoding='utf-8') as file:
#         json.dump(resp, file, indent=4, ensure_ascii=False)



# 2025/9/14 检查结果无误
with open("/home/gpu/milvus/backend/colpali/ViDoSeek/vidoseek_import_record.json", 'r', encoding='utf-8') as file:
        data = json.load(file)


totalRows = 0
pdfCount = 0        
details = data["data"]["details"]
for item in details:
    if item["progress"] == 100 and item["state"] == "Completed":
        if item["totalRows"] == item["importedRows"]:
            totalRows += item["totalRows"]
            pdfCount += 1
print(totalRows)
print(pdfCount)