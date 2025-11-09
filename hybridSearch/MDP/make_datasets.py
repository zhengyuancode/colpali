import json
import random
def main():
    with open("/home/gpu/dzy/M3-CaseRAG/experiment_singleHop/MMLongBench-Doc/data/MMlongDoc.json", 'r', encoding='utf-8') as file:
        MMlongDoc = json.load(file)["examples"]
    with open("/home/gpu/dzy/M3-CaseRAG/experiment_singleHop/vidore/docvqa/docvqa.json", 'r', encoding='utf-8') as file:
        docvqa = json.load(file)
    with open("/home/gpu/dzy/M3-CaseRAG/experiment_singleHop/vidore/tatdqa/tatdqa.json", 'r', encoding='utf-8') as file:
        tatdqa = json.load(file)
        
    MMlongDoc_chosen = []
    for item in MMlongDoc:
        pages = list(dict.fromkeys(item["evidence_pages"]))
        if len(pages) <=5 and len(pages) >= 1:
            MMlongDoc_chosen.append({
                "query": item["question"],
                "collection_name": "MMLongDoc",
                "file_name": item["doc_id"],
                "evidence_page_nums": pages
            })
            
    docvqa_chosen = []
    for item in docvqa:
        docvqa_chosen.append({
            "query": item["query"],
            "collection_name": "docvqa",
            "file_name": "docvqa",
            "evidence_page_nums": [item["page_num"]]
        })
        
    tatdqa_chosen = []
    for item in tatdqa:
        tatdqa_chosen.append({
            "query": item["query"],
            "collection_name": "tatdqa",
            "file_name": "tatdqa",
            "evidence_page_nums": [item["page_num"]]
        })
    
    total_chosen = MMlongDoc_chosen + docvqa_chosen + tatdqa_chosen
    
    random.shuffle(total_chosen)
    eval_list = total_chosen[-200:]
    test_list = total_chosen[-400:-200]
    train_list = total_chosen[:-400]
    print(f"train size: {len(train_list)}, eval size: {len(eval_list)}, test size: {len(test_list)}")
    
    with open("./datasets/train.json", 'w', encoding='utf-8') as file:
        json.dump(train_list, file, ensure_ascii=False, indent=4)
    with open("./datasets/eval.json", 'w', encoding='utf-8') as file:
        json.dump(eval_list, file, ensure_ascii=False, indent=4)
    with open("./datasets/test.json", 'w', encoding='utf-8') as file:
        json.dump(test_list, file, ensure_ascii=False, indent=4)
        
if __name__ == "__main__":
    main()