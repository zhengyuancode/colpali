import json
import os
from openai import OpenAI
from tqdm import tqdm
from datasets import load_dataset,load_from_disk
from pathlib import Path


def evaluate(answer_path,datasets_path):
    ds = load_from_disk(datasets_path)
    ds_dict = {item["questionId"]: item for item in ds}
    with open(answer_path, 'r', encoding='utf-8') as file:
        answer = json.load(file)
    # old_scores = []
    # for ans in answer["singleHop"]:
    #     if "eval_score" in ans:
    #         old_scores.append(ans["eval_score"])
    # if len(old_scores) != 280:
    #     print("旧分数统计失败")
    #     return
    sub_dataset={"examples":[]}
    null_count = 0
    for item in tqdm(answer["singleHop"],desc="Rerange datasets"):
        if not item:
            null_count += 1
            continue
        uid = item["uid"]
        if uid in ds_dict and not ("judge" in item):
            sub_dataset["examples"].append(ds_dict[uid])
    
    if(len(sub_dataset["examples"]) != len(answer["singleHop"]) - null_count):
        print("评估数据错误")
        return
    # if(len(sub_dataset["examples"]) != 355):
    #     print("评估数据错误")
    #     return
    judeges = []
    j = 0
    for i in tqdm(range(len(answer["singleHop"])), desc="Evaluating examples"):
        if not answer["singleHop"][i]:
            i += 1
            continue
        if "judge" in answer["singleHop"][i]:
            continue
        if answer["singleHop"][i]["query"] != sub_dataset["examples"][j]["query"]:
            print("评估数据错误")
            return
        if answer["singleHop"][i]["uid"] != sub_dataset["examples"][j]["questionId"]:
            print("评估数据错误")
            return
        sub_data = sub_dataset["examples"][j]
        answer_paths = answer["singleHop"][i]["pages"]
        judege = 0
        for ans in answer_paths:
            answer_list = str(Path(ans).stem).split('_')
            sub_data_list = [sub_data["questionId"],str(sub_data["docId"]),sub_data["image_filename"],sub_data["page"]]
            if answer_list == sub_data_list:
                judege = 1
                break
        judeges.append(judege)
        answer["singleHop"][i]["judge"] = judege
        j += 1
        
    acc = 0
    for j in judeges:
        if j == 0:
            continue
        elif j == 1:
            acc += 1
        else:
            print("judge格式有误")
    
    accuracy = round(acc / len(judeges),3)
    answer["eval_results"]={
        "total": len(judeges),
        "acc": acc,
        "accuracy": accuracy
    }
    with open(answer_path, 'w', encoding='utf-8') as f:
        json.dump(answer, f, ensure_ascii=False, indent=4)
    print(f"评估完成，在{answer_path}查看结果")
    
def main():
    evaluate("Muti_hybrid_search_text_in_img_results.json","./vidore_data/docvqa_test_subsampled")

if __name__ == "__main__":
    main()
