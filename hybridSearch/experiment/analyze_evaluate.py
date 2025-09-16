import json
import os
from openai import OpenAI
from tqdm import tqdm

DEFAULT_SYSTEM_TEMPLATE = """System:
You are an expert evaluation system for a question answering chatbot.

You are given the following information:
- a user query and reference answer
- a generated answer

You may also be given a reference answer to use for reference in your evaluation.

Your job is to judge the relevance and correctness of the generated answer.
Output a single score that represents a holistic evaluation.
You must return your response in a line with only the score.
Do not return answers in any other format.

Follow these guidelines for scoring:
- Your score has to be between 1 and 5, where 1 is the worst and 5 is the best.
- If the generated answer is not relevant to the user query, you should give a score of 1.
- If the generated answer is relevant but contains mistakes, you should give a score between 2 and 3.
- If the generated answer is relevant and fully correct, you should give a score between 4 and 5.

Example Response:
4

User:
## User Query
{query}

## Reference Answer
{reference_answer}

## Generated Answer
{generated_answer}
"""


client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="sk-f78b07615c8a45128d760579e6d42e1f",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
def llm_generate(query,reference_answer,generated_answer):
    prompt =  DEFAULT_SYSTEM_TEMPLATE.format(
        query=query,
        reference_answer=reference_answer,
        generated_answer=generated_answer
    )
    
    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen3-max-preview",
        messages=[
            {"role": "system", "content": "You can only answer one number"},
            {"role": "user", "content": prompt},
        ],

    )
    return(json.loads(completion.model_dump_json())["choices"][0]["message"]["content"])

def evaluate(dataset_path,answer_path):
    with open(dataset_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
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
    for item in answer["singleHop"]:
        uid = item["uid"]
        for data in dataset["examples"]:
            if (data["uid"] == uid) and not ("eval_score" in item):
                sub_dataset["examples"].append(data)
                break
    
    if(len(sub_dataset["examples"]) != len(answer["singleHop"])):
        print("评估数据错误")
        return
    # if(len(sub_dataset["examples"]) != 355):
    #     print("评估数据错误")
    #     return
    
    scores = []
    for i in tqdm(range(len(answer["singleHop"])), desc="Evaluating examples"):
        if "eval_score" in answer["singleHop"][i]:
            continue
        if answer["singleHop"][i]["query"] != sub_dataset["examples"][i]["query"]:
            print("评估数据错误")
            return
        if answer["singleHop"][i]["uid"] != sub_dataset["examples"][i]["uid"]:
            print("评估数据错误")
            return
        query = answer["singleHop"][i]["query"]
        reference_answer = sub_dataset["examples"][i]["reference_answer"]
        generated_answer = answer["singleHop"][i]["answer"]
        score = llm_generate(query,reference_answer,generated_answer)
        if score == "1" or score == "2" or score == "3" or score == "4" or score == "5":
            scores.append(score)
            answer["singleHop"][i]["eval_score"] = score
        else:
            print("LLM回答格式有误")
            return 
        
    acc = 0
    for s in scores:
        if s == "1" or s == "2" or s == "3":
            continue
        elif s == "4" or s == "5":
            acc += 1
        else:
            print("LLM回答格式有误")
    
    accuracy = round(acc / len(scores),3)
    answer["eval_results"]={
        "total": len(scores),
        "acc": acc,
        "accuracy": accuracy
    }
    with open(answer_path, 'w', encoding='utf-8') as f:
        json.dump(answer, f, ensure_ascii=False, indent=4)
    print(f"评估完成，在{answer_path}查看结果")
    
def main():
    evaluate("vidoseek_singleHop.json","Muti_vector_Img_search_results.json")

if __name__ == "__main__":
    main()
