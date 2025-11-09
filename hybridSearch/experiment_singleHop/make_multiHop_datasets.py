import json
from openai import OpenAI
from tqdm import tqdm

QWENAPIKEY="sk-f78b07615c8a45128d760579e6d42e1f"
DMXAPIKEY="sk-gWMA9DJgGb2QzeIa7L7nOvWeXpeESrBAB6SXVflIjnafbonl"

QWENclient = OpenAI(
    api_key=QWENAPIKEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    max_retries=3
)

DMXAPIclient = OpenAI(
    api_key=DMXAPIKEY,
    base_url="https://www.dmxapi.cn/v1",
    max_retries=3
)

prompt = """
You need to determine whether [question] is a multi hop inference problem based on the [question] and [answer] I provided to you. If it is, return "1", otherwise return "0"
Note that multi hop reasoning problems do not equal multi page problems!
- Multi hop reasoning: The premise of this problem is that the model does not know all the pages and needs to repeatedly search for information. The problem can be decomposed/diverged into smaller sub problems, requiring the model to logically connect and reason between multiple document pages, and the answer cannot be directly obtained from a single retrieved page.
- Multi page problem: This type of problem is based on the model being able to see all pages. The problem itself is not complicated, but it requires viewing multiple document pages. It only needs to be integrated or counted (such as "How many tables are there in the entire document?"), it simply extracts information from the document (such as "Which titles would provide insights into the importance of light efficiency and quality metrics in lighting products?"), without the need for logical reasoning steps.
------------------------------------------------------------------
example 1：
[question]:"How many tables are there in this document to represent the income situation in 2018?"
[answer]:"8"

#your output:
0

example 2:
[question]:"Which titles would provide insights into the importance of light efficiency and quality metrics in lighting products?"
[answer]:"["Uiterst hoge lichtefficientie (tot meer dan 100Lm/W)", "dimmable2.200KI 2.600K CRI95", "ambient-dimmable2.000K·2.900KCRI95"]"

#your output:
0
------------------------------------------------------------------
[question]:{question}
[answer]:{answer}
"""

def check_data_llm(data):
    if len(data["evidence_pages"]) <= 1:
        return 0
    user_prompt = prompt.format(
        question = data["question"],
        answer = data["answer"]
    )
    try:
        completion = DMXAPIclient.chat.completions.create(
            model="Qwen3-Next-80B-A3B-Instruct",
            messages=[
                {"role": "system", "content": "You can only answer 0 or 1"},
                {"role": "user", "content": user_prompt},
            ],
        ) 
        return int(json.loads(completion.model_dump_json())["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"llm error{e}")
        return 0
    
def main():
    file_path = "/home/gpu/dzy/M3-CaseRAG/experiment_multiHop/MMLongBench-Doc/data/MMlongDoc.json"
    data_list = []
    multihop_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        data_list = json.load(file)["examples"]
    for data in tqdm(data_list,desc="check data"):
        llm_judge = check_data_llm(data)
        try:
            if llm_judge == 1:
                multihop_data.append(data)
        except Exception as e:
            print(f"LLM format error:{llm_judge}")
    with open("/home/gpu/dzy/M3-CaseRAG/experiment_multiHop/MMLongBench-Doc/data/multihop_MMLongBench-Doc.json", 'w', encoding='utf-8') as f:
        json.dump(multihop_data, f, ensure_ascii=False, indent=4) 

if __name__ == "__main__":
    main()