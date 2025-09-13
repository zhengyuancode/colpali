import os
from openai import OpenAI
import json
import re

client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key="sk-f78b07615c8a45128d760579e6d42e1f",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

def parse_coq_string(question,text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    chain = []
    current_entry = None
    final_content = None
    current_query_num = None
    
    print(text)

    for line in lines:
        # 匹配 [Query X]: 内容
        query_match = re.match(r'\[Query (\d+)\]:\s*(.*)$', line, re.IGNORECASE)
        if query_match:
            if current_entry is not None:
                chain.append(current_entry)
            num = int(query_match.group(1))
            content = query_match.group(2)
            current_entry = {
                "query": content,
                "unsolved_query": "",
                "answer": ""
            }
            current_query_num = num
            continue

        # 匹配 [Unsolved Query]: 内容
        unsolved_match = re.match(r'\[Unsolved Query (\d+)\]:\s*(.*)$', line, re.IGNORECASE)
        if unsolved_match and current_entry is not None:
            current_entry["unsolved_query"] = unsolved_match.group(2)
            continue

        # 匹配 [Answer X]: 内容
        answer_match = re.match(r'\[Answer (\d+)\]:\s*(.*)$', line)
        if answer_match and current_entry is not None:
            num = int(answer_match.group(1))
            if num == current_query_num:
                current_entry["answer"] = answer_match.group(2)
            continue

        # 匹配 [Final Content]: 内容
        final_match = re.match(r'\[Final Content\]:\s*(.*)$', line)
        if final_match:
            final_content = final_match.group(1)

    # 添加最后一个条目
    if current_entry is not None:
        chain.append(current_entry)

    return {
        "question": question,
        "chain": chain,
        "final": final_content
    }

def getHistoricalAns(historical_right_CoQ):
    history=""
    if historical_right_CoQ is None or len(historical_right_CoQ) == 0:
        return history
    for i in range(0,len(historical_right_CoQ)):
        history += f"[Query {i+1}]: "+historical_right_CoQ[i]["query"]+"\n"
        history += f"[Answer {i+1}]: "+historical_right_CoQ[i]["answer"]+"\n"
    return history

def getQAcheck(query,answer,base64_images):
    
    checkprompt=f"Here's a question for you:\n {query}\nThis is someone's answer: \n{answer}\nYou need to read the image of the given document page and answer whether the above answer is correct.\nYour answer can only be 'yes'/'no'/'imperfect'/'unknown'\n\nFor example:\n[question]: What is the shape of the Earth?\n[answer]: The Earth is a perfect sphere.\nIf it is found to be correct based on the document page image provided by the user:\nyes\nIf it is found to be incorrect based on the image of the document page provided by the user:\nno\nIf it is found that the answer is not complete enough based on the document page image provided by the user:\nimperfect\nIf the question and answer cannot be verified based on the document page image provided by the user, or if the user does not provide a document page image:\nunknown"
            
    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-vl-max-2025-08-13",
        messages=[
            {"role": "system", "content": [{"type": "text",
                                            "text": "You are an judge who is only able to respond with 'yes', 'no', 'imperfect' or 'unknown'."
                                            }]},
            {
                "role": "user", 
                "content": base64_images + [{"type": "text", "text": checkprompt}]
            },
        ],
    )
    
    return json.loads(completion.model_dump_json())["choices"][0]["message"]["content"]

def getQueries(question,history,Q_count_max,base64_images):
    CoQprompt=f"Construct a global reasoning chain for this complex [Question] : {question} \nYou should generate a query to the search engine based on what you already know at each step of the reasoning chain, starting with [Query]. \nIf you know the answer for [Query], generate it starting with [Answer].\nYou can try to generate the final answer for the [Question] by referring to the [Query]-[Answer] pairs, starting with [Final Content].\nIf you don't know the answer, generate a query to search engine based on what you already know and do not know, starting with [Unsolved Query].\nIf there are historical answers here, you need to continue writing and improve the historical answers based on the reference document page images.\nNow historical answers:\n{history}\nYou need to ensure that the final answer has at least one [Query] - [Answer] or [Query] - [Unsolved Query] pair, and that there are no more than {Q_count_max} [Query] - [Answer] or [Query] - [Unsolved Query] pairs.\n\nFor example:\n[Question]: Where do Greyhound buses that are in the birthplace of Spirit If...'s performer leave from?\n[Query 1]: Who is the performer of Spirit If... ? \nIf you don't know the answer:\n[Unsolved Query 1]: Who is the performer of Spirit If... ?\nIf you know the answer:\n[Answer 1]: The performer of Spirit If… is Kevin Drew.\n[Query 2]: Where was Kevin Drew born?\nIf you don't know the answer:\n[Unsolved Query 2]: Where was Kevin Drew born?\nIf you know the answer:\n[Answer 2]: Toronto.\n[Query 3]: Where do greyhound buses in Toronto leave from?\nIf you don't know the answer:\n[Unsolved Query 3]: Where do greyhound buses in Toronto leave from?\nIf you know the answer:\n[Answer 3]: Toronto Coach Terminal.\n[Final Content]: The performer of Spirit If… is Kevin Drew [1]. Kevin Drew was born in Toronto [2]. Greyhound buses in Toronto leave from Toronto Coach Terminal [3]. So the final answer is Toronto Coach Terminal."

    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-vl-max-2025-08-13",
        messages=[
            {"role": "system", "content": [{"type": "text",
                                            "text": "You are a document specific retriever who excels at identifying whether user questions need to be decomposed. \
                                            You can only decompose the questions that need to be decomposed, and the questions you decompose will be used for document retrieval. \
                                            Therefore, you need to pay attention to whether the decomposed questions can be vectorized and best matched in the vector library"
                                            }]},
            {
                "role": "user", 
                "content": base64_images + [{"type": "text", "text": CoQprompt}]
            },
        ],
    )

    # {
    # 'question':...,
    # 'chain':[
    #     {
    #         'query':...,
    #         'unsolved_query':...,
    #         'answer':...
    #     },
    #     ...
    #     ],
    # 'final':...
    # }
    return parse_coq_string(question,json.loads(completion.model_dump_json())["choices"][0]["message"]["content"])


  
# print(getQueries("What did Bob eat yesterday and where did he eat it?","",5,[]))
# print(getHistoricalAns(
#     [
#         {'query': 'What did Bob eat yesterday?', 'unsolved_query': '', 'answer': 'apple'}
#     ]
# ))
# history = getHistoricalAns(
#     [
#         {'query': 'What did Bob eat yesterday?', 'unsolved_query': '', 'answer': 'apple'}
#     ]
# )
# print(getQueries("What did Bob eat yesterday and where did he eat it?",history,5,[]))

# print(getQAcheck("what is the main content of the passage?","It's about Love.",[]))