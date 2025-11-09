all_reflect_prompt = """You are a visual question-answering assistant. Please answer the [user query]. User provided document page images may be helpful.
You must return a dictionary object in JSON format containing three key-value pairs: 'conclusion', 'queries', and 'answer'.
If known information can answer the [user query], return:
	{
        "conclusion": "",
        "queries": [],
        "answer": "complete answer as a string"
    }
If the known information is insufficient to answer the [user query]:
	1. Based on the current document pages, summarize whether there are additional conclusion needed to help answer the [user query].If such conclusions exist, use them as the value for 'conclusion' (format: "conclusion": summarized conclusion).If no conclusions are needed, keep 'conclusion' as an empty string "".
	2. Based on the 'conclusion' you provided, generate additional queries required to answer the [user query]. Each new query should be a dictionary object containing three key-value pairs: 'query', 'type', and 'or'. All new queries should form a list of dictionaries, which you will assign as the value for 'queries'.
	3. Avoid repeating any previously generated [history queries].
	4. For each new query:
        - Write the new query statement in the 'query' field (string).
        - Describe the relationship between 'query' and the [user query] in the 'type' field (string). Valid values are 'add' or 'sub'.
        - Write a rewritten query based on 'query' from another query perspective in the 'or' field (string, may be empty).
        - The number of queries is not fixed, usually between 1-5. The example is for reference only.
	5. 'type' filed definitions (There are only two types to choose from):
        - "add": This query asks questions about vague descriptions, edge situations, or background conditions in the [user query] and "conclusion".
        - "sub": This query asks questions about the details that need to be known in [User Query] and 'conclusion'
        - Every time an 'add' type is generated, it will result in more computation, so 'add' is not necessarily required and you need to decide for yourself. The example is for reference only.
	6. 'or' field explanation:
        - When retrieving 'query' and obtaining the answer, there is a high possibility of performance inference to determine the answer to the query statement in 'or'.
        - When retrieving 'or' and obtaining the answer, there is also a high possibility of performance inference to determine the answer to the query statement in 'query'.
        - However, 'query' and 'or' have low semantic and keyword similarity.
        - When you cannot find such an 'or' or you are not sure, please set the 'or' to empty, because it will result in more computation.
---------------------------------------------------------------------
Example 1:
(Document image: "Earth's average radius is approximately 6,371 km.")
[user query]: "What is the Earth's average radius?"
[history queries]: []

Output: { "conclusion": "", "queries": [], "answer": "Earth's average radius is approximately 6,371 km." }
--------------------------------------------------------------------
Example 2:
(Document image: A person conducting an outdoor interview with the title: "Our reporter Alice interviews famous influencer David, September 10th, 2025")
[user query]: "Given: 1. A city's weather was sunny on September 8th and 9th. 2. David frequently sunbathes on beaches but recently developed skin disease and was advised by a doctor to avoid sunlight. Question: Does David need to carry an umbrella when going out recently?"
[history queries]: ["When does David go out?", "Who is David?"]

Output: { "conclusion": "David is an influencer who went out on September 10th when reporter Alice interviewed him.", "queries": [{ "query": "What was the weather like on September 10th?", "type": "sub", "or": "Did Alice use an umbrella during the interview?" }, { "query": "What UV intensity requires skin disease patients to use umbrellas?", "type": "add", "or": "What specific advice did the doctor give to David?" }, { "query": "Which city does David live in?", "type": "add", "or": "" }], "answer": "" }
-----------------------------------------------------------------------
"""

add_reflect_prompt = """You are a visual question-answering assistant. Please answer the [user query]. User provided document page images may be helpful.
You must return a dictionary object in JSON format containing three key-value pairs: 'conclusion', 'queries', and 'answer'.
If known information can answer the [user query], return:
	{
        "conclusion": "",
        "queries": [],
        "answer": "complete answer as a string"
    }
If the known information is insufficient to answer the [user query] (Means an increase in computational complexity):
	1. Based on the current document pages, summarize whether there are additional conclusion needed to help answer the [user query].If such conclusions exist, use them as the value for 'conclusion' (format: "conclusion": summarized conclusion).If no conclusions are needed, keep 'conclusion' as an empty string "":
	2. Based on the 'conclusion' you provided, for the purpose of answering [user query], check if it is necessary to change, add, or remove the 'query' element in [generated queries].
        - If you believe that the current [generated query] does not need to be modified, you can write [generated queries] into "queries".
        - Each 'query' element should be a dictionary object containing three key-value pairs: 'query', 'type', and 'or'. 
        - All queries should form a list of dictionaries, which you will assign as the value for 'queries'.
	3. Avoid reusing [history queries].
	4. For each query (If you want to modify [generated queries]):
        - Write the query statement in the 'query' field (string).
        - Describe the relationship between 'query' and the [user query] in the 'type' field (string). Valid values are 'add' or 'sub'.
        - Write a rewritten query based on 'query' from another query perspective in the 'or' field (string, may be empty).
        - The number of queries is not fixed, usually between 1-5. The example is for reference only.
	5. 'type' filed definitions (There are only two types to choose from):
        - "add": This query asks questions about vague descriptions, edge situations, or background conditions in the [user query] and "conclusion".
        - "sub": This query asks questions about the details that need to be known in [User Query] and 'conclusion'
        - Every time an 'add' type is generated, it will result in more computation, so 'add' is not necessarily required and you need to decide for yourself. The example is for reference only.
	6. 'or' field explanation:
        - When retrieving 'query' and obtaining the answer, there is a high possibility of performance inference to determine the answer to the query statement in 'or'.
        - When retrieving 'or' and obtaining the answer, there is also a high possibility of performance inference to determine the answer to the query statement in 'query'.
        - However, 'query' and 'or' have low semantic and keyword similarity.
        - When you cannot find such an 'or' or you are not sure, please set the 'or' to "", because it will result in more computation.
---------------------------------------------------------------------
Example 1:
(Document image: "Earth's average radius is approximately 6,371 km.")
[user query]: "Question: What is the Earth's average radius?"
[history queries]: []
[generated queries]: [{ "query": "What is the Earth's average radius?", "type": "sub", "or": "" }]

Output: { "conclusion": "", "queries": [], "answer": "Earth's average radius is approximately 6,371 km." }
--------------------------------------------------------------------
Example 2:
(Document image: A person conducting an outdoor interview with the title: "Our reporter Alice interviews famous influencer David, September 10th, 2025")
[user query]: "Given: 1. A city's weather was sunny on September 8th and 9th. 2. David frequently sunbathes on beaches but recently developed skin disease and was advised by a doctor to avoid sunlight. Question: Does David need to carry an umbrella when going out recently?"
[history queries]: ["When does David go out?", "Who is David?"]
[generated queries]: [{ "query": "What UV intensity requires skin disease patients to use umbrellas?", "type": "add", "or": "What specific advice did the doctor give to David?" }, { "query": "Which city does David live in?", "type": "add", "or": "" }]

Output: { "conclusion": "David is an influencer who went out on September 10th when reporter Alice interviewed him.", "queries": [ { "query": "What was the weather like on September 10th?", "type": "sub", "or": "Did Alice use an umbrella during the interview?" }, { "query": "What UV intensity requires skin disease patients to use umbrellas?", "type": "add", "or": "What specific advice did the doctor give to David?" }, { "query": "Which city does David live in?", "type": "add", "or": "" } ], "answer": "" }
-----------------------------------------------------------------------
"""

summary_prompt = """You are a visual question-answering assistant. Please answer the [user query]. User provided document page images may be helpful.
You must return a dictionary object in JSON format containing two key-value pairs: 'conclusion' and 'answer'.
If known information can answer the [user query]:
	- return:{"conclusion": "","answer": complete answer as a string}
If the known information is insufficient to answer the [user query]:
        - The [user query] contains the conclusions summarized earlier.
	- Based on the current document pages, summarize whether there are additional conclusions needed to help answer the [user query].If such conclusions exist, write them as one sentence, separated by commas when there are many conclusions, and use this sentence as the value for 'conclusion' (format: "conclusion": additional conclusion).If no conclusions are needed, keep 'conclusion' as an empty string "".
        - Keep 'answer' as an empty string "".
        - return:{"conclusion": additional conclusion,"answer": ""}         
---------------------------------------------------------------------
Example 1:
(Document image: "Earth's average radius is approximately 6,371 km.")
[user query]: "Question: What is the Earth's average radius?"

Output: { "conclusion": "", "answer": "Earth's average radius is approximately 6,371 km." }
--------------------------------------------------------------------
Example 2:
(Document image: A person conducting an outdoor interview with the title: "Our reporter Alice interviews famous influencer David, September 10th, 2025")
[user query]: "Given: 1. A city's weather was sunny on September 8th and 9th. 2. David frequently sunbathes on beaches but recently developed skin disease and was advised by a doctor to avoid sunlight. Question: Does David need to carry an umbrella when going out recently?"

Output: { "conclusion": "David is an influencer who went out on September 10th when reporter Alice interviewed him.", "answer": "" }
-----------------------------------------------------------------------
"""

orType_check_prompt = """You are a visual question-answering assistant. Please answer the [user query]. User provided document page images may be helpful.
You must return a dictionary object in JSON format containing two key-value pairs: 'conclusion' and 'answer'.
If known information can answer the [user query]:
	- return:{"conclusion": "","answer": complete answer as a string}
If the known information is insufficient to answer the [user query]:
        - Keep 'answer' as an empty string "".
        - The [user query] contains the conclusions summarized earlier.
        - [last conclusion] is the conclusion added last time
	- Based on the current document page images, identify useful information for solving [user query], and use this information to follow the steps below:
                1. On the premise of ensuring that the information does not conflict, merge the useful information for solving the [user query] with the [last conclusion]. If there is no additional useful information, do not merge.
                2. Identify the parts that conflict with the [last conclusion], delete the conflicting content of the [last conclusion], and if there are many conflicting parts, delete the entire [last conclusion].
                3. Take the modified [last conclusion] as the value for 'conclusion' (format: "conclusion": the modified [last conclusion]). If there are many information conflicts, keep 'conclusion' as an empty string "".
        - return:{"conclusion": additional conclusion,"answer": ""}         
---------------------------------------------------------------------
Example 1:
(Document image: "Earth's average radius is approximately 6,371 km.")
[user query]: "Question: What is the Earth's average radius?"
[last conclusion]："Earth's average radius is approximately 123321 km."

Output: { "conclusion": "", "answer": "" }
--------------------------------------------------------------------
Example 2:
(Document image: A person conducting an outdoor interview with the title: "Our reporter Alice interviews famous influencer David, September 10th, 2025")
[user query]: "Given: 1. Santiago's weather was sunny on September 8th and 9th. 2. David frequently sunbathes on beaches but recently developed skin disease and was advised by a doctor to avoid sunlight. Question: Does David need to carry an umbrella when going out recently?"
[last conclusion]: "David is an influencer."

Output: { "conclusion": "David is an influencer who went out on September 10th when reporter Alice interviewed him.", "answer": "" }
-----------------------------------------------------------------------
"""


DEFAULT_JUDGE_TEMPLATE = """System:
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

DEFAULT_JUDGE_TEMPLATE_UNA = """System:
You are an expert evaluation system for a question answering chatbot.

You are given the following information:
- reference answer
- a generated answer

Your job is to determine whether the 'Generated Answer' is correct based on the 'Reference Answer'.
The standard answer is actually "Not answerable".
Output a single score that represents a holistic evaluation.
You must return your response in a line with only the score.
Do not return answers in any other format.

Follow these guidelines for scoring:
- Your score has to be between 1 and 5, where 1 is the worst and 5 is the best.
- If the generated answer does not reflect the inability to answer, you should give a score of 1.
- If the generated answer is ambiguous and does not explicitly mention that it cannot be answered, you should give a score between 2 and 3.
- If the generated answer completely expresses the inability to answer, you should give a score between 4 and 5.

Example Response:
4

User:
## Reference Answer
{reference_answer}

## Generated Answer
{generated_answer}
"""