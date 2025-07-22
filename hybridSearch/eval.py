import json

# def load_jsonl(file_path):
#     data = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             data.append(json.loads(line))
#     return data

def load_jsonl(file_path, max_lines=5):
    """加载JSONL文件的前N行并返回数据列表"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            data.append(json.loads(line))
    return data

# # 加载本地数据
# corpus_local = load_jsonl('./ViDoSeek/corpus.jsonl')
# queries_local = load_jsonl('./ViDoSeek/queries.jsonl')
# qrels_local = load_jsonl('./ViDoSeek/qrels.jsonl')

# # 验证数据完整性
# print(f"文档数量: {len(corpus_local)}")
# print(f"查询数量: {len(queries_local)}")
# print(f"相关性标注: {len(qrels_local)}")

corpus_local = load_jsonl('./ViDoSeek/corpus.jsonl', max_lines=1)
print("="*50)
print(f"corpus.jsonl 前5条文档:{corpus_local}")
print("="*50)