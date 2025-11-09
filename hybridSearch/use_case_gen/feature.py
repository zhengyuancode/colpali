# feature.py
import numpy as np


def get_query_embedding(embeder, query):
    # 默认是128维多向量
    query_embeddings = embeder.encode_text(
                        texts=[query],
                        task="retrieval",
                        prompt_name="query",
                        return_multivector=True,
                    )
    query_np = query_embeddings[0].float().cpu().numpy()
    return query_np, len(query_np)

def aggregate_query_embedding(query_np):
    # 如果是 torch tensor，先转 numpy
    if "torch" in str(type(query_np)):
        query_np = query_np.detach().cpu().numpy()

    # 检查维度是否正确
    if query_np.ndim != 2:
        raise ValueError(f"Expected query_np with 2 dims, got {query_np.ndim}")

    # 聚合方式：平均池化 (mean pooling)
    return np.mean(query_np, axis=0)

def build_state_for_query(query, embeder, scaler=None):
    # 1) embedding & projection
    query_np, token_len = get_query_embedding(embeder, query)
    query_emb = aggregate_query_embedding(query_np)
    query_norm = np.linalg.norm(query_emb)
    # 2) token length
    token_len_feat = np.log1p(token_len)
    # concat
    # state = np.concatenate([query_emb.astype(np.float32), token_len_feat], axis=0)
    state = np.array([token_len_feat, query_norm], dtype=np.float32)
    if scaler is not None:
        state = scaler.transform(state.reshape(1, -1))[0]
    return state, query_np
