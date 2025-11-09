import gymnasium as gym
import numpy as np
import time
from gymnasium import spaces
from feature import build_state_for_query
from pathlib import Path

class RAGTopNEnv(gym.Env):

    def __init__(self, train_data, retriever, embeder, milvus_client, scaler, beta=0.1, max_time=1.0):
        super().__init__()
        self.train_data = train_data
        self.retriever = retriever
        self.beta = beta
        self.max_time = max_time  # for time normalization
        self.current_idx = 0
        self.embeder = embeder
        self.milvus_client = milvus_client
        self.scaler = scaler
        self._last_state = None

        # observation: vector length depends on build_state_for_query
        sample_state = build_state_for_query(train_data[0]["query"], embeder, scaler=scaler)[0]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=sample_state.shape, dtype=np.float32)
        self.action_space = spaces.Discrete(15)  # actions 0..14 -> topN = action+1

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.current_idx >= len(self.train_data):
            self.current_idx = 0
        self.query = self.train_data[self.current_idx]["query"]
        self.collection_name = self.train_data[self.current_idx]["collection_name"]
        self.file_name = self.train_data[self.current_idx]["file_name"]
        self.gt = self.train_data[self.current_idx]["evidence_page_nums"]
        self.current_idx += 1
        state, query_np = build_state_for_query(self.query, self.embeder, scaler=self.scaler)  # topN here only to fetch retriever stats
        state = np.asarray(state, dtype=np.float32)
        self.query_np = query_np
        self._last_state = state  # cache for single-step
        return state, {}

    def step(self, action):
        topN = int(action) + 1
        # 1) retrieval with timing
        t0 = time.time()
        # print(f"Retrieving topN={topN} for query: {self.query}, file_name: {self.file_name}, collection: {self.collection_name}")
        pages, scores = self.retriever(self.query_np, self.collection_name, [self.file_name], self.milvus_client, coarse=topN) 
        # print(f"Retrieved pages: {pages[:5]} with scores: {scores[:5]}")
        retrieval_time = time.time() - t0
        # 3) compute recall@5 using ground truth
        recall_at_5 = compute_recall_at_5(self.gt, pages) 
        # 4) normalize retrieval_time
        retrieval_time_norm = min(1.0, retrieval_time / self.max_time)
        # 5) reward
        reward = recall_at_5 - self.beta * retrieval_time_norm
        # clip reward to stabilize training
        reward = float(np.clip(reward, -1.0, 1.0))

        if self._last_state is None:
            next_state = build_state_for_query(self.query, self.embeder, scaler=self.scaler)[0]
            next_state = np.asarray(next_state, dtype=np.float32)
        else:
            next_state = self._last_state

        done = True  # single-step
        info = {"recall@5": recall_at_5, "retrieval_time": retrieval_time}
        return next_state, reward, done, False, info

def compute_recall_at_5(gt, docs):
    if gt is None or len(gt) == 0:
        return 0.0
    topk = docs[:5]
    topk_pages = [int(Path(p).stem) for p in topk]
    hits = sum(1 for d in topk_pages if d in gt)
    return hits / min(len(gt), 5)
