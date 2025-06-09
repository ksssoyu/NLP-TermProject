from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import pandas as pd
import pickle
import numpy as np
import random
from tqdm import tqdm

# 1. 논문 메타데이터 불러오기
df = pd.read_csv("nlp_papers.csv").fillna("")
paper_texts = {row["id"]: row["title"] for _, row in df.iterrows()}

# 2. 그래프 로드
with open("nlp_similarity_graph_with_cluster.gpickle", "rb") as f:
    G = pickle.load(f)

# 3. ✅ all-mpnet-base-v2 모델 로드 및 임베딩 생성
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
nodes = list(G.nodes())
texts = [paper_texts[n] for n in nodes if n in paper_texts]
print(f"🔢 총 임베딩할 논문 수: {len(texts)}")

embeddings = model.encode(texts, batch_size=16, show_progress_bar=True)

# 4. 매핑
id2embedding = {
    n: e for n, e in zip(nodes, embeddings)
}

# 5. 실제 그래프 엣지의 유사도 계산
similarities = []
for a, b in tqdm(G.edges(), desc="📈 그래프 유사도 계산 중"):
    if a in id2embedding and b in id2embedding:
        sim = cosine_similarity([id2embedding[a]], [id2embedding[b]])[0][0]
        similarities.append(sim)

# 6. 랜덤 엣지 유사도 (baseline 비교용)
random_similarities = []
random_edges = random.sample([(a, b) for a in nodes for b in nodes if a != b], len(similarities))
for a, b in tqdm(random_edges, desc="🎲 랜덤 유사도 계산 중"):
    if a in id2embedding and b in id2embedding:
        sim = cosine_similarity([id2embedding[a]], [id2embedding[b]])[0][0]
        random_similarities.append(sim)

# 7. 결과 출력
print(f"📊 그래프 엣지 평균 유사도: {np.mean(similarities):.4f}")
print(f"🎲 랜덤 엣지 평균 유사도: {np.mean(random_similarities):.4f}")
