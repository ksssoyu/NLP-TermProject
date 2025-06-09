import pandas as pd
import numpy as np
import pickle
import networkx as nx
import random
import ast
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# --------------------------
# 1. CSV 로드 및 임베딩 파싱
# --------------------------
df = pd.read_csv("merged_nlp_papers_for_enrichment.csv")

# 임베딩 문자열을 안전하게 리스트로 파싱
def parse_embedding(e_str):
    try:
        vec = ast.literal_eval(e_str)
        return np.array(vec) if isinstance(vec, list) else None
    except:
        return None

# 임베딩이 존재하는 논문만 필터링
df = df[df["embedding"].notna()].copy()
df["parsed_embedding"] = df["embedding"].apply(parse_embedding)
df = df[df["parsed_embedding"].notnull()]

print(f"✅ 유효한 임베딩 수: {len(df)}")

# --------------------------
# 2. 그래프 로드
# --------------------------
with open("nlp_similarity_graph_with_cluster.gpickle", "rb") as f:
    G = pickle.load(f)
graph_node_set = set(G.nodes())
embedding_node_set = set(df["paperId"])
print("📌 샘플 그래프 노드 10개:")
print(list(graph_node_set)[:10])
print(type(next(iter(graph_node_set))))
print(type(next(iter(embedding_node_set))))
print(f"🧩 그래프 노드 수: {len(graph_node_set)}")
print(f"🧠 임베딩 보유 논문 수: {len(embedding_node_set)}")
print(f"🔗 교집합 수: {len(graph_node_set & embedding_node_set)}")
# --------------------------
# 3. ID → 임베딩 매핑
# --------------------------
id2embedding = dict(zip(df["paperId"], df["parsed_embedding"]))

# 그래프에 존재하면서 임베딩도 있는 노드만 대상으로
valid_node_set = set(G.nodes()).intersection(id2embedding.keys())
valid_nodes = list(valid_node_set)

# --------------------------
# 4. 실제 그래프 엣지 유사도 계산
# --------------------------
similarities = []

for a, b in tqdm(G.edges(), desc="📈 그래프 유사도 계산 중"):
    if a in id2embedding and b in id2embedding:
        vec_a = id2embedding[a].reshape(1, -1)
        vec_b = id2embedding[b].reshape(1, -1)
        sim = cosine_similarity(vec_a, vec_b)[0][0]
        similarities.append(sim)

print(f"✅ 계산된 그래프 엣지 수: {len(similarities)}")

# --------------------------
# 5. 랜덤 엣지 유사도 계산
# --------------------------
random_similarities = []
sample_size = len(similarities)

random_edges = random.sample([(a, b) for a in valid_nodes for b in valid_nodes if a != b], sample_size)

for a, b in tqdm(random_edges, desc="🎲 랜덤 유사도 계산 중"):
    vec_a = id2embedding[a].reshape(1, -1)
    vec_b = id2embedding[b].reshape(1, -1)
    sim = cosine_similarity(vec_a, vec_b)[0][0]
    random_similarities.append(sim)

# --------------------------
# 6. 결과 출력
# --------------------------
if similarities and random_similarities:
    mean_graph = np.mean(similarities)
    mean_random = np.mean(random_similarities)
    print(f"\n📊 그래프 엣지 평균 유사도: {mean_graph:.4f}")
    print(f"🎲 랜덤 엣지 평균 유사도: {mean_random:.4f}")
    print(f"🔍 차이: {mean_graph - mean_random:.4f}")
else:
    print("⚠️ 유사도 리스트가 비어 있어 결과를 출력할 수 없습니다.")
