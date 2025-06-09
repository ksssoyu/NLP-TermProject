from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import pandas as pd
import pickle
import numpy as np
import random
import ast
from tqdm import tqdm

# --------------------------
# 1. CSV 불러오기 및 텍스트 구성
# --------------------------
df = pd.read_csv("merged_nlp_papers_for_enrichment.csv").fillna("")

def parse_fields_of_study(fos_str):
    try:
        fos_list = ast.literal_eval(fos_str)
        if isinstance(fos_list, list):
            return ", ".join(fos_list)
        else:
            return ""
    except:
        return ""

paper_texts = {}
for _, row in df.iterrows():
    pid = row["paperId"]
    title = row["title"]
    fos = parse_fields_of_study(row["s2FieldsOfStudy"])
    if pid and title:
        paper_texts[pid] = f"Title: {title} Concepts: {fos}"

# --------------------------
# 2. 그래프 로드
# --------------------------
with open("nlp_similarity_graph_with_cluster.gpickle", "rb") as f:
    G = pickle.load(f)

# --------------------------
# 3. 임베딩 생성 (Sentence-BERT 사용)
# --------------------------
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
nodes = list(G.nodes())

valid_nodes = [n for n in nodes if n in paper_texts]
texts = [paper_texts[n] for n in valid_nodes]

print(f"🔢 총 임베딩할 논문 수: {len(texts)}")

embeddings = model.encode(texts, batch_size=16, show_progress_bar=True)

# --------------------------
# 4. ID → 임베딩 매핑
# --------------------------
id2embedding = {n: e for n, e in zip(valid_nodes, embeddings)}

# --------------------------
# 5. 그래프 엣지 유사도 계산
# --------------------------
similarities = []
for a, b in tqdm(G.edges(), desc="📈 그래프 엣지 유사도 계산 중"):
    if a in id2embedding and b in id2embedding:
        sim = cosine_similarity([id2embedding[a]], [id2embedding[b]])[0][0]
        similarities.append(sim)

# --------------------------
# 6. 랜덤 엣지 유사도 계산
# --------------------------
random_similarities = []
sample_size = len(similarities)
node_list = list(id2embedding.keys())
random_edges = random.sample([(a, b) for a in node_list for b in node_list if a != b], sample_size)

for a, b in tqdm(random_edges, desc="🎲 랜덤 유사도 계산 중"):
    sim = cosine_similarity([id2embedding[a]], [id2embedding[b]])[0][0]
    random_similarities.append(sim)

# --------------------------
# 7. 결과 출력
# --------------------------
print(f"\n📊 그래프 엣지 평균 유사도: {np.mean(similarities):.4f}")
print(f"🎲 랜덤 엣지 평균 유사도: {np.mean(random_similarities):.4f}")
print(f"🔍 차이: {np.mean(similarities) - np.mean(random_similarities):.4f}")
