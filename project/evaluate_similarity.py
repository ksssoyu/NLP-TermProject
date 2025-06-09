import pickle
import networkx as nx
import json
import random
from tqdm import tqdm

# 1. 그래프 로딩
with open("nlp_similarity_graph_with_cluster.gpickle", "rb") as f:
    G = pickle.load(f)

def parse_concepts(concept_str):
    try:
        return set(c.strip().lower() for c in json.loads(concept_str.replace('""', '"')) if c.strip())
    except:
        return set()

# 2. 노드 → concept 매핑 미리 생성
concept_map = {}
for node in G.nodes:
    cstr = G.nodes[node].get("concepts", "")
    concepts = parse_concepts(cstr)
    if concepts:
        concept_map[node] = concepts

# 3. 평균 이웃 개념 유사도 계산
neighbor_similarities = []
for node in tqdm(concept_map):
    my_concepts = concept_map[node]
    for neighbor in G.neighbors(node):
        if neighbor in concept_map:
            neighbor_concepts = concept_map[neighbor]
            intersection = my_concepts & neighbor_concepts
            union = my_concepts | neighbor_concepts
            if union:
                jaccard = len(intersection) / len(union)
                neighbor_similarities.append(jaccard)

# 4. 랜덤 노드 쌍 유사도 계산 (같은 수만큼)
all_nodes = list(concept_map.keys())
random_similarities = []
sample_size = len(neighbor_similarities)

for _ in tqdm(range(sample_size)):
    a, b = random.sample(all_nodes, 2)
    a_concepts = concept_map[a]
    b_concepts = concept_map[b]
    intersection = a_concepts & b_concepts
    union = a_concepts | b_concepts
    if union:
        jaccard = len(intersection) / len(union)
        random_similarities.append(jaccard)

# 5. 결과 출력
mean_neighbor = sum(neighbor_similarities) / len(neighbor_similarities) if neighbor_similarities else 0
mean_random = sum(random_similarities) / len(random_similarities) if random_similarities else 0

print(f"📊 평균 이웃 개념 유사도 (실제 엣지): {mean_neighbor:.4f}")
print(f"🎲 평균 랜덤 노드 쌍 개념 유사도: {mean_random:.4f}")
print(f"🔍 차이: {mean_neighbor - mean_random:.4f}")
