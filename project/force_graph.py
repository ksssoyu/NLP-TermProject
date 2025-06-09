#그래프 만들기. 그래프만들 때 distance를 similarity 역수로 설정.
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle

# ---------------------------
# 1. CSV 파일 불러오기
# ---------------------------
similarity_path = 'nlp_papers_similarity.csv'
metadata_path = 'merged_nlp_papers_for_enrichment.csv'

df_sim = pd.read_csv(similarity_path)
df_meta = pd.read_csv(metadata_path).set_index('paperId')  # paper id 기준으로 인덱싱

# ---------------------------
# 2. NetworkX 그래프 생성
# ---------------------------
G = nx.Graph()

threshold = 0.1
filtered_df = df_sim[df_sim['combined_score'] >= threshold]

# ---------------------------
# 3. 노드 및 속성 추가 + 가중치 간선 추가
# ---------------------------
for _, row in filtered_df.iterrows():
    paper_a = row['paper_a']
    paper_b = row['paper_b']
    score = row['combined_score']
    
    # 노드 A가 없으면 추가
    if not G.has_node(paper_a) and paper_a in df_meta.index:
        G.add_node(paper_a,
                   title=df_meta.at[paper_a, 'title'],
                   year=df_meta.at[paper_a, 'year'],
                   authors=df_meta.at[paper_a, 'authors'],
                   concepts=df_meta.at[paper_a, 's2FieldsOfStudy'],
                   citations_count=df_meta.at[paper_a, 'citationCount'])
        
    # 노드 B가 없으면 추가
    if not G.has_node(paper_b) and paper_b in df_meta.index:
        G.add_node(paper_b,
                   title=df_meta.at[paper_b, 'title'],
                   year=df_meta.at[paper_b, 'year'],
                   authors=df_meta.at[paper_b, 'authors'],
                   concepts=df_meta.at[paper_b, 's2FieldsOfStudy'],
                   citations_count=df_meta.at[paper_b, 'citationCount'])

    # 간선 추가
    G.add_edge(paper_a, paper_b, weight=score, distance=1/ score)

# ---------------------------
# 4. Force-Directed Layout
# ---------------------------
pos = nx.spring_layout(G, weight='distance', seed=42)

# ---------------------------
# 5. 저장
# ---------------------------
with open('nlp_similarity_graph.gpickle', 'wb') as f:
    pickle.dump(G, f)

# ---------------------------
# 6. 시각화
# ---------------------------
plt.figure(figsize=(14, 14))
nx.draw_networkx_nodes(G, pos, node_size=20, node_color='skyblue', alpha=0.8)
nx.draw_networkx_edges(G, pos, width=0.5, edge_color='gray', alpha=0.6)
plt.title("Force-Directed Graph of NLP Papers (based on combined_score)", fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.show()
