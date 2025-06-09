import pandas as pd

# CSV 파일 불러오기
df = pd.read_csv("paper_clusters.csv")

# 클러스터별 노드 수 세기
cluster_sizes = df["community_id"].value_counts().sort_values(ascending=False)



# 클러스터 ID와 크기 열로 변환
df_cluster_sizes = cluster_sizes.reset_index()
df_cluster_sizes.columns = ["cluster_id", "num_nodes"]

# 저장
df_cluster_sizes.to_csv("cluster_sizes_from_csv.csv", index=False)
print("✅ cluster_sizes_from_csv.csv 파일로 저장 완료!")
