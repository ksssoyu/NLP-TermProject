import networkx as nx
import pandas as pd
import pickle
from collections import Counter, defaultdict
import community as community_louvain 
import csv  

class CitationGraphAnalyzer:
    def __init__(self, graph_path):
        with open(graph_path, "rb") as f:
            self.G = pickle.load(f)

        self.pagerank_scores = None
        self.eigenvector_scores = None
        self.louvain_partition = None

    def run_pagerank(self, alpha=0.85):
        self.pagerank_scores : dict[str, float] = nx.pagerank(self.G, alpha=alpha)

    def run_eigenvector_centrality(self):
        G_undirected = self.G.to_undirected()
        self.eigenvector_scores : dict[str, float] = nx.eigenvector_centrality_numpy(G_undirected)

    def run_louvain_clustering(self):
        G_undirected = self.G.to_undirected()
        self.louvain_partition : dict[str, int] = community_louvain.best_partition(G_undirected)
        nx.set_node_attributes(self.G, self.louvain_partition, "louvain_cluster")

    def get_top_papers_by_pagerank(self, top_n=10):
        if self.pagerank_scores is None:
            raise ValueError("Run PageRank first using run_pagerank()")

        top_papers = sorted(self.pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        for i, (node, score) in enumerate(top_papers, 1):
            title = self.G.nodes[node].get("title", "N/A")
            year = self.G.nodes[node].get("year", "N/A")

    def get_top_concepts_by_cluster(self, top_n=10):
        if self.louvain_partition is None:
            raise ValueError("Louvain clustering has not been run yet.")

        cluster_concepts = defaultdict(list)

        for node, data in self.G.nodes(data=True):
            cluster_id = data.get("louvain_cluster")
            concepts_str = data.get("concepts", "")
            if not cluster_id or not concepts_str:
                continue
            concepts = [c.strip().lower() for c in concepts_str.split(";") if c.strip()]
            cluster_concepts[cluster_id].extend(concepts)

            
        with open("cluster_concepts.csv", "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["cluster_id", "concept", "count"])
            for cluster_id, concept_list in cluster_concepts.items():
                counter = Counter(concept_list)
                for concept, count in counter.most_common(top_n):
                    writer.writerow([cluster_id, concept, count])


    def save_pagerank_to_csv(self, filename="pagerank_rankings.csv"):
        if self.pagerank_scores is None:
            raise ValueError("Run PageRank first")

        df = pd.DataFrame([
            {
                "id": node,
                "title": self.G.nodes[node].get("title"),
                "year": self.G.nodes[node].get("year"),
                "pagerank": score,
                "cluster": self.G.nodes[node].get("louvain_cluster", None)
            }
            for node, score in sorted(self.pagerank_scores.items(), key=lambda x: x[1], reverse=True)
        ])
        df.to_csv(filename, index=False)
