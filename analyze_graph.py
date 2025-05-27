import networkx as nx
import pandas as pd
import pickle
from collections import Counter, defaultdict
import csv
from louvain import LouvainClustering
import random
import os       

class CitationGraphAnalyzer:
    def __init__(self, graph: nx.Graph | str):
        if isinstance(graph, str):
            with open(graph, "rb") as f:
                self.G: nx.DiGraph = pickle.load(f)
        else:
            self.G: nx.DiGraph = graph.copy()

        self.pagerank_scores: dict[str, float] = None
        self.eigenvector_scores: dict[str, float] = None
        self.louvain_partition: dict[str, int] = None

    #---------------louvain--------------------------------------------------
    def run_louvain_clustering(self, save_cluster_sizes=True, filename="cluster_sizes.csv"):
        self.louvain = LouvainClustering(self.G)
        self.louvain_partition = self.louvain.run_louvain()
        nx.set_node_attributes(self.G, self.louvain_partition, "louvain_cluster_id")

        if save_cluster_sizes:
            self.louvain.save_cluster_sizes(filename)

    def analyze_per_cluster(self, min_cluster_size=5, count=100, output_dir="cluster_outputs"):
        if self.louvain_partition is None:
            raise ValueError("Louvain clustering must be run first.")

        os.makedirs(output_dir, exist_ok=True)
        cluster_ids = set(self.louvain_partition.values())
        for cluster_id in cluster_ids:
            cluster_nodes = [n for n, d in self.G.nodes(data=True) if d.get("louvain_cluster_id") == cluster_id]
            if len(cluster_nodes) < min_cluster_size:
                continue
            subgraph = self.G.subgraph(cluster_nodes).copy()

            sub_analyzer = CitationGraphAnalyzer(subgraph)  # for analysis in subgraph scope
            sub_analyzer.run_pagerank()
            sub_analyzer.run_eigenvector_centrality()

            cluster_dir = os.path.join(output_dir, f"cluster_{cluster_id}")
            os.makedirs(cluster_dir, exist_ok=True)

            sub_analyzer.save_pagerank_to_csv(os.path.join(cluster_dir, "pagerank.csv"))
            sub_analyzer.save_eigenvector_to_csv(os.path.join(cluster_dir, "eigenvector.csv"))
    
    #---------------run rankings----------------------------------------------
    def run_pagerank(self, alpha=0.85):
        self.pagerank_scores = nx.pagerank(self.G, alpha=alpha)

    def run_eigenvector_centrality(self):
        G_undirected = self.G.to_undirected()

        # Skip graphs too small to compute centrality
        if G_undirected.number_of_nodes() < 3:
            self.eigenvector_scores = {}
            return

        try:
            largest_cc_nodes = max(nx.connected_components(G_undirected), key=len)
            G_largest = G_undirected.subgraph(largest_cc_nodes).copy()
            self.eigenvector_scores = nx.eigenvector_centrality_numpy(G_largest)
        except Exception as e:
            self.eigenvector_scores = {}

    #---------top10 quick lookup---------
    def get_top_papers_by_pagerank(self, top_n=10):
        if self.pagerank_scores is None:
            raise ValueError("Run PageRank first")

        top_papers = sorted(self.pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return top_papers

    def get_top_papers_by_eigen_centr(self, top_n=10):
        if self.eigenvector_scores is None:
            raise ValueError("Run eigenvector centrality first")

        top_papers = sorted(self.eigenvector_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return top_papers

    #-----------save to csvs-------------
    def save_top_concepts_by_cluster(self, top_n=10):
        if self.louvain_partition is None:
            raise ValueError("Louvain clustering has not been run yet.")

        cluster_concepts = defaultdict(list)

        for _, data in self.G.nodes(data=True):
            cluster_id = data.get("louvain_cluster_id")
            concepts_str = data.get("concepts", "")
            if cluster_id is None or not concepts_str:
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

    def save_cluster_sizes(self, filename="cluster_sizes.csv"):
        if hasattr(self, 'louvain'):
            self.louvain.save_cluster_sizes(filename)
        else:
            raise ValueError("Louvain clustering must be run first.")
        
    def save_pagerank_to_csv(self, filename="pagerank_rankings.csv"):
        if self.pagerank_scores is None:
            raise ValueError("Run PageRank first")

        df = pd.DataFrame([
            {
                "id": node,
                "title": self.G.nodes[node].get("title"),
                "year": self.G.nodes[node].get("year"),
                "pagerank": score,
                "cluster": self.G.nodes[node].get("louvain_cluster_id", None)
            }
            for node, score in sorted(self.pagerank_scores.items(), key=lambda x: x[1], reverse=True)
        ])
        df.to_csv(filename, index=False)

    def save_eigenvector_to_csv(self, filename="eigenvector_rankings.csv"):
        if self.eigenvector_scores is None:
            raise ValueError("Run eigenvector centrality first")

        df = pd.DataFrame([
            {
                "paper_id": node,
                "title": self.G.nodes[node].get("title"),
                "year": self.G.nodes[node].get("year"),
                "eigenvector": score,
                "cluster_id": self.G.nodes[node].get("louvain_cluster_id", None)
            }
            for node, score in sorted(self.eigenvector_scores.items(), key=lambda x: x[1], reverse=True)
        ])
        df.to_csv(filename, index=False)

