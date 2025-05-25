import networkx as nx
import community as community_louvain
import csv
from collections import defaultdict

class LouvainClustering:
    def __init__(self, graph: nx.Graph):
        self.graph = graph.to_undirected()
        self.partition: dict[str, int] = {}

    def run_louvain(self) -> dict[str, int]:
        self.partition = community_louvain.best_partition(self.graph)
        return self.partition

    def save_cluster_sizes(self, filename="cluster_sizes.csv"):
        if not self.partition:
            raise ValueError("Run Louvain clustering first")

        cluster_sizes = defaultdict(int)
        for node, cluster_id in self.partition.items():
            cluster_sizes[cluster_id] += 1

        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["cluster_id", "node_count"])
            for cluster_id, count in sorted(cluster_sizes.items()):
                writer.writerow([cluster_id, count])

 
