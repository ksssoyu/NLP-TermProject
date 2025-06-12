import networkx as nx
import igraph as ig
import leidenalg as la
import logging
import pandas as pd

# --- Configuration ---
# The path to the graph file created by the graph builder script.
INPUT_GRAPH_FILE = "final_v2_graph.graphml"
# The number of top clusters to display in the summary.
TOP_N_CLUSTERS = 5
# The number of top papers to show from each cluster to identify its theme.
PAPERS_PER_CLUSTER = 20

# --- Leiden Algorithm Configuration ---
# The resolution parameter controls the size of the communities.
# - Higher values lead to more, smaller communities.
# - Lower values lead to fewer, larger communities.
# A good starting point is often 1.0. You can tune this value.
RESOLUTION_PARAMETER = 1.7
# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Functions ---

def analyze_clusters(graph_path):
    """
    Loads a graph, performs community detection using the Leiden algorithm,
    and prints a summary of the largest communities.
    
    Args:
        graph_path (str): The path to the input .graphml file.
    """
    logging.info(f"Loading graph from {graph_path}...")
    try:
        # Load the networkx graph
        G_nx = nx.read_graphml(graph_path)
    except FileNotFoundError:
        logging.error(f"File not found: {graph_path}. Please run the graph builder script first.")
        return

    if not G_nx.nodes():
        logging.warning("Graph is empty. Cannot perform clustering.")
        return

    logging.info(f"Graph loaded with {G_nx.number_of_nodes()} nodes and {G_nx.number_of_edges()} edges.")

    # --- 1. Convert networkx graph to igraph ---
    # The leidenalg library works with igraph, so we need to convert.
    # We create a new igraph instance from the edge list of our networkx graph.
    logging.info("Converting networkx graph to igraph format for Leiden algorithm...")
    # Get nodes in a specific order to map indices back later
    ordered_nodes = list(G_nx.nodes())
    # Create a mapping from node ID (paperId) to integer index
    node_to_idx = {node: i for i, node in enumerate(ordered_nodes)}
    
    # Get edges with weights
    edges_with_weights = [(node_to_idx[u], node_to_idx[v], d.get('weight', 1.0)) for u, v, d in G_nx.edges(data=True)]
    
    # Create the igraph object
    G_ig = ig.Graph.TupleList(edges=[e[:2] for e in edges_with_weights], directed=True)
    G_ig.es['weight'] = [e[2] for e in edges_with_weights]

    # --- 2. Run the Leiden Algorithm ---
    logging.info(f"Running Leiden algorithm with resolution parameter: {RESOLUTION_PARAMETER}...")
    
    # We use RBConfigurationVertexPartition which is suitable for directed graphs and uses the resolution parameter.
    partition = la.find_partition(
        G_ig, 
        la.RBConfigurationVertexPartition,
        weights='weight',
        resolution_parameter=RESOLUTION_PARAMETER,
        seed=42
    )
    
    modularity = G_ig.modularity(partition, weights='weight')
    logging.info("Leiden algorithm finished.")
    
    # --- 3. Process and Analyze Results ---
    num_clusters = len(partition)
    print("\n" + "="*80)
    print("Clustering Analysis Summary")
    print("="*80)
    print(f"Algorithm: Leiden")
    print(f"Resolution Parameter: {RESOLUTION_PARAMETER}")
    print(f"Number of clusters (communities) found: {num_clusters}")
    print(f"Modularity of the partition: {modularity:.4f}")
    print("="*80)

    # --- 4. Display Top Clusters ---
    # Map the cluster results (which use integer indices) back to paper IDs.
    clusters = [[] for _ in range(num_clusters)]
    for i, cluster_id in enumerate(partition.membership):
        clusters[cluster_id].append(ordered_nodes[i])
        
    # Sort clusters by size (number of papers)
    clusters.sort(key=len, reverse=True)
    
    print(f"\nDisplaying Top {min(TOP_N_CLUSTERS, num_clusters)} Largest Clusters:\n")
    
    # Calculate PageRank on the original networkx graph to identify important papers within clusters
    pagerank_scores = nx.pagerank(G_nx, weight='weight')
    
    for i, cluster_nodes in enumerate(clusters[:TOP_N_CLUSTERS]):
        print(f"--- Cluster {i+1}: {len(cluster_nodes)} papers ---")
        
        # Find the most important papers within this cluster based on their PageRank score
        cluster_pageranks = {node: pagerank_scores.get(node, 0) for node in cluster_nodes}
        top_papers_in_cluster = sorted(cluster_pageranks, key=cluster_pageranks.get, reverse=True)
        
        for j, paper_id in enumerate(top_papers_in_cluster[:PAPERS_PER_CLUSTER]):
            node_data = G_nx.nodes[paper_id]
            title = node_data.get('title', 'N/A')
            year = node_data.get('year', 'N/A')
            display_title = (title[:75] + '...') if title and len(title) > 75 else title
            print(f"  {j+1}. ({year}) {display_title}")
        print("") # Add a blank line for readability

# --- Main Execution ---
if __name__ == "__main__":
    analyze_clusters(INPUT_GRAPH_FILE)
