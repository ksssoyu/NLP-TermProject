import networkx as nx
import igraph as ig
import leidenalg as la
import logging
import pandas as pd
import time
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# The path to the graph file created by the graph builder script.
INPUT_GRAPH_FILE = "final_v2_graph.graphml"
# Directory to save the output visualizations
OUTPUT_DIR = "final_v2_temporal_visualizations"

# --- Sliding Window Configuration ---
START_YEAR = 2010
END_YEAR = 2023 
WINDOW_SIZE = 3 
STEP_SIZE = 1   

# --- Analysis Configuration for each Subgraph ---
TOP_PAGERANK_PAPERS_TO_SHOW = 10
TOP_CLUSTERS_TO_SHOW = 5
PAPERS_PER_CLUSTER_TO_SHOW = 10
LEIDEN_RESOLUTION = 1.0

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Visualization Function ---

def visualize_clustered_subgraph(subgraph, partition, window_str, pagerank_scores):
    """
    Creates and saves a visualization of a clustered subgraph.
    Nodes are colored by cluster membership.
    
    Args:
        subgraph (nx.DiGraph): The networkx subgraph for the time window.
        partition (la.VertexPartition): The Leiden partition result.
        window_str (str): A string representing the time window (e.g., "2010-2012").
        pagerank_scores (dict): Pre-calculated PageRank scores for node sizing.
    """
    if not subgraph.nodes():
        logging.warning(f"Skipping visualization for {window_str} as subgraph is empty.")
        return

    logging.info(f"Generating visualization for {window_str}...")
    
    fig, ax = plt.subplots(figsize=(20, 20), dpi=100)
    
    # Use a spring layout for better node separation
    pos = nx.spring_layout(subgraph, k=0.5, iterations=50, seed=42)
    
    # --- Visual Properties ---
    # Node colors are based on cluster ID from the Leiden partition
    node_colors = partition.membership
    # Use a discrete colormap that handles the number of clusters well
    if max(node_colors, default=-1) >= 0:
        cmap = plt.cm.get_cmap('viridis', max(node_colors) + 1)
    else: # Handle case with no clusters
        cmap = plt.cm.get_cmap('viridis')

    
    # Node sizes are based on PageRank score within the subgraph
    min_size, max_size = 20, 2000
    pr_values = [pagerank_scores.get(n, 0) for n in subgraph.nodes()]
    max_pr = max(pr_values) if pr_values else 0
    if max_pr > 0:
        scaled_sizes = [min_size + (max_size - min_size) * (v / max_pr) for v in pr_values]
    else:
        scaled_sizes = [min_size] * len(pr_values)
        
    # --- Draw the Graph ---
    nx.draw_networkx_nodes(subgraph, pos, node_size=scaled_sizes, node_color=node_colors, cmap=cmap, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(subgraph, pos, width=0.5, edge_color='grey', alpha=0.5, arrows=False, ax=ax)
    
    # --- Finalize and Save Plot ---
    ax.set_title(f"Research Clusters for {window_str}", fontdict={'fontsize': 24})
    ax.axis('off')
    
    try:
        # Ensure output directory exists
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        output_path = os.path.join(OUTPUT_DIR, f"cluster_graph_{window_str}.png")
        plt.savefig(output_path, bbox_inches='tight')
        logging.info(f"Visualization saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save visualization file for {window_str}: {e}")
        
    plt.close(fig) # Close the figure to free up memory

# --- Main Analysis Function ---

def sliding_window_analysis_with_viz(graph_path):
    """
    Loads a graph and performs a sliding window analysis, including visualization.
    """
    logging.info(f"Loading full graph from {graph_path}...")
    try:
        G = nx.read_graphml(graph_path)
    except FileNotFoundError:
        logging.error(f"File not found: {graph_path}. Please run the graph builder script first.")
        return

    if not G.nodes():
        logging.warning("Graph is empty. Cannot perform analysis.")
        return
        
    logging.info(f"Full graph loaded with {G.number_of_nodes()} nodes.")
    
    for node, data in G.nodes(data=True):
        try:
            G.nodes[node]['year'] = pd.to_numeric(data.get('year'))
        except (ValueError, TypeError):
            G.nodes[node]['year'] = 0

    for start in range(START_YEAR, END_YEAR + 1, STEP_SIZE):
        end = start + WINDOW_SIZE - 1
        window_label = f"{start}-{end}"
        
        print("\n" + "="*80)
        print(f"ANALYZING TIME WINDOW: {window_label}")
        print("="*80)
        
        nodes_in_window = [n for n, d in G.nodes(data=True) if d.get('year') and start <= d['year'] <= end]
        if not nodes_in_window:
            logging.warning(f"No papers found in the {window_label} time window. Skipping.")
            continue
            
        subgraph = G.subgraph(nodes_in_window)
        logging.info(f"Created subgraph for {window_label} with {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges.")
        
        if subgraph.number_of_edges() == 0:
            logging.warning(f"Subgraph for {window_label} has no edges. Skipping analysis for this window.")
            # Still visualize the (isolated) nodes if you want
            # To do this, we create a dummy partition where each node is its own cluster
            dummy_partition = ig.VertexClustering(None, [i for i, _ in enumerate(subgraph.nodes())])
            dummy_pagerank = {n: 0 for n in subgraph.nodes()}
            visualize_clustered_subgraph(subgraph, dummy_partition, window_label, dummy_pagerank)
            continue

        pagerank_scores = nx.pagerank(subgraph, weight='weight')
        top_papers = sorted(pagerank_scores, key=pagerank_scores.get, reverse=True)
            
        print(f"\n--- Top {TOP_PAGERANK_PAPERS_TO_SHOW} Most Influential Papers ({window_label}) ---")
        for i, paper_id in enumerate(top_papers[:TOP_PAGERANK_PAPERS_TO_SHOW]):
            node_data = subgraph.nodes[paper_id]
            title = node_data.get('title', 'N/A')
            display_title = (title[:75] + '...') if title and len(title) > 75 else title
            print(f"  {i+1}. ({int(node_data.get('year', 0))}) {display_title}")

        try:
            ordered_subgraph_nodes = list(subgraph.nodes())
            node_to_idx = {node: i for i, node in enumerate(ordered_subgraph_nodes)}
            edges_with_weights = [(node_to_idx[u], node_to_idx[v], d.get('weight', 1.0)) for u, v, d in subgraph.edges(data=True)]
            
            # --- IMPORTANT FIX: Create igraph object that includes isolated nodes ---
            num_vertices = len(ordered_subgraph_nodes)
            G_ig = ig.Graph(n=num_vertices, directed=True)
            # Now add the edges
            if edges_with_weights:
                G_ig.add_edges([e[:2] for e in edges_with_weights])
                G_ig.es['weight'] = [e[2] for e in edges_with_weights]
            
            partition = la.find_partition(G_ig, la.RBConfigurationVertexPartition, weights='weight', resolution_parameter=LEIDEN_RESOLUTION, seed=42)
            num_clusters = len(partition)
            print(f"\n--- Top {min(TOP_CLUSTERS_TO_SHOW, num_clusters)} Research Clusters ({window_label}) ---")
            
            clusters = [[] for _ in range(num_clusters)]
            for i, cluster_id in enumerate(partition.membership):
                clusters[cluster_id].append(ordered_subgraph_nodes[i])
            clusters.sort(key=len, reverse=True)
            
            for i, cluster_nodes in enumerate(clusters[:TOP_CLUSTERS_TO_SHOW]):
                print(f"\n  Cluster {i+1} ({len(cluster_nodes)} papers):")
                cluster_pageranks = {node: pagerank_scores.get(node, 0) for node in cluster_nodes}
                top_in_cluster = sorted(cluster_pageranks, key=cluster_pageranks.get, reverse=True)
                for j, paper_id in enumerate(top_in_cluster[:PAPERS_PER_CLUSTER_TO_SHOW]):
                    node_data = subgraph.nodes[paper_id]
                    title = node_data.get('title', 'N/A')
                    display_title = (title[:70] + '...') if title and len(title) > 70 else title
                    print(f"    - ({int(node_data.get('year', 0))}) {display_title}")

            # --- Call the visualization function ---
            visualize_clustered_subgraph(subgraph, partition, window_label, pagerank_scores)

        except Exception as e:
            logging.error(f"Could not perform clustering or visualization for {window_label} subgraph: {e}", exc_info=True)

# --- Main Execution ---
if __name__ == "__main__":
    start_time = time.time()
    sliding_window_analysis_with_viz(INPUT_GRAPH_FILE)
    end_time = time.time()
    logging.info(f"Sliding window analysis and visualization completed in {end_time - start_time:.2f} seconds.")
