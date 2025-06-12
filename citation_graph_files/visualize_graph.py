import networkx as nx
import matplotlib.pyplot as plt
import logging

# --- Configuration ---
# The path to the graph file created by the graph builder script.
INPUT_GRAPH_FILE = "citation_graph.graphml"
# The number of top-ranking papers to consider for visualization.
# We will select the #1 paper from this list to be the center of our visualization.
TOP_N = 1
# File to save the visualization to.
OUTPUT_IMAGE_FILE = "citation_ego_network.png"

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Functions ---

def visualize_ego_network(graph_path):
    """
    Loads a citation graph, finds a central node, and visualizes its ego network.
    
    Args:
        graph_path (str): The path to the input .graphml file.
    """
    logging.info(f"Loading graph from {graph_path}...")
    try:
        G = nx.read_graphml(graph_path)
    except FileNotFoundError:
        logging.error(f"File not found: {graph_path}. Please run the graph builder script first.")
        return

    if not G.nodes():
        logging.warning("Graph is empty. Cannot generate visualization.")
        return

    # --- 1. Find a Central Node ---
    logging.info("Calculating PageRank to find a central node for visualization...")
    try:
        pagerank_scores = nx.pagerank(G, weight='weight')
        # Sort by PageRank to find the top node
        center_node_id = sorted(pagerank_scores, key=pagerank_scores.get, reverse=True)[TOP_N - 1]
        center_node_data = G.nodes[center_node_id]
        logging.info(f"Selected '{center_node_data.get('title', 'N/A')}' as the center of the ego network.")
    except Exception as e:
        logging.error(f"Failed to calculate PageRank to select a central node: {e}")
        # As a fallback, just pick the first node if PageRank fails
        center_node_id = list(G.nodes())[0]
        center_node_data = G.nodes[center_node_id]
        logging.warning(f"Using fallback node '{center_node_data.get('title', 'N/A')}' as the center.")


    # --- 2. Create the Ego Subgraph ---
    predecessors = list(G.predecessors(center_node_id))
    successors = list(G.successors(center_node_id))
    
    subgraph_nodes = [center_node_id] + predecessors + successors
    subgraph = G.subgraph(subgraph_nodes)
    
    logging.info(f"Created subgraph with {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges.")

    # --- 3. Prepare for Visualization ---
    fig, ax = plt.subplots(figsize=(20, 20), dpi=100)
    
    pos = nx.spring_layout(subgraph, k=0.8, iterations=50, seed=42)

    # Node sizes
    node_sizes = [G.nodes[n].get('citationCount', 0) for n in subgraph.nodes()]
    min_size, max_size = 30, 3000
    max_node_size_val = max(node_sizes) if node_sizes else 0
    if max_node_size_val > 0:
        scaled_sizes = [min_size + (max_size - min_size) * ((n if n > 0 else 1)**0.5 / max_node_size_val**0.5) for n in node_sizes]
    else:
        scaled_sizes = [min_size] * len(node_sizes)
        
    # Node colors
    node_years = [G.nodes[n].get('year', 2010) for n in subgraph.nodes()]
    cmap = plt.cm.viridis

    # Edge widths
    edge_weights = [d['weight'] for u, v, d in subgraph.edges(data=True)]
    scaled_edge_widths = [w * 0.25 for w in edge_weights]
    
    # --- 4. Draw the Graph ---
    logging.info("Drawing the graph...")
    
    nx.draw_networkx_nodes(subgraph, pos, node_size=scaled_sizes, node_color=node_years, cmap=cmap, alpha=0.8, ax=ax)
    
    nx.draw_networkx_edges(subgraph, pos, width=scaled_edge_widths, edge_color='grey', alpha=0.5, arrows=True, arrowsize=12, ax=ax)
    
    # --- IMPORTANT CHANGE: Only label the central node ---
    labels = {}
    for n in subgraph.nodes():
        if n == center_node_id:
            # Get title and year for the center node
            title = G.nodes[n].get('title', 'N/A')
            year = G.nodes[n].get('year', 'N/A')
            
            # Sanitize the title by replacing backslashes
            sanitized_title = title.replace('\\', '/')
            
            # Create the label string for the center node
            labels[n] = f"{sanitized_title[:30]}...\n({year})"
        else:
            # All other nodes get an empty label
            labels[n] = ''

    nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=8, font_color='black', ax=ax)
    
    # Highlight the center node
    nx.draw_networkx_nodes(subgraph, pos, nodelist=[center_node_id], node_size=scaled_sizes[subgraph_nodes.index(center_node_id)], node_color='red', edgecolors='black', linewidths=2, ax=ax)

    # --- 5. Finalize and Save Plot ---
    ax.set_title(f"Ego Network of '{center_node_data.get('title', 'N/A')}'", fontdict={'fontsize': 20})
    ax.axis('off')
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(node_years), vmax=max(node_years)))
    sm.set_array([]) 
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5)
    cbar.set_label('Publication Year')

    try:
        plt.savefig(OUTPUT_IMAGE_FILE, bbox_inches='tight')
        logging.info(f"Visualization saved to {OUTPUT_IMAGE_FILE}")
    except Exception as e:
        logging.error(f"Failed to save visualization file: {e}")
        
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    visualize_ego_network(INPUT_GRAPH_FILE)
