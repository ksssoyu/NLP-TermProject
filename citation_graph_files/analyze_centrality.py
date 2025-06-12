import networkx as nx
import logging

# --- Configuration ---
# The path to the graph file created by the graph builder script.
INPUT_GRAPH_FILE = "final_v2_graph.graphml"
# The number of top-ranking papers to display for each metric.
TOP_N = 50

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Functions ---

def analyze_graph_centrality(graph_path):
    """
    Loads a citation graph and calculates PageRank and Eigenvector Centrality.
    
    Args:
        graph_path (str): The path to the input .graphml file.

    Returns:
        tuple: A tuple containing the graph, PageRank scores, and Eigenvector Centrality scores.
               Returns (None, None, None) if the graph cannot be loaded.
    """
    logging.info(f"Loading graph from {graph_path}...")
    try:
        # Load the graph, preserving node and edge attributes
        G = nx.read_graphml(graph_path)
    except FileNotFoundError:
        logging.error(f"File not found: {graph_path}. Please run the graph builder script first.")
        return None, None, None
    except Exception as e:
        logging.error(f"An error occurred while loading the graph: {e}")
        return None, None, None
    
    if not G.nodes():
        logging.warning("Graph is empty. Cannot perform analysis.")
        return G, None, None
        
    logging.info("Graph loaded successfully. Calculating centrality measures...")

    # 1. Calculate PageRank
    # PageRank measures the importance of a node based on the number and quality
    # of links pointing to it. We use the 'weight' attribute of our edges.
    logging.info("Calculating PageRank (weighted)...")
    try:
        pagerank_scores = nx.pagerank(G, weight='weight')
        logging.info("PageRank calculation complete.")
    except Exception as e:
        logging.error(f"Failed to calculate PageRank: {e}")
        pagerank_scores = None

    # 2. Calculate Eigenvector Centrality
    # Eigenvector centrality measures the influence of a node in a network.
    # It assigns scores based on the principle that connections to high-scoring nodes
    # contribute more to the score of the node in question.
    logging.info("Calculating Eigenvector Centrality (weighted)...")
    try:
        # Increase max_iter for larger or more complex graphs to ensure convergence.
        # This might fail on graphs that are not strongly connected.
        eigenvector_scores = nx.eigenvector_centrality(G, weight='weight', max_iter=1000, tol=1.0e-8)
        logging.info("Eigenvector Centrality calculation complete.")
    except nx.PowerIterationFailedConvergence as e:
        logging.error(f"Eigenvector Centrality failed to converge: {e}. "
                      "This can happen with disconnected graphs. Try analyzing the largest connected component.")
        eigenvector_scores = None
    except Exception as e:
        logging.error(f"Failed to calculate Eigenvector Centrality: {e}")
        eigenvector_scores = None

    return G, pagerank_scores, eigenvector_scores

def print_top_papers(graph, scores, metric_name, top_n):
    """
    Prints a formatted table of the top N papers based on a given score.
    """
    if not scores:
        print(f"Could not display top papers for {metric_name} due to calculation error.")
        return
        
    # Sort the papers by score in descending order
    sorted_papers = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    
    print("\n" + "="*80)
    print(f"Top {top_n} Papers by {metric_name}")
    print("="*80)
    print(f"{'Rank':<5} | {'Score':<18} | {'Year':<6} | {'Citations':<10} | {'Title'}")
    print("-"*80)

    for i, (node_id, score) in enumerate(sorted_papers[:top_n]):
        node_data = graph.nodes[node_id]
        title = node_data.get('title', 'N/A')
        year = node_data.get('year', 'N/A')
        # Ensure year is displayed as an integer if it's a float (e.g., 2010.0)
        if isinstance(year, float) and year.is_integer():
            year = int(year)
        
        citations = node_data.get('citationCount', 0)
        if isinstance(citations, float) and citations.is_integer():
            citations = int(citations)

        # Truncate long titles for display
        display_title = (title[:70] + '...') if title and len(title) > 70 else title
        
        print(f"{i+1:<5} | {score:<18.6f} | {str(year):<6} | {str(citations):<10} | {display_title}")
    print("="*80)

# --- Main Execution ---
if __name__ == "__main__":
    # Analyze the graph to get centrality scores
    G, pr_scores, eig_scores = analyze_graph_centrality(INPUT_GRAPH_FILE)

    if G:
        # Print the results
        print_top_papers(G, pr_scores, "PageRank", TOP_N)
        print_top_papers(G, eig_scores, "Eigenvector Centrality", TOP_N)
