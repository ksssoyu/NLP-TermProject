import networkx as nx
import powerlaw
import matplotlib.pyplot as plt
import logging

# --- Configuration ---
# This should be the path to the graph you constructed previously.
INPUT_GRAPHML_FILE = "final_v2_graph.graphml"
OUTPUT_PLOT_FILE = "citation_degree_distribution.png"

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_citation_power_law(graph_path):
    """
    Loads a citation graph, analyzes its in-degree distribution to check for
    power-law characteristics, and generates a plot.
    """
    logging.info(f"Loading graph from {graph_path}...")
    try:
        G = nx.read_graphml(graph_path)
    except FileNotFoundError:
        logging.error(f"Graph file not found: '{graph_path}'. Please check the path and filename.")
        return

    logging.info(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # --- In-Degree Analysis (Citation Count) ---
    # The in-degree of a node in a citation graph represents how many times it has been cited.
    in_degrees = [d for n, d in G.in_degree()]
    
    # Filter out nodes with zero in-degree as they don't fit the power-law model
    in_degrees = [d for d in in_degrees if d > 0]
    
    if not in_degrees:
        logging.warning("No nodes with citations (in-degree > 0) found. Cannot perform power-law analysis.")
        return

    logging.info("Fitting power-law distribution to in-degree (citation count) data...")

    # Use the powerlaw package to fit the data
    # This automatically finds the best minimum value (xmin) for the power-law fit
    fit = powerlaw.Fit(in_degrees, discrete=True)
    
    # --- Print the Statistical Results ---
    # The alpha value is the exponent of the power law
    alpha = fit.power_law.alpha
    # The KS distance measures the goodness of fit (lower is better)
    ks_distance = fit.power_law.KS()

    print("\n" + "="*50)
    print("  Power-Law Analysis Results (In-Degree/Citations)")
    print("="*50)
    print(f"Fitted Power-Law Exponent (alpha): {alpha:.4f}")
    print(f"Goodness of Fit (Kolmogorov-Smirnov distance): {ks_distance:.4f}")
    print(f"The 'alpha' value should be between 2 and 3 for typical scale-free networks.")
    print(f"A low KS distance (e.g., < 0.05) indicates a very good fit.")
    print("="*50)

    # --- Generate and Save the Visualization ---
    logging.info(f"Generating and saving plot to {OUTPUT_PLOT_FILE}...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the empirical data (Complementary Cumulative Distribution Function)
    fit.plot_ccdf(ax=ax, color='b', linewidth=2, label='Empirical Data')
    
    # Plot the fitted power-law line
    fit.power_law.plot_ccdf(ax=ax, color='r', linestyle='--', label=f'Power-Law Fit (α={alpha:.2f})')

    ax.set_title('In-Degree Distribution of the Citation Network')
    ax.set_xlabel('Degree (Number of Citations)')
    ax.set_ylabel('CCDF (P(X>=k))')
    ax.legend()
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    
    plt.savefig(OUTPUT_PLOT_FILE)
    logging.info("Plot saved successfully.")
    plt.show()


if __name__ == "__main__":
    # You may need to install the 'powerlaw' package first:
    # pip install powerlaw
    analyze_citation_power_law(INPUT_GRAPHML_FILE)
