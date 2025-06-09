# %%
import pickle
import networkx as nx
import matplotlib.pyplot as plt

# %%
def load_and_visualize_graph(gpickle_path):
    # Load the graph
    with open(gpickle_path, "rb") as f:
        G = pickle.load(f)
    
    # Basic layout (you can try others like shell_layout, circular_layout, etc.)
    pos = nx.spring_layout(G, seed=42)

    # Draw the graph
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_weight="bold")
    plt.title("Graph Visualization from GPickle")
    plt.show()

if __name__ == "__main__":
    # Replace with your actual .gpickle path
    gpickle_path = r"graph_similarity_cluster00.gpickle"
    load_and_visualize_graph(gpickle_path)

# %%
