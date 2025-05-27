# %%
import pickle
import networkx as nx
import matplotlib.pyplot as plt

# %%
def load_and_visualize_graph(gpickle_path):
    # Load the graph
    with open("nlp_citation_graph.gpickle", "rb") as f:
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
    gpickle_path = r"C:\Users\USER\general\3-1 NLP&IR\alexnet_research_paper_analysis\nlp_citation_graph.gpickle"
    load_and_visualize_graph(gpickle_path)

# %%
