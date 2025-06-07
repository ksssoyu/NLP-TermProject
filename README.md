Analysis
1. semantic_scholars_sync_fetch : Fetch paper data starting from 13 landmark nlp-related papers, starting from 2010, with minimum citation count set to 0 (time taken too long even with 1)
2. construct_graph : constructs citation graph with weighted edges
3. cluster : leiden community detection and summarize the largest community
4. anaylyze_centrality : computes pagerank and eigenvector centrality on the entire graph
5. visualize_graph : visualizes the largest cluster(attention is the centre)
6. sliding_window_analysis : from 2010~2012 window and onward, summarize and visualize each subgraph
7. combined_analysis : run Leiden and K-means with respect to embedded vector. Each cluster image represents leiden community, and the colors represent vector-based clusters. Colors are same for same k-means cluster accross different leiden clusters.

Uilities
1. process_single_paper : integrates single paper to the graph
2. citation_counter : simply lists up the top n papers by raw citation count
3. fetch_embeddings : fetches embedded vector for each paper
4. fix_missing_embeddings : re-fetches missing embedding values
5. validate_composition : simply checks the percentage to which the dataset is related to field of nlp
