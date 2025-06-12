SEMANTIC SCHOLARS CITATION GRAPH ANALYSIS
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-FOR SUBMISSION-
1. semantic_scholars_sync_fetch.py : Fetch paper data starting from 13 landmark nlp-related papers
2. batch_fill_embedding.py : Used to fill up missing embedding values.
3. construct_graph.py : constructs citation graph with weighted edges
4. cluster.py : leiden community detection and summarize the top n clusters
5. cluster_umap_cc.py : HDBSCAN along with UMAP(dimension reduction tool) and show top papers by citation count
6. anaylyze_centrality.py : computes pagerank and eigenvector centrality on the entire graph
7. sliding_window_analysis.py : from 2010~2012 window and onward, summarize and visualize each subgraph
8. combined_analysis.py : run Leiden and K-means with respect to embedded vector. Each cluster image represents leiden community, and the colors represent vector-based clusters. Colors are same for same k-means cluster accross different leiden clusters.
9. adaptive_dyna.py : for each window, HDBSCAN cluster by embedded vectors with parameter tunings via testing multiple sets of parameters while recording the highest scored set of parameters. Shows 3-view ranking for each window, raw citation count, pagerank and HITS score. The keywords for each cluster are also extracted.
10. dyna_analysis_report.py : Enabling manual granular tweaks on the parameter sets obtained from adaptive learning to find best clustering resolution for each window, Summarize each window by showing 3-view rankings and keywords
11. power_law.py : analyzes long-tailed power distribution
12. validate pagerank : compute kendall's tau value to evaluate the full graph
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Analysis
1. semantic_scholars_sync_fetch : Fetch paper data starting from 13 landmark nlp-related papers, starting from 2010, with minimum citation count set to 0 (time taken too long even with 1)
2. batch_fill_embedding.py : Used to fill up missing embedding values
3. construct_graph : constructs citation graph with weighted edges
4. cluster : leiden community detection and summarize the top n clusters
5. anaylyze_centrality : computes pagerank and eigenvector centrality on the entire graph
6. visualize_graph : visualizes the largest cluster
7. sliding_window_analysis : from 2010~2012 window and onward, summarize and visualize each subgraph
8. cluster umap, umap_cc : HDBSCAN along with UMAP(dimension reduction tool)
9. combined_analysis : run Leiden and K-means with respect to embedded vector. Each cluster image represents leiden community, and the colors represent vector-based clusters. Colors are same for same k-means cluster accross different leiden clusters.
10. dynamic & adaptive analysis : for each window, HDBSCAN cluster by embedded vectors with parameter tunings throughout multiple set of testings recording the highest scoreed set of parameters. 3-view ranking for each window, raw citation count, pagerank and HITS score. The keywords for each cluster are also extracted. -> crucial for temporal analysis 
11. power_law : analyzes long-tailed power distribution
12. validate pagerank : compute kendall's tau value to evaluate the full graph

Uilities
1. process_single_paper : integrates single paper to the graph
2. citation_counter : simply lists up the top n papers by raw citation count
3. fetch_embeddings : fetches embedded vector for each paper
4. fix_missing_embeddings : fix up missing embedding values
5. validate_composition : simply checks the percentage to which the dataset is related to field of nlp
6. filters by citation, nlp relevance and venue : filters papers by each criterion
7. show & list papers: used to check the direct results.
