# 📚 Analyzing Academic Influence and Emerging Trends via OpenAlex Citation Graphs

> A graph-based exploration of scholarly impact and topic evolution in NLP using OpenAlex citation data.

---

## 🚀 Motivation

Understanding how academic knowledge evolves is crucial in fast-growing fields like Natural Language Processing (NLP). Traditional systems often rely on keyword matching or raw citation counts, overlooking **structural influence** in citation networks.

This project leverages the **OpenAlex** dataset — a comprehensive open-source scholarly graph — to:
- Detect **emerging topics**,
- Identify **influential papers** using centrality metrics,
- Analyze **knowledge clusters** through citation-based community detection.

### 📌 What is a Knowledge Cluster?

A **knowledge cluster** is a group of papers closely linked by citation relationships. These clusters naturally reflect specific research subfields and can be discovered using graph-based community detection algorithms like **Louvain** or **Label Propagation**.

---

## 🎯 Objectives

1. **Emerging Topic Detection**
   - Slice citation graphs by year (2015–2025)
   - Track centrality metrics (e.g., PageRank) over time
   - Identify papers with rapid influence growth

2. **Influential Paper Discovery**
   - Rank papers using PageRank, HITS, Eigenvector centrality
   - Use co-citation analysis to detect under-the-radar impactful works

3. **Knowledge Cluster Analysis**
   - Detect research communities via **Louvain method**
   - Summarize clusters with representative concepts and keywords
   - Track how clusters evolve, merge, or split over time

---

## 📦 Dataset

- **Source**: [OpenAlex.org](https://openalex.org/)
- **Scope**: NLP-related papers retrieved using the keyword `"natural language processing"`
- **Format**: JSON → Pandas DataFrame

### 📑 Key Fields

| Column | Description |
|--------|-------------|
| `id` | Unique OpenAlex paper ID |
| `title` | Paper title |
| `year` | Publication year |
| `cited_by_count` | Raw citation count |
| `doi` | Digital Object Identifier |
| `first_author` | Name of first author |
| `referenced_works` | List of cited OpenAlex IDs |
| `concepts` | High-level OpenAlex concepts |
| `keywords` | Paper keywords |

### 📈 Citation Graph

- **Node**: Paper (with valid title)
- **Edge**: Directed link A → B (A cites B)
- **Graph Type**: `networkx.DiGraph`

---

## 🧪 Methodology

1. **Data Collection**
   - Use `requests` and OpenAlex API to retrieve NLP papers
   - Extract metadata and references

2. **Citation Graph Construction**
   - Build directed graph using `networkx`
   - Nodes = papers, Edges = citations

3. **Centrality Analysis**
   - Compute PageRank, HITS, degree centrality per year
   - Identify influential papers and fast-rising topics

4. **Community Detection**
   - Use **Louvain method** to find clusters
   - Summarize each cluster with top concepts/keywords

5. **Visualization**
   - Plot topic timelines, centrality rankings
   - Graph visualizations with **Pyvis**, **Plotly**, or **Gephi**

---

## 📊 Expected Outcomes

- 🔍 **List of emerging papers** in NLP (2015–2025)
- 🏅 **Ranking of structurally influential papers**
- 🧠 **Topic clusters** and their evolution
- 🌐 **Interactive visualizations** of the citation network
- 🛠 A lightweight **toolkit** for scholarly trend analysis

---

## 🛠 Tools and Libraries

- **API & Data**: OpenAlex REST API, JSON
- **Processing**: `pandas`, `requests`, `tqdm`, `json`
- **Graph Analysis**: `networkx`, `community` (Louvain)
- **Visualization**: `Pyvis`, `Plotly`, `Matplotlib`, `Gephi`

---

## 🗓 Timeline

| Week | Tasks |
|------|-------|
| **Week 1** | Data collection from OpenAlex, cleaning, structuring |
| **Week 2** | Graph construction & centrality computation |
| **Week 3** | Cluster detection & topic evolution analysis |
| **Week 4** | Final report, slides, and visualizations |

---

## 👥 Team Responsibilities

| Member | Role |
|--------|------|
| **Soyu Kim** | Data collection, API integration, DataFrame structuring |
| **Cheoloh Park** | Graph construction, centrality computation |
| **Sanghun Lee** | Community detection, cluster topic analysis, visualizations |
| **Yejin Cho** | Result synthesis, reporting, final presentation |

> ✨ All members contribute across tasks, with specific leadership roles.

---

## 🧠 Contribution

This project contributes a **scalable**, **interpretable**, and **visually rich** approach to academic network analysis. It goes beyond citation counts to reveal **structural influence**, **topic dynamics**, and **emergent knowledge clusters** — empowering researchers with deeper insights into the scholarly ecosystem.

---
