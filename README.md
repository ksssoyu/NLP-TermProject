# 📘 NLP Influence Dashboard

A visualization system for exploring how ideas evolve across time in the field of NLP through citation and co-authorship networks.



## ✅ How to Run the Project

### 1️⃣ Backend (FastAPI + Specter Embedding)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```
⚠️ **Note**: Wait until you see Application startup complete. before starting the frontend.
Starting the frontend too early may cause fetch errors.

### 2️⃣ Frontend (React)
```bash
cd frontend
npm install
npm start
```
## 📂 Page Overview
### 🔍 SearchPage.jsx
- Search for papers using a keyword (e.g., “GPT”) using Keyword + PageRank sorting
- Returns top 10 relevant papers with metadata

### 🌟 InfluencePage.jsx

- Displays top influential papers (citation network)
- Displays top influential authors (co-authorship network)
- Uses PageRank for ranking

### 👤 AuthorPage.jsx

- Visualizes a selected author's collaboration graph
- Shows co-authored papers and partnership strength
- Force-directed graph interaction

### 📈 TrajectoryPage.jsx

- Line chart of an author's yearly collaboration activity
- Helps track co-authorship trends over time

### 🧑‍🤝‍🧑 AuthorYearPage.jsx

- Clustered view of authors in a given 3-year window
- Top authors per cluster ranked by co-authorship weight
- Lists papers and collaborators per author

### 🔗 CitationPage.jsx

- Interactive view of citation relationships
- Clickable nodes and edges to trace citation paths

### 🧠 TopicPage.jsx

- Timeline of topic clusters over 3-year sliding windows
- Shows representative keywords (TF-IDF weighted)
- Tracks topic evolution and emerging trends

### 🗂️ CitationYearPage.jsx

- Displays top 10 papers per cluster in each time window
- Uses PageRank to identify influential papers
- Reveals key papers and topic shifts across time

### 🛠️ Tech Stack

- Frontend: React, D3.js
- Backend: FastAPI
- Clustering: HDBSCAN + UMAP (unsupervised, density-based)
- Ranking: PageRank, HITS, Citation Count
