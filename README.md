# 🎞️ CineMind — Hybrid Movie Recommendation System

A hybrid movie recommendation system that combines **Content-Based Filtering** (TF-IDF + Cosine Similarity) with **Collaborative Filtering** (SVD matrix factorization) to deliver personalized movie suggestions. Built with an interactive Arabic/English Streamlit interface.

---

## 📌 How It Works

Most recommender systems use either content-based or collaborative filtering alone — each has blind spots. CineMind combines both:

```
User Query (user_id + movie_title)
         │
         ├──► Content-Based (TF-IDF on genres → Cosine Similarity)
         │         "Find movies with similar genres"
         │
         └──► Collaborative (SVD on ratings matrix)
                   "Find movies liked by similar users"
         │
         ▼
   Hybrid Score = 0.5 × Content Score + 0.5 × Collaborative Score
         │
         ▼
   Top-N Recommendations
```

---

## 🗂️ Dataset

- **Source:** [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- **movies.csv** — 9,742 movies with titles and genres
- **ratings.csv** — 100,836 ratings from real users

---

## 🏗️ System Components

### 1. Content-Based Filtering
```python
# TF-IDF on movie genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
```
Converts genre tags into TF-IDF vectors, then computes cosine similarity between all movie pairs. Movies with overlapping genre profiles score higher.

### 2. Collaborative Filtering (SVD)
```python
# Matrix factorization via Truncated SVD
user_movie_matrix = ratings.pivot_table(
    index='userId', columns='movieId', values='rating'
).fillna(0)

svd = TruncatedSVD(n_components=20, random_state=42)
matrix_reduced = svd.fit_transform(csr_matrix(user_movie_matrix.values))
predicted_ratings = np.dot(matrix_reduced, svd.components_)
```
Decomposes the sparse user-movie ratings matrix into latent factors, then reconstructs predicted ratings for unseen movies.

### 3. Hybrid Scoring
```python
final_score = (0.5 * collaborative_score) + (0.5 * content_score)
```

---

## 🖥️ Streamlit App

Bilingual (Arabic/English) web interface:
- Select a **User ID** and a **movie you liked**
- Get **Top-5 personalized recommendations** with genre tags and match scores

```bash
streamlit run app.py
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/omaralimona05-maker/cinemind-recommender.git
cd cinemind-recommender
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the dataset
Place `movies.csv` and `ratings.csv` in the root directory.  
Download from [MovieLens](https://grouplens.org/datasets/movielens/latest/).

### 4. Run the notebook
```bash
jupyter notebook IP_Final_project.ipynb
```

### 5. Launch the app
```bash
streamlit run app.py
```

---

## 📊 Example Output

```
Input: User ID = 1, Movie = "Toy Story (1995)"

Recommended Movies:
─────────────────────────────────────────────
Antz (1998)                    Score: 0.847
A Bug's Life (1998)            Score: 0.821
Monsters, Inc. (2001)          Score: 0.803
The Lion King (1994)           Score: 0.791
Shrek (2001)                   Score: 0.778
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-yellowgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![Pandas](https://img.shields.io/badge/Pandas-2.x-lightblue)
![NumPy](https://img.shields.io/badge/NumPy-1.x-blue)
![SciPy](https://img.shields.io/badge/SciPy-1.x-purple)

---

## 📁 Project Structure

```
cinemind-recommender/
├── IP_Final_project.ipynb   # Full pipeline notebook
├── app.py                   # Streamlit web app
├── movies.csv               # Movie metadata
├── ratings.csv              # User ratings
├── requirements.txt
└── README.md
```

---

## 💡 Key Concepts Used

| Concept | Application |
|---------|------------|
| TF-IDF | Genre vectorization |
| Cosine Similarity | Content-based matching |
| SVD (Truncated) | Dimensionality reduction on ratings matrix |
| Matrix Factorization | Collaborative filtering |
| Sparse Matrices | Efficient storage of user-movie matrix |
| Hybrid Filtering | Combining both approaches |

---

## 👤 Author

**Ali Yasser** — Intelligent Systems Student, Alexandria National University  
GitHub: [@omaralimona05-maker](https://github.com/omaralimona05-maker)
