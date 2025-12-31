import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests

st.title("🎬 Movie Recommender System")

# ---------------- API Key ----------------
API_KEY = "da2c3f99f3f7a5642448d0d1add88bde"

# ---------------- Load Data ----------------
movies_dict = pickle.load(open("movies.pkl", "rb"))
movies = pd.DataFrame(movies_dict)


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def get_similarity():
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()
    return cosine_similarity(vectors)

similarity = get_similarity()

# similarity = pickle.load(open("similarity.pkl", "rb"))

# ---------------- Fetch Poster ----------------
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
        data = requests.get(url, timeout=5).json()
        poster_path = data.get("poster_path", None)

        if poster_path:
            full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
        else:
            full_path = "https://via.placeholder.com/500x750?text=No+Image"

        return full_path
    except Exception as e:
        print("Error fetching poster:", e)
        return "https://via.placeholder.com/500x750?text=Error"

# ---------------- Recommend Function ----------------
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]

    distances = similarity[movie_index]

    # ensure distances is 1D numeric
    distances = np.array(distances, dtype=float).flatten()

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_posters

# ---------------- Streamlit UI ----------------
selected_movie_name = st.selectbox(
    "Select a movie",
    movies['title'].values
)

if st.button("Recommend"):
    names, posters = recommend(selected_movie_name)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(names[0])
        st.image(posters[0])
    with col2:
        st.text(names[1])
        st.image(posters[1])
    with col3:
        st.text(names[2])
        st.image(posters[2])
    with col4:
        st.text(names[3])
        st.image(posters[3])
    with col5:
        st.text(names[4])
        st.image(posters[4])