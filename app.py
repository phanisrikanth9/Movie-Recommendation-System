"""Streamlit app for the movie recommender.

    streamlit run app.py

Posters are optional: put a TMDB API key in .streamlit/secrets.toml
(TMDB_API_KEY = "...") or the TMDB_API_KEY environment variable.
Without a key the app still works and shows text cards.
"""
import os

import requests
import streamlit as st

from recommender import load_artifacts, recommend

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_data
def films():
    return load_artifacts()


def tmdb_key():
    try:
        return st.secrets.get("TMDB_API_KEY") or os.environ.get("TMDB_API_KEY")
    except Exception:  # no secrets.toml
        return os.environ.get("TMDB_API_KEY")


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def poster(movie_id, key):
    try:
        r = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}",
                         params={"api_key": key}, timeout=5)
        path = r.json().get("poster_path")
        return f"https://image.tmdb.org/t/p/w342{path}" if path else None
    except requests.RequestException:
        return None


data = films()
key = tmdb_key()

st.title("Movie Recommender")
st.caption("Content-based recommendations over 4,800 TMDB films: genres, director, cast, "
           "keywords and plot, compared with TF-IDF and cosine similarity.")

genres = sorted({g for f in data for g in f["g"]})
left, right = st.columns([1, 2])
with left:
    genre = st.selectbox("Filter by genre (optional)", ["All genres"] + genres)
pool = [f for f in data if genre == "All genres" or genre in f["g"]]
with right:
    title = st.selectbox("Pick a movie you like", [f"{f['t']} ({f['y']})" for f in pool])
count = st.slider("How many recommendations", 5, 20, 10)

src, recs = recommend(title.rsplit(" (", 1)[0], data, n=count)
st.subheader(f"Because you liked {src['t']}")
st.write(f"{src['y']} · {', '.join(src['g'])} · directed by {src['d'] or 'unknown'} · ★ {src['r']}")

cols = st.columns(5)
for i, (film, score, why) in enumerate(recs):
    with cols[i % 5]:
        url = poster(film["id"], key) if key else None
        if url:
            st.image(url, use_container_width=True)
        st.markdown(f"**{film['t']}** ({film['y']})")
        st.caption(f"★ {film['r']} · match {score:.0%}")
        if why:
            st.caption("Why: " + "; ".join(why))

st.divider()
st.caption("Data: TMDB 5000 Movie Dataset (Kaggle). Built by Phani Srikanth Samayam · "
           "[GitHub](https://github.com/phanisrikanth9/Movie-Recommendation-System)")
