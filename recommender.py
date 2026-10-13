"""Content-based movie recommender built on the TMDB 5000 dataset.

    python recommender.py build          # data/*.csv.gz -> artifacts/
    python recommender.py "The Dark Knight"

Pipeline
--------
1. Join movies and credits on the movie id (the original version joined on title,
   which mixes up different films that share a title).
2. For every film keep: genres, keywords, top-3 cast, director, overview.
3. Turn each film into one "tag" document. Multi-word names are glued together
   ("Christopher Nolan" -> "christophernolan") so a person is one token, and the
   structured fields are repeated so they weigh more than plot words.
4. Porter-stem the text, vectorise it, and rank films by cosine similarity.
5. Re-rank gently by a vote-weighted rating so obscure, badly rated films don't
   outrank well-known close matches.
6. Keep only each film's 20 nearest neighbours. The original saved the full
   4,806 x 4,806 similarity matrix (185 MB, stored with Git LFS); the neighbour
   list is under 1 MB and gives the same recommendations.
"""
import ast
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
ARTIFACTS = os.path.join(HERE, "artifacts")
TOP_K = 20

# How many times each field is repeated in the tag document (its weight).
FIELD_WEIGHTS = {"genres": 4, "keywords": 1, "cast": 2, "director": 3, "overview": 1}

_stemmer = PorterStemmer()


# ---------------------------------------------------------------- loading
def _names(cell, limit=None, job=None):
    """Parse TMDB's JSON-ish list column into a list of names."""
    try:
        items = ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return []
    if job:
        items = [i for i in items if i.get("job") == job]
    names = [i["name"] for i in items]
    return names[:limit] if limit else names


def load_movies():
    movies = pd.read_csv(os.path.join(DATA, "tmdb_5000_movies.csv.gz"))
    credits = pd.read_csv(os.path.join(DATA, "tmdb_5000_credits.csv.gz"))
    df = movies.merge(credits.drop(columns="title"), left_on="id", right_on="movie_id")
    df = df.dropna(subset=["overview"]).reset_index(drop=True)
    out = pd.DataFrame({
        "movie_id": df["id"].astype(int),
        "title": df["title"],
        "year": pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(0).astype(int),
        "rating": df["vote_average"].astype(float),
        "votes": df["vote_count"].astype(int),
        "popularity": df["popularity"].astype(float),
        "genres": df["genres"].apply(_names),
        "keywords": df["keywords"].apply(_names),
        "cast": df["cast"].apply(lambda c: _names(c, limit=3)),
        "director": df["crew"].apply(lambda c: _names(c, limit=1, job="Director")),
        "overview": df["overview"],
    })
    return out.sort_values("popularity", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- features
def _glue(names):
    # "Joseph Gordon-Levitt" -> "josephgordonlevitt": one token per person or genre
    # (a hyphen would otherwise split the name and double its weight)
    return [re.sub(r"[^0-9a-z]", "", n.lower()) for n in names]


def _stem(text):
    return " ".join(_stemmer.stem(w) for w in text.split())


def tag_document(row, weights=FIELD_WEIGHTS):
    parts = []
    for field in ("genres", "keywords", "cast", "director"):
        parts += _glue(row[field]) * weights[field]
    parts += [row["overview"].lower()] * weights["overview"]
    return _stem(" ".join(parts))


def build_model(movies, weighted=True, tfidf=True, max_features=5000):
    weights = FIELD_WEIGHTS if weighted else dict.fromkeys(FIELD_WEIGHTS, 1)
    docs = movies.apply(lambda r: tag_document(r, weights), axis=1)
    Vec = TfidfVectorizer if tfidf else CountVectorizer
    kwargs = {"sublinear_tf": True} if tfidf else {}
    vectors = Vec(max_features=max_features, stop_words="english", **kwargs).fit_transform(docs)
    return cosine_similarity(vectors)


def quality_prior(movies, m=200):
    """Vote-weighted rating (the IMDb Top 250 formula): a 9.0 from 12 votes is pulled
    towards the average, a 9.0 from 12,000 votes is not. Scaled to 0..1."""
    v, r = movies["votes"].to_numpy(float), movies["rating"].to_numpy(float)
    c = np.average(r, weights=v)
    wr = (v * r + m * c) / (v + m)
    return (wr - wr.min()) / (wr.max() - wr.min())


def top_neighbours(sim, k=TOP_K, prior=None, prior_weight=0.4):
    """Rank by similarity, gently re-weighted by quality so obscure, poorly rated
    films don't outrank well-known close matches."""
    sim = sim.copy()
    np.fill_diagonal(sim, -1)
    rank = sim if prior is None else sim * ((1 - prior_weight) + prior_weight * prior[None, :])
    idx = np.argsort(-rank, axis=1)[:, :k]
    scores = np.take_along_axis(sim, idx, axis=1)
    return idx, scores


# ---------------------------------------------------------------- evaluation
def evaluate(movies, idx, k=10):
    """There are no user ratings here, so quality is measured with two proxies:
    how many of the top-k share at least one genre, and how often a film by the
    same director appears in the top-k when the director made other films."""
    genres = movies["genres"].apply(set).tolist()
    director = movies["director"].apply(lambda d: d[0] if d else None).tolist()
    by_director = movies.groupby(movies["director"].apply(lambda d: d[0] if d else None)).size()
    genre_hits, dir_hits, dir_total = [], 0, 0
    for i, row in enumerate(idx[:, :k]):
        genre_hits.append(np.mean([bool(genres[i] & genres[j]) for j in row]))
        if director[i] and by_director.get(director[i], 0) > 1:
            dir_total += 1
            dir_hits += any(director[j] == director[i] for j in row)
    return {"genre_overlap@10": round(float(np.mean(genre_hits)), 3),
            "same_director_hit@10": round(dir_hits / max(dir_total, 1), 3)}


# ---------------------------------------------------------------- build / query
def build():
    movies = load_movies()
    results = {}
    prior = quality_prior(movies)
    for name, kw, use_prior in [("original (count, unweighted)", dict(weighted=False, tfidf=False), False),
                                ("tf-idf, weighted fields", dict(weighted=True, tfidf=True), False),
                                ("tf-idf, weighted fields + quality re-rank", dict(weighted=True, tfidf=True), True)]:
        idx, scores = top_neighbours(build_model(movies, **kw), prior=prior if use_prior else None)
        results[name] = evaluate(movies, idx)
        results[name]["mean_rating@10"] = round(float(movies["rating"].to_numpy()[idx[:, :10]].mean()), 2)
    print(json.dumps(results, indent=2))

    os.makedirs(ARTIFACTS, exist_ok=True)
    films = []
    for i, r in movies.iterrows():
        films.append({
            "id": int(r.movie_id), "t": r.title, "y": int(r.year), "r": round(r.rating, 1),
            "g": r.genres, "d": r.director[0] if r.director else "", "c": r.cast,
            "k": r.keywords[:6], "o": r.overview if len(r.overview) <= 240 else r.overview[:237].rsplit(" ", 1)[0] + "…",
            "n": [[int(j), round(float(s), 3)] for j, s in zip(idx[i], scores[i])],
        })
    with open(os.path.join(ARTIFACTS, "movies.json"), "w", encoding="utf-8") as f:
        json.dump(films, f, ensure_ascii=False, separators=(",", ":"))
    with open(os.path.join(ARTIFACTS, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"{len(films)} films -> artifacts/movies.json "
          f"({os.path.getsize(os.path.join(ARTIFACTS, 'movies.json')) / 1e6:.1f} MB)")


def load_artifacts():
    with open(os.path.join(ARTIFACTS, "movies.json"), encoding="utf-8") as f:
        return json.load(f)


def reasons(a, b):
    """Plain-language reasons two films were matched."""
    out = []
    if a["d"] and a["d"] == b["d"]:
        out.append(f"same director ({a['d']})")
    cast = [c for c in a["c"] if c in b["c"]]
    if cast:
        out.append("shared cast: " + ", ".join(cast))
    genres = [g for g in a["g"] if g in b["g"]]
    if genres:
        out.append(", ".join(genres))
    kw = [k for k in a["k"] if k in b["k"]]
    if kw:
        out.append("themes: " + ", ".join(kw[:3]))
    return out


def recommend(title, films=None, n=10):
    films = films or load_artifacts()
    matches = [i for i, f in enumerate(films) if f["t"].lower() == title.lower()]
    if not matches:
        raise KeyError(f"'{title}' is not in the dataset")
    src = films[matches[0]]  # films are sorted by popularity, so the best-known one wins
    return src, [(films[j], s, reasons(src, films[j])) for j, s in src["n"][:n]]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "build":
        src, recs = recommend(" ".join(sys.argv[1:]))
        print(f"Because you liked {src['t']} ({src['y']}):")
        for f, s, why in recs:
            print(f"  {s:.2f}  {f['t']} ({f['y']})  - {'; '.join(why)}")
    else:
        build()
