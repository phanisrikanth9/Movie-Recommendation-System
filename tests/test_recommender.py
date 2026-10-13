"""Run:  pytest -q"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from recommender import _glue, load_artifacts, reasons, recommend  # noqa: E402

FILMS = load_artifacts()


def test_names_become_single_tokens():
    assert _glue(["Joseph Gordon-Levitt", "Science Fiction"]) == ["josephgordonlevitt", "sciencefiction"]


def test_every_film_has_neighbours_and_never_itself():
    for i, f in enumerate(FILMS):
        assert len(f["n"]) == 20
        assert all(j != i for j, _ in f["n"])


def test_sequels_rank_first():
    _, recs = recommend("The Dark Knight", FILMS, n=3)
    titles = [f["t"] for f, _, _ in recs]
    assert "The Dark Knight Rises" in titles and "Batman Begins" in titles


def test_title_lookup_is_case_insensitive():
    src, _ = recommend("toy story", FILMS, n=1)
    assert src["t"] == "Toy Story"


def test_unknown_title_raises():
    with pytest.raises(KeyError):
        recommend("Not A Real Movie 12345", FILMS)


def test_reasons_mention_director():
    a = {"d": "Christopher Nolan", "c": [], "g": ["Drama"], "k": []}
    b = {"d": "Christopher Nolan", "c": [], "g": ["Drama"], "k": []}
    assert reasons(a, b)[0] == "same director (Christopher Nolan)"
