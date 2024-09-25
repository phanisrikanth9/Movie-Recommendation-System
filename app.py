import streamlit as st 
import pickle
import pandas as pd
import requests

# Function to fetch the movie poster using the TMDB API
def fetch_poster(movie_id):
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=c4c9897af1ecb7aa91be175d81e5694f&language=en-US')
    data = response.json()
    return f"https://image.tmdb.org/t/p/w185/{data['poster_path']}"

# Function to fetch the movie rating using the TMDB API
def fetch_rating(movie_id):
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=c4c9897af1ecb7aa91be175d81e5694f&language=en-US')
    data = response.json()
    return data['vote_average']

# Loading the precomputed movies data and similarity matrix
movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

# Extract unique genres
genres = sorted(list(set([genre for sublist in movies['genres'] for genre in sublist])))

# Recommendation function
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_movies_posters = []
    movie_ratings = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movies_posters.append(fetch_poster(movie_id))
        movie_ratings.append(fetch_rating(movie_id))
    
    return recommended_movies, recommended_movies_posters, movie_ratings

# Streamlit UI
st.set_page_config(page_title="Movie Recommender", page_icon="🎥", layout="wide")
st.title('🎬 Movie Recommender System')
st.markdown('<style>h2{color: #ff6347;} h3{color: #4682b4;} .recommend-button>button{background-color: #87CEEB; color: white;}</style>', unsafe_allow_html=True)
st.markdown('<h2 style="color: #87CEEB;">Find Your Next Favorite Movie</h2>', unsafe_allow_html=True)

# Genre selection dropdown
selected_genre = st.selectbox(
   "Select a genre",
   genres,
   key='genre_selector',
   index=0
)

# Filter movies based on the selected genre
filtered_movies = movies[movies['genres'].apply(lambda x: selected_genre in x)]

# Movie selection dropdown based on the selected genre
selected_movie_name = st.selectbox(
   "Select a movie from the chosen genre",
   filtered_movies['title'].values,
   key='movie_selector'
)

# Recommendation button
if st.button('Recommend'):
    names, posters, ratings = recommend(selected_movie_name)
    st.write('### Movies you may also like:')

    # Displaying recommended movies in columns
    cols = st.columns(5)
    for col, name, poster, rating in zip(cols, names, posters, ratings):
        with col:
            st.image(poster, use_column_width=True)
            st.markdown(f"**{name}**")
            st.markdown(f"Rating: {rating}")

# Footer
# Footer
st.markdown("---")
footer = """
<style>
footer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
 background-color: #000000; /* Black color */
    text-align: center;
    padding: 10px;
}
</style>
<footer>
    Developed by <b>Phani Srikanth Samayam</b> | 
    <a href='https://www.linkedin.com/in/phani-srikanth-samayam-70942926a/' target='_blank'>LinkedIn</a>
</footer>
"""
st.markdown(footer, unsafe_allow_html=True)
