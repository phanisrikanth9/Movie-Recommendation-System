import pandas as pd
import numpy as np
import ast
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer

# Downloading nltk resources
nltk.download('punkt')

# Initializing the stemmer
ps = PorterStemmer()

# Function to extract the first 3 cast members from the cast column
def convert_cast(text):
    cast_list = []
    try:
        for i in ast.literal_eval(text):
            cast_list.append(i['name'])
        return cast_list[:3]
    except:
        return []

# Function to extract genres and keywords from JSON-like string format
def convert(text):
    L = []
    try:
        for i in ast.literal_eval(text):
            L.append(i['name'])
        return L
    except:
        return []

# Function to extract director's name from the crew column
def fetch_director(text):
    director_list = []
    try:
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                director_list.append(i['name'])
        return director_list
    except:
        return []

# Function for stemming
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

# Loading the datasets
movies = pd.read_csv(r'')#copy the path of the tmdb_5000_movies.csv and paste it here
credits = pd.read_csv(r'')#copy the path of the tmdb_5000_credits.csv and paste it here

# Merging the datasets on title
movies = movies.merge(credits, on='title')

# Keeping only relevant columns
movies = movies[['movie_id', 'title', 'genres', 'keywords', 'overview', 'cast', 'crew']]

# Handling missing values
movies.dropna(inplace=True)

# Convert genres, keywords, cast, and crew columns from JSON-like strings to lists
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)

# Filling missing overviews with empty strings
movies['overview'] = movies['overview'].apply(lambda x: x if isinstance(x, str) else "")

# Creating the 'tags' column by concatenating overview, genres, keywords, cast, and crew
movies['tags'] = movies['overview'] + " " + movies['genres'].apply(lambda x: " ".join(x)) + " " + movies['keywords'].apply(lambda x: " ".join(x)) + " " + movies['cast'].apply(lambda x: " ".join(x)) + " " + movies['crew'].apply(lambda x: " ".join(x))

# Applying stemming to the tags column
movies['tags'] = movies['tags'].apply(stem)

# Using CountVectorizer to convert the 'tags' column into numerical features (bag of words)
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# Calculating cosine similarity between the feature vectors
similarity = cosine_similarity(vectors)

# Saving the processed data and similarity matrix using pickle
pickle.dump(movies.to_dict(), open('movies_dict.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("Data preprocessing completed and saved.")
