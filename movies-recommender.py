# Importing necessary libraries
import numpy as np
import pandas as pd
import ast
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Loading the datasets
movies = pd.read_csv(r'')#copy the path of the tmdb_5000_movies.csv and paste it here
credits = pd.read_csv(r'')#copy the path of the tmdb_5000_credits.csv and paste it here

# Preview the first few rows of the movies dataset
movies.head()

# Sorting movies by popularity
movies = movies.sort_values(by='popularity', ascending=False)
movies.head()

# Preview the first few rows of the credits dataset
credits.head()

# Merging the two datasets on the 'title' column
movies = movies.merge(credits, on='title')
movies.head()

# Sorting merged movies dataset by popularity
movies = movies.sort_values(by='popularity', ascending=False)
movies.head()

# Selecting relevant columns for content-based recommendation
movies = movies[['movie_id', 'title', 'genres', 'keywords', 'overview', 'cast', 'crew']]

# Limiting the dataset to the top 2500 movies
movies = movies.head(2500)
movies.info()

# Preview the first row of the dataset
movies.head(1)

# Checking for null values
movies.isnull().sum()

# Dropping movies with null overviews
movies.dropna(inplace=True)
movies.isnull().sum()

# Checking for duplicate movies
movies.duplicated().sum()

# Preview the 'genres' column format
movies.iloc[0].genres

# Function to convert list of dictionaries to list of genre names
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

# Function to convert cast list to first 3 cast members
def convert_cast(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter < 3:
            L.append(i['name'])
            counter += 1
    return L

# Function to fetch the director from the crew list
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

# Applying conversion functions to 'genres', 'keywords', 'cast', and 'crew' columns
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)
movies.head()

# Converting 'overview' column to list of words
movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies.head()

# Removing spaces in 'genres', 'keywords', 'cast', and 'crew' columns
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
movies.head()

# Creating a new column 'tags' by concatenating 'overview', 'keywords', 'cast', and 'crew' columns
movies['tags'] = movies['overview'] + movies['keywords'] + movies['cast'] + movies['crew']
movies.head()

# Creating a new DataFrame with 'movie_id', 'title', and 'tags' columns
new_df = movies[['movie_id', 'title', 'tags']]
new_df.head()

# Converting 'tags' column from list to string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df.head()

# Converting 'tags' column to lowercase
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
new_df.head()

# Initializing Porter Stemmer
ps = PorterStemmer()

# Function to apply stemming to text
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

# Applying stemming to 'tags' column
new_df['tags'] = new_df['tags'].apply(stem)
new_df.head()

# Initializing CountVectorizer with max features of 2000 and excluding stop words
cv = CountVectorizer(max_features=2000, stop_words='english')

# Transforming 'tags' column into feature vectors
vectors = cv.fit_transform(new_df['tags']).toarray()
vectors

# Checking created features
cv.get_feature_names_out()

# Calculating cosine similarity between vectors
similarity = cosine_similarity(vectors)

# Function to recommend movies based on cosine similarity
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)

# Example recommendation
recommend('Spider-Man')

# Saving the new DataFrame and similarity matrix using pickle
pickle.dump(new_df.to_dict(), open('movies_dict.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
