from flask import Flask, jsonify
from flask_cors import CORS
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)
movies = pd.read_csv('cleaned_movie_dataset.csv')
featuresForClient = ['id', 'title', 'popularity', 'vote_average', 'vote_count', 'genres', 'cast', 'director',
                     'keywords', 'overview']

@app.route('/getMovie/<title>', methods = ['GET'])
def get_movie(title:str):
    movie = movies.loc[movies['title'] == title][featuresForClient]
    return movie.to_json(orient='records')


@app.route('/selection-list/<k>', methods=['GET'])
def movie_selection(k):
    global movies, featuresForClient
    movies = movies.sort_values(by="popularity", ascending=False)
    return movies.head(int(k))[featuresForClient].to_json(orient='records', indent=4)


@app.route('/recommend/<movie_title>/<k>', methods=['GET'])
def movie_recommender(movie_title,k):
    global movies, featuresForClient
    movie_vectors = pd.read_csv('movie_vectors.csv', header=None)

    cvec = CountVectorizer()
    movie_vectors = pd.DataFrame(cvec.fit_transform(movie_vectors[0].astype(str)).toarray())

    def recommend_similar_movies(movie):
        movie = movie.lower()
        movie_index = movies[movies['title'].str.lower() == movie].index

        if len(movie_index) == 0:
            return []

        movie_index = movie_index[0]
        target_vector = movie_vectors.iloc[movie_index].values.reshape(1, -1)

        cos_sim = cosine_similarity(target_vector, movie_vectors).flatten()
        similar_indices = cos_sim.argsort()[-11:-1][::-1]

        recommendations = pd.DataFrame(movies.iloc[similar_indices])
        recommendations = recommendations[featuresForClient].to_json(orient='records', indent=4)
        return recommendations

    recommendations = recommend_similar_movies(movie_title)
    return recommendations


if __name__ == '__main__':
    app.run()
