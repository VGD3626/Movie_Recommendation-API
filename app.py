from flask import Flask, jsonify
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

@app.route('/recommend/<movie_title>', methods=['POST'])
def movie_recommender(movie_title):
    movies = pd.read_csv('cleaned_movie_dataset.csv')
    movie_vectors = pd.read_csv('movie_vectors.csv', header=None)

    cvec = TfidfVectorizer()
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

        recommendations = movies.iloc[similar_indices]['title'].tolist()
        return recommendations

    recommendations = recommend_similar_movies(movie_title)
    return jsonify(recommendations)


if __name__ == '__main__':
    app.run()
