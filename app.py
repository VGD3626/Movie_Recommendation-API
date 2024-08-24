from flask import Flask
import requests
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('./movie_dataset.csv')

app = Flask(__name__)


@app.route('/recommend/<movie_title>', methods=['POST'])
def movie_recommender(movie_title):
    global movies

    # Data Pre-processing
    movies['release_year'] = pd.to_datetime(movies.loc['release_date']).dt.year
    movies['title'] = movies['title'].str.lower()

    selected_features = ['id', 'title', 'popularity', 'vote_average', 'vote_count', 'genres', 'cast', 'director',
                         'keywords', 'overview',
                         'release_year', 'tagline']
    movies = movies[selected_features]

    imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value="")
    movies = pd.DataFrame(imputer.fit_transform(movies), columns=selected_features)

    cat_vars = ['title', 'genres', 'director', 'keywords', 'overview', 'tagline']
    movie_vectors = movies[cat_vars]
    movie_vectors = movie_vectors.apply(lambda x: ' '.join(x.astype(str)), axis=1)

    cvec = TfidfVectorizer()
    movie_vectors = cvec.fit_transform(list(movie_vectors))

    def recommend_similar_movies(movie):
        movie = movie.lower()
        movie_index = movies.loc[movies['title'] == movie].index

        if len(movie_index) == 0:
            print(f"Movie titled '{movie}' not found.")
            return

        cos_sim = []

        for i in range(movie_vectors.shape[0]):
            if i != movie_index:
                similarity = cosine_similarity(movie_vectors[movie_index], movie_vectors[i])
                cos_sim.append((i, similarity))

        cos_sim.sort(key=lambda x: x[1], reverse=True)

        k = 10
        recommendations = []
        for (i, _) in cos_sim[0:k]:
            recommendations.append(movies.iloc[i]['title'])
        return recommendations

    recommendations = recommend_similar_movies(movie_title)
    v=[]
    for movie in recommendations:
        movie_id = int(movies[movies['title'] == movie]['id'])
        url = f"https://movies-api14.p.rapidapi.com/movie/{movie_id}"
        headers = {
            "x-rapidapi-key": "e56d68f485msh320b2cb83f8dedep1ca4d7jsnb4259afc8402",
            "x-rapidapi-host": "movies-api14.p.rapidapi.com"
        }
        v.append(requests.get(url, headers=headers).json())

    print(v)
    return v


if __name__ == '__main__':
    app.run()
