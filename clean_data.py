import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

movies = pd.read_csv('./movie_dataset.csv')

# cleaning movie data
movies['title'] = movies['title'].str.lower()
selected_features = ['id', 'title', 'popularity', 'vote_average', 'vote_count', 'genres', 'cast', 'director',
                     'keywords', 'overview', 'tagline']
movies = movies[selected_features]
imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value="")
movies = pd.DataFrame(imputer.fit_transform(movies), columns=selected_features)

# creating movie_vectors
cat_vars = [ 'title', 'genres', 'director', 'keywords', 'overview', 'tagline']
movie_vectors = movies[cat_vars]
movie_vectors = movie_vectors.apply(lambda x: ' '.join(x.astype(str)), axis=1)


#saving data
movies.to_csv('cleaned_movie_dataset.csv')
movie_vectors.to_csv('movie_vectors.csv', header=False)
