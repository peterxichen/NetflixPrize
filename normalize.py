import numpy as np
import pandas as pd
import math
from scipy.sparse import csr_matrix
import scipy.sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score

# contains titles and release years associated with each ID
movie_titles = pd.read_csv('movie_titles.txt', names=['col'],
                           engine = 'python', sep='\t')
movie_titles = pd.DataFrame(movie_titles.col.str.split(',',2).tolist(),
                            columns = ['ID','Year','Name'])

# sparse matrix of movies by user
# each element a rating (1-5) or nonresponse (0)
ratings_csr = scipy.sparse.load_npz('netflix_full_csr.npz')
print('Shape of matrix: ', ratings_csr.shape)

"""
# Perform a truncated SVD suited for sparse datasets
# Analyze which movies are associated with different latent dimensions
n_components = 5
svd = TruncatedSVD(n_components = n_components)
Z = svd.fit_transform(ratings_csr)
components = svd.components_
print(svd.explained_variance_ratio_)
for i in range(0,n_components):
    Z_sort = np.argsort(np.abs(Z[:,i]))
    print('Component ' + str(i))
    for j in range(1,5):
        movie_index = Z_sort[-j]
        movie_title = movie_titles[movie_titles['ID'] == str(movie_index)]['Name']
        movie_weight = Z[movie_index,i]
        print(str(movie_title) + '\nWeight:' + str(movie_weight))
    print(' ')
"""

# Global Effects Normalization

# gather nonzero ratings
test = scipy.sparse.find(ratings_csr)
rows = test[0]
cols = test[1]
vals = test[2]

# rating as sum of overall average + customer offset + movie offset
ratings_sum = ratings_csr.sum()
num_ratings = (ratings_csr!=0).sum()
y_pred = []
for i in range(len(vals)):
    print(i/len(vals))
    base_rating = (ratings_sum - vals[i])/(num_ratings-1)
    users = ratings_csr[rows[i]]
    movies = ratings_csr[:cols[i]]
    user_effect = (users.sum()-vals[i])/((users!=0).sum()-1)
    if (math.isnan(user_effect) or math.isinf(user_effect)):
        user_effect = 0
    else:
        user_effect=user_effect-base_rating
    movie_effect = (movies.sum()-vals[i])/((movies!=0).sum()-1)
    if (math.isnan(movie_effect) or math.isinf(movie_effect)):
        movie_effect=0
    else:
        movie_effect=movie_effect-base_rating
    y_pred.append(base_rating+user_effect+movie_effect)
r2 = r2_score(vals, y_pred)
print(r2)

