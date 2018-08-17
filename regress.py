import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import scipy.sparse
from sklearn import linear_model
from sklearn.model_selection import train_test_split
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

# remove inactive reviewers
ratings = ratings_csr[:,ratings_csr.getnnz(0)>0]
print('Shape of matrix: ', ratings.shape)

# run movie-centric regression
def reg(movie_id):
    reg = linear_model.LinearRegression()
    mov = ratings[movie_id]
    temp = scipy.sparse.find(mov)
    y = temp[2]
    mov_mat = np.ravel(np.full((ratings.shape[0], 1), True))
    mov_mat[movie_id] = False # matrix of values to be regressed on
    if (len(y) > 1000):
        n_users = 1000
        random_sample_users = np.random.choice(len(y), size = n_users)
        y = y[random_sample_users]
        X = ratings[mov_mat]
        X = X[:,temp[1][random_sample_users]].transpose().todense()
    else:
        X = ratings[mov_mat]
        X = X[:,temp[1]].transpose().todense()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    return(r2_score(y_test, y_pred))

r2 = []
for i in range(1,17771):
    r2.append(reg(i))
    print(i)
    

    
