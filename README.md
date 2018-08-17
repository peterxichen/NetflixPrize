# NetflixPrize
----------------------------------------------
Open competition sponsored by Netflix for best algorithm to predict user ratings for films based on previous ratings. Project for COS 424.

Abstract
----------------------------------------------

In this study, we analyze Netflix data in which 2,649,430 users have provided rat-ings on a 1 to 5 scale for 17,770 movies to predict scores for a given movie anduser pairing.  We implement a variety of different methods, including normaliza-tion of global variables, linear regression, and latent class clustering. We observehere that simpler normalization and regression models produce better predictionsthan the more complex clustering models.  Finally, we demonstrate a method toextend the dataset using IMDb ratings and sentiment analysis from the movie’stitles to improve the performance of our models.

Deployment
----------------------------------------------

This directory comprises the following files

- cluster.py, predicts for ratings using our clustering technique (performing truncated SVD to perform feature selection, then use  t-SNE to graph clusters and select best parameters, then run MiniBatchKMeans to cluster on the training dataset), then regressing on the clusters

- netflix_full_csr.npz, matrix containing ratings on the 1 to 5 for 17,770 movies from 2,649,430 users.

- movie_titles.txt, contains the movie titles, reference ID, and release year.

- imdb_scraper.py, uses IMDbPY to scrape average user ratings and genre information for each movie from IMDb

- normalize.py, predicts for ratings using global effects normalization

- regress.py, predicts for ratings using movie-centric approach to ordinary least squares regression

- sentiment.py, use Python package VADER to assign a sentiment score (-1 to +1) for each movie title

Authors
----------------------------------------------

- Peter Chen
- Ben Burgess
