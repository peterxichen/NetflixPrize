import pandas as pd
from imdb import IMDb

ia = IMDb()

# contains titles and release years associated with each ID
movie_titles = pd.read_csv('movie_titles.txt', names=['col'],
                           engine = 'python', sep='\t')
movie_titles = pd.DataFrame(movie_titles.col.str.split(',',2).tolist(),
                            columns = ['ID','Year','Name'])

# pulls genre and ratings from IMDB
movie_titles["Genres"] = ""
movie_titles["Rating"] = ""
i = 0
for index, row in movie_titles.iterrows():
    title = row["Name"]
    s_result = ia.search_movie(title)
    if len(s_result) == 0:
        continue
    movie = s_result[0]
    ia.update(movie)
    try:
        rating = movie['rating']
    except:
        continue
    try:
        genres = movie['genre']
    except:
        continue
    row["Rating"] = rating
    row["Genres"] = genres
    print(i)
    i = i + 1

# export to csv
movie_titles.to_csv('temp.csv')

