import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# contains titles and release years associated with each ID
movie_titles = pd.read_csv('movie_titles.txt', names=['col'],
                           engine = 'python', sep='\t')
movie_titles = pd.DataFrame(movie_titles.col.str.split(',',2).tolist(),
                            columns = ['ID','Year','Name'])

# assign a sentiment score (-1 to +1) for each movie title
sid = SentimentIntensityAnalyzer()
movie_titles["Sentiment"] = ""
i = 0
for index, row in movie_titles.iterrows():
    title = row["Name"]
    score = sid.polarity_scores(title)['compound']
    row["Sentiment"] = score
    print(i)
    i = i + 1

# export to csv
movie_titles.to_csv('temp.csv')
