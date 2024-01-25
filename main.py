import numpy as np
import pandas as pd
import re
import nltk
nltk.download('punkt')
nltk.download('corpus')
nltk.download('stopwords')
df = pd.read_csv("IMDB_Top250Engmovies2_OMDB_Detailed (1).csv")

#convert lowercase and remove numbers, punctuations,
df['clean_plot'] = df['Plot'].str.lower()
df['clean_plot'] = df['clean_plot'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
df['clean_plot'] = df['clean_plot'].apply(lambda x: re.sub('\s+', ' ', x))

#tokenize sentence

df['clean_plot'] = df['clean_plot'].apply(lambda x: nltk.word_tokenize(x))
#print(df['clean_plot'])

stop_words = nltk.corpus.stopwords.words('english')
plot = []
for sentence in df['clean_plot']:
    temp = []
    for word in sentence:
        if word not in stop_words or len(word) >= 3: #less than 3 is probably punctuation or extra word
            temp.append(word)
    plot.append(temp)

#print(plot)
df['clean_plot'] = plot #changing clean plot to plot without stop word and punctuation

#print(df['Title'])

df['Genre'] = df[('Genre')].apply(lambda x: x.split(','))
df['Actors'] = df[('Actors')].apply(lambda x: x.split(',')[:4])
df['Director'] = df[('Director')].apply(lambda x: x.split(','))

def clean(x): # x sentence
    temp = []
    for word in x:
        temp.append(word.lower().replace(' ', ''))
    return temp


df['Genre'] = [clean(x) for x in df['Genre']]
df['Actors'] = [clean(x) for x in df['Actors']]
df['Director'] = [clean(x) for x in df['Director']]

columns = ['clean_plot', 'Genre', 'Actors', 'Director']
temp = []
for i in range(len(df)):
    words = ''
    for col in columns:
        words+= ' '.join(df[col][i]) + ' '
    temp.append(words)

df['clean_input'] = temp
df = df[['Title', 'clean_input']] #processed data

#feature extraction

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
tfidf = TfidfVectorizer()
features = tfidf.fit_transform(df['clean_input'])
cosine_sim = cosine_similarity(features, features)

#movie recommender
index = pd.Series(df['Title'])


def recommend_movies(title):
    movies = []
    idx = index[index == title].index[0]
    #print(idx)
    score = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
    top10 = list(score.iloc[1:11].index)
    #print(top10)

    for i in top10:
        movies.append(df['Title'][i])
    return movies


print(recommend_movies(input("What movie do you need recommendations for? ")))









