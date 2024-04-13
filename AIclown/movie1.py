from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import PorterStemmer

app = Flask(__name__, template_folder='templates1')

a = pd.read_csv("tmdb_5000_movies.csv")
b = pd.read_csv("tmdb_5000_credits.csv")
merged_df = a.merge(b, on="title")
merged_df.dropna(subset=['overview'], inplace=True)

def extract_genres(text):
    genres_list = []
    for genre in ast.literal_eval(text):
        genres_list.append(genre["name"])
    return genres_list

def extract_keywords(text):
    keywords_list = []
    for keyword in ast.literal_eval(text):
        keywords_list.append(keyword["name"])
    return keywords_list

def fetch_director(text):
    for i in ast.literal_eval(text):
        if i["job"] == "Director":
            return i["name"]
    return None

merged_df["genres"] = merged_df["genres"].apply(extract_genres)
merged_df["keywords"] = merged_df["keywords"].apply(extract_keywords)
merged_df["director"] = merged_df["crew"].apply(fetch_director)

merged_df["overview"] = merged_df["overview"].apply(lambda x: " ".join(x.split()) if isinstance(x, str) else "")

merged_df["keywords"] = merged_df["keywords"].apply(lambda x: [word.replace(" ", "") for word in x])
merged_df["genres"] = merged_df["genres"].apply(lambda x: [word.replace(" ", "") for word in x])
merged_df["tags"] = merged_df["overview"] + merged_df["genres"].astype(str) + merged_df["keywords"].astype(str)
merged_df["tags"] = merged_df["tags"].apply(lambda x: " ".join(x.split()) if isinstance(x, str) else x)
merged_df["tags"] = merged_df["tags"].apply(lambda x: x.lower() if isinstance(x, str) else x)

new_df = merged_df[["id", "title", "tags"]]

nltk.download('punkt')
from nltk.stem import PorterStemmer
ps = PorterStemmer()

def stems(text):
    l = []
    for i in text.split():
        l.append(ps.stem(i))
    return " ".join(l)

new_df["tags"] = new_df["tags"].apply(stems)

cv = CountVectorizer(max_features=5000, stop_words="english")
vector = cv.fit_transform(new_df["tags"]).toarray()
similarly = cosine_similarity(vector)

def recommend(movie):
    try:
        movie = movie.lower()  
        index = new_df[new_df["title"].str.lower() == movie].index[0]  
        distances = sorted(list(enumerate(similarly[index])), reverse=True, key=lambda x: x[1])
        recommendations = [new_df.iloc[i[0]].title for i in distances[1:6]]
        return recommendations
    except Exception as e:
        print(f"Error: {e}")
        return []

@app.route('/', methods=['GET','POST'])
def index():
    return render_template('index1.html')

@app.route('/recommend', methods=['POST'])
def get_recommendation():
    movie_name = request.form.get('movie_name')
    recommendations = recommend(movie_name)
    return render_template('result1.html', movie_name=movie_name, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
