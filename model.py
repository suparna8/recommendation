#Import the flask module
import os
import pickle
import sys

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances


app = Flask(__name__)

class Recommend:
    def __init__(self, path):
        self.data_path = path
        df = pd.read_csv("sample30.csv")
        df = df.drop(['brand','categories','manufacturer','reviews_date','reviews_didPurchase','reviews_doRecommend','reviews_text','reviews_title','reviews_userCity','reviews_userProvince','user_sentiment'], axis = 1)
        df.drop_duplicates(subset = ['id', 'name', 'reviews_username'], inplace=True)
        df.dropna(subset=['reviews_username'], inplace=True)
        df.to_csv("generated_samples30.csv")
        # Test and Train split of the dataset.
        train, test = train_test_split(df, test_size=0.30, random_state=31)
        # Pivot the train ratings' dataset into matrix format in which columns are movies and the rows are user IDs.
        df_pivot = df.pivot(
            index='reviews_username',
            columns='name',
            values='reviews_rating'
        ).fillna(0)
        dummy_train = train.copy()
        dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)
        # Convert the dummy train dataset into matrix format.
        dummy_train = dummy_train.pivot(
            index='reviews_username',
            columns='name',
            values='reviews_rating'
        ).fillna(1)
        # Creating the User Similarity Matrix using pairwise_distance function.
        user_correlation = 1 - pairwise_distances(df_pivot, metric='cosine')
        user_correlation[np.isnan(user_correlation)] = 0
        # Create a user-movie matrix.
        df_pivot = train.pivot(
            index='reviews_username',
            columns='name',
            values='reviews_rating'
        )
        mean = np.nanmean(df_pivot, axis=1)
        df_subtracted = (df_pivot.T-mean).T
        # Creating the User Similarity Matrix using pairwise_distance function.
        user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
        user_correlation[np.isnan(user_correlation)] = 0
        user_correlation[user_correlation<0]=0
        user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))
        self.user_final_rating = np.multiply(user_predicted_ratings,dummy_train)
    

def recommend(self, user_name):
        result = self.user_final_rating.loc[user_name].sort_values(ascending=False)[0:20]
        model = pickle.load("model.pkl", "rw")
        vectorizer = pickle.load("vectorizer.pkl", "rw")
        final_recommendations = []
        for item in result :
        reviews = self.df[self.df["name"] == item]
        final_review = ""
        for review in reviews:
        final_review = final_review + " " +reviews
        feature_Vector = vectorizer.transform(final_review)
        if(model.predict(feature_Vector)):
        final_recommendations.push(item);
        return jsonify ({final_recommendations})
