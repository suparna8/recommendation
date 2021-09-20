#Import the flask module
from flask import Flask,  render_template, jsonify
import os
import sys
import pandas as pd 
import numpy as np
from scipy.sparse import csr_matrix
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from model import Recommend

app = Flask(__name__)

r = Recommend("sample30.csv")


@app.route('/',  methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/recommend/<username>', methods=['GET'])
def recommend(username):
    return r.recommend(username)