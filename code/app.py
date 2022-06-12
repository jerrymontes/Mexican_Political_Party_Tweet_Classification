import streamlit as st
import pickle
import numpy as np
import time

import nltk
import numpy as np
import pandas as pd
import seaborn as sns

from nltk import word_tokenize
from nltk.stem import SnowballStemmer

from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


nltk.download('punkt')

with open('spanish.txt') as file:
    spanish = file.read().split()

def stemming_tokenizer(doc):
    stemming = SnowballStemmer('spanish')
    return [stemming.stem(w) for w in word_tokenize(doc)]

st.title('Mexican Political Party Tweet Classifer') 

st.header("Let's predict which Mexican political party tweeted a particular tweet")

st.subheader('Tweet must be from one of the 6 official Twitter accounts associated with Mexican political parties: @AccionNacional, @PRI_Nacional, @PRDMexico, @partidoverdemex, @MovCiudadanoMX, or @PartidoMorenaMx')

tweet = st.text_input("Enter tweet from official Twitter account of a Mexican political party:", max_chars = 320)

with open("./saved-sk_learn-model.pkl", "rb") as f:
    sk_learn_classifier = pickle.load(f)

# with open("saved-tensorflow-model.pkl", "rb") as tf:
#     tensorflow_classifier = pickle.load(tf)

st.subheader('Scikit-Learn Model Prediction (CountVectorizer and Multinomial Naive Bayes Classifier):')

with st.spinner("Predicting using Scikit-Learn Model..."):
    time.sleep(2)
    sk_learn_prediction = sk_learn_classifier.predict([tweet])
sk_learn_prediction

# st.subheader('Tensorflow Model Prediction (BERT Tokenizer and Transfer Learning):')

# with st.spinner("Predicting using Tensorflow Model..."):
#     time.sleep(2)
#     tensorflow_prediction = predict_twitter_handle(tensorflow_classifier, tweet)
# tensorflow_prediction