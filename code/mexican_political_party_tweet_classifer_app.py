# Importing modules
import nltk
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import streamlit as st
import tensorflow as tf
import time

from nltk import word_tokenize
from nltk.stem import SnowballStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from tqdm.auto import tqdm
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification


nltk.download('punkt')

# Importing Spanish stop words list
with open('spanish.txt') as file:
    spanish = file.read().split()

# Importing Spanish stemming tokenizer
def stemming_tokenizer(doc):
    stemming = SnowballStemmer('spanish')
    return [stemming.stem(w) for w in word_tokenize(doc)]

# App Architecture
st.title('Mexican Political Party Tweet Classifer') 

st.header("Let's predict which Mexican political party tweeted a particular tweet")

st.subheader('Tweet must be from one of the 6 official Twitter accounts associated with Mexican political parties: @AccionNacional, @PRI_Nacional, @PRDMexico, @partidoverdemex, @MovCiudadanoMX, or @PartidoMorenaMx')

# User Input
tweet = st.text_input("Enter tweet from official Twitter account of a Mexican political party:", max_chars = 320)

# Loading saved scikit-learn model
with open("./saved-sk_learn-model.pkl", "rb") as f:
    sk_learn_classifier = pickle.load(f)

# Loading saved Tensorflow model
@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model('./tensorflow_model')

twitter_user_model = load_model()
tokenizer = AutoTokenizer.from_pretrained("M47Labs/spanish_news_classification_headlines_untrained")

def prepare_data(input_text, tokenizer):
    token = tokenizer.encode_plus(
        input_text,
        max_length=280, 
        truncation=True, 
        padding='max_length', 
        add_special_tokens=True,
        return_tensors='tf'
    )
    return {
        'input_ids': tf.cast(token.input_ids, tf.float64),
        'attention_mask': tf.cast(token.attention_mask, tf.float64)
    }

def make_prediction(model, processed_data, classes=['partidoverdemex' 'pri_nacional', 'accionnacional', 'partidomorenamx', 'prdmexico', 'movciudadanomx']):
    probs = model.predict(processed_data)[0]
    return classes[np.argmax(probs)]

input_text = tweet
processed_data = prepare_data(input_text, tokenizer)
result = make_prediction(twitter_user_model, processed_data=processed_data)
result








# def prepare_data(tweet, tokenizer):
#     token = tokenizer.encode_plus(
#         tweet,
#         max_length=280, 
#         truncation=True, 
#         padding='max_length', 
#         add_special_tokens=True,
#         return_tensors='tf'
#     )
#     return {
#         'input_ids': tf.cast(token.input_ids, tf.float64),
#         'attention_mask': tf.cast(token.attention_mask, tf.float64)
#     }

# processed_data = prepare_data(tweet, tokenizer)


# def make_prediction(model, processed_data, classes=['partidoverdemex' 'pri_nacional', 'accionnacional', 'partidomorenamx', 'prdmexico', 'movciudadanomx']):
#     probs = model.predict(processed_data)[0]
#     return classes[np.argmax(probs)]


# def main():
#     prediction = None
#     model = tf_model

#     with st.spinner("Predicting using Tensorflow Model..."):
#             if tweet is not None:
#                 prediction = make_prediction(tf_model, processed_data=processed_data)
                    
#                 time.sleep(5)    
#     if prediction is not None:
#         st.write(f'The model prediction is...{prediction}')
    
# if __name__ == '__main__':
#     main()                









# # Predictions
# st.subheader('Scikit-Learn Model Prediction (CountVectorizer and Multinomial Naive Bayes Classifier):')

# # Scikit-Learn Prediction
# with st.spinner("Predicting using Scikit-Learn Model..."):
#     time.sleep(2)
#     sk_learn_prediction = sk_learn_classifier.predict([tweet])
# sk_learn_prediction

# # Tensorflow Prediction
# st.subheader('Tensorflow Model Prediction (BERT Tokenizer and Transfer Learning):')

# with st.spinner("Predicting using Tensorflow Model..."):
#     time.sleep(2)
#     tensorflow_prediction = make_prediction(model, processed_data=processed_data)
# tensorflow_prediction
