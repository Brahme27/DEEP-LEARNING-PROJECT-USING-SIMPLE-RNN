import numpy as np 
import tensorflow as tf  
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


##load imdb
word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}

#Load the pretrained model with RELU activation
model=load_model('Simple_RNN_IMDB_Model.h5')


def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?')for i in encoded_review])


def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review


#prediction function
def predict(review):
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input)

    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'

    return sentiment,prediction[0][0]


import streamlit as st  
st.title("IMDB Movie Review Sentimental Analysis")
st.write("Enter a Movie review to classify it as positive or negative")


#User Inpur
user_input=st.text_area("Movie_review")


if st.button('Classify'):
    preprocess_input=preprocess_text(user_input)

    #make prediction
    prediction=model.predict(preprocess_input)
    sentiments='Positive' if prediction[0][0]>0.5 else 'Negative'


    #Display the result
    st.write("The sentiment of the review is: ",sentiments)
    st.write("Prediction score",prediction[0][0])
else:
    st.write("please enter a movie review")