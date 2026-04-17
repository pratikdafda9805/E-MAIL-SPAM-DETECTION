import streamlit as st
import pickle
import string
# from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = pickle.load(open('vectorizer.pkl','rb'))
mnb = pickle.load(open('model.pkl','rb'))
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def text_preprocessing(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)



st.title('Email Spam Detection')

input_sms = st.text_area ("Enter your email")
if st.button('Predict'):
    #preprocessing
    transform_sms = text_preprocessing(input_sms)
    #vectorizer
    vector =tfidf.transform([transform_sms])
    #predict
    prediction = mnb.predict(vector)[0]
    #display
    if prediction == 1 :
        st.header('Email spam detection')
    else:
        st.header('Email ham detection')