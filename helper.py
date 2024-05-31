import pickle
import pandas as pd
import numpy as np
import nltk
import gensim
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity 

wordmodel = pickle.load(open('cv.pkl','rb'))


# Stop word removal & Lemmatization
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocess 
def preprocess_text(text):
    # Tokenization
    words = word_tokenize(text.lower())
    # Remove stopwords and non-alphabetic tokens
    words = [word for word in words if word.isalpha() and word not in stop_words]
    # Lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

# Document Vector
def doc_vec(tokens):
    embeddings = [wordmodel.wv[word] for word in tokens if word in wordmodel.wv.index_to_key]
    if len(embeddings) > 0:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(100)

# Common Words
def test_common_words(q1,q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))    
    return len(w1 & w2)

# Total Words
def test_total_words(q1,q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))    
    return (len(w1) + len(w2))

# Transforming Columns
def quest(q1,q2):
    query = []
    
    #length of strings in both question 
    query.append(len(q1))
    query.append(len(q2))
    
    #words in both questions
    query.append(len(q1.split(" ")))
    query.append(len(q2.split(" ")))
    
    #Common words
    query.append(test_common_words(q1,q2))
    
    #Total Words
    query.append(test_total_words(q1,q2))
    
    #Word Share
    query.append(round(test_common_words(q1,q2)/test_total_words(q1,q2),2))
    
    #cosine similarity
    token1 = preprocess_text(q1)
    token2 = preprocess_text(q2)
    vector1 = doc_vec(token1).reshape(1, -1)
    vector2 = doc_vec(token2).reshape(1, -1)
    cos_sim = cosine_similarity(vector1,vector2)[0,0]
    query.append(cos_sim)
    
    return np.array(query).reshape(1,8)




    






