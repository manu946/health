import joblib
from fastapi import FastAPI
import uvicorn
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle
import sklearn
import string
from gensim.models import KeyedVectors
import spacy

nl = spacy.blank('en')
lemmatizer = WordNetLemmatizer()
stp = ['i',
       'me',
       'my',
       'myself',
       'we',
       'our',
       'ours',
       'ourselves',
       'you',
       "you're",
       "you've",
       "you'll",
       "you'd",
       'your',
       'yours',
       'yourself',
       'yourselves',
       'he',
       'him',
       'his',
       'himself',
       'she',
       "she's",
       'her',
       'hers',
       'herself',
       'it',
       "it's",
       'its',
       'itself',
       'they',
       'them',
       'their',
       'theirs',
       'themselves',
       'what',
       'which',
       'who',
       'whom',
       'this',
       'that',
       "that'll",
       'these',
       'those',
       'am',
       'is',
       'are',
       'was',
       'were',
       'be',
       'been',
       'being',
       'have',
       'has',
       'had',
       'having',
       'do',
       'does',
       'did',
       'doing',
       'a',
       'an',
       'the',
       'and',
       'but',
       'if',
       'or',
       'because',
       'as',
       'until',
       'while',
       'of',
       'at',
       'by',
       'for',
       'with',
       'about',
       'against',
       'between',
       'into',
       'through',
       'during',
       'before',
       'after',
       'above',
       'below',
       'to',
       'from',
       'up',
       'down',
       'in',
       'out',
       'on',
       'off',
       'over',
       'under',
       'again',
       'further',
       'then',
       'once',
       'here',
       'there',
       'when',
       'where',
       'why',
       'how',
       'all',
       'any',
       'both',
       'each',
       'few',
       'more',
       'most',
       'other',
       'some',
       'such',
       'no',
       'nor',
       'not',
       'only',
       'own',
       'same',
       'so',
       'than',
       'too',
       'very',
       's',
       't',
       'can',
       'will',
       'just',
       'don',
       "don't",
       'should',
       "should've",
       'now',
       'd',
       'll',
       'm',
       'o',
       're',
       've',
       'y',
       'ain',
       'aren',
       "aren't",
       'couldn',
       "couldn't",
       'didn',
       "didn't",
       'doesn',
       "doesn't",
       'hadn',
       "hadn't",
       'hasn',
       "hasn't",
       'haven',
       "haven't",
       'isn',
       "isn't",
       'ma',
       'mightn',
       "mightn't",
       'mustn',
       "mustn't",
       'needn',
       "needn't",
       'shan',
       "shan't",
       'shouldn',
       "shouldn't",
       'wasn',
       "wasn't",
       'weren',
       "weren't",
       'won',
       "won't",
       'wouldn',
       "wouldn't"]

app = FastAPI()
ml_model = pickle.load(open('model.pkl', 'rb'))
lb_encoder = joblib.load('label_encoder.joblib')
model = KeyedVectors.load("word2vec.wordvectors", mmap='r')


def transform_text(text):
    text = str(text)
    text = text.lower()
    # word token
    text = nl(text)
    # removing special chars
    y = []
    for i in text:
        i = str(i)
        # only numbers and alphbets
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for ix in text:
        if ix not in stp and ix not in string.punctuation:
            y.append(ix)
    # removing stopwords
    # Stemming
    text = y[:]
    y.clear()
    for prt in text:
        y.append(lemmatizer.lemmatize(prt))

    return " ".join(y)


def doc_vector(doc):
    doc = [word for word in doc.split() if word in model]
    return np.mean(model[doc], axis=0)


@app.get('/')
def hh():
    return 'hello'


@app.get('/predict')
def gt(query: str, gender: str):
    gender = gender.lower()
    encoded = 1 if gender == "male" else 0
    inp = doc_vector(transform_text(query))
    gender = np.array([encoded, ])
    xx = np.hstack((inp, gender))
    pred = ml_model.predict_proba([xx])
    fx = []
    for xx in pred[0]:
        if xx > 0.0:
            fx.append(xx)

    returned_dic = {}
    nd = np.argpartition(fx, -5)[-5:]
    nd = list(nd)
    index = 0
    for gg in lb_encoder.inverse_transform(nd):
        returned_dic[f'{index}'] = gg
        index += 1
    return returned_dic


