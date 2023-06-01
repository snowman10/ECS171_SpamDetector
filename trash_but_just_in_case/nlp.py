import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, LSTM, Dropout, Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
import tensorflow as tf

import pandas as pd
import numpy as np
import seaborn
import nltk
import sklearn as sk
import string

from sklearn.model_selection import train_test_split 
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer


import matplotlib.pyplot as plt
import scipy as sp

from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support as score

def readAndFormatData(file):
  data = pd.read_csv(file, encoding='latin-1')
  data.drop(labels=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
  data.rename(columns={'v1':'Class','v2':'Text'},inplace=True)
  data['numClass'] = data['Class'].map({'ham':0, 'spam':1})
  return data

def getStopwords():
  nltk.download('stopwords')
  stopwords = nltk.corpus.stopwords.words('english')
  return stopwords

def splitTrainTest(data):
  X = data.drop(columns=['Class', 'numClass'], axis = 1)
  X = X['Text'].copy()
  y = data['Class']
  return X, y

def text_preprocess(text):
  text = text.translate(str.maketrans('', '', string.punctuation))
  text = [word for word in text.split() if word.lower() not in stopwords]
  return " ".join(text)

def stemmer(text):
  text = text.split()
  words = ""
  for i in text:
    stemmer = SnowballStemmer("english")
    words += (stemmer.stem(i))+" "
  return words

def runVectorizer(data_text, data_class):
  vectorizer = TfidfVectorizer()
  matrix = vectorizer.fit_transform(data_text)
  data_train, data_test, class_train, class_test = train_test_split(matrix, data_class, test_size=0.1, random_state=13)

  model = LogisticRegression(solver='liblinear', penalty='l1')
  model.fit(data_train, class_train)
  pred = model.predict(data_test)
  accuracy = accuracy_score(class_test,pred)
  return accuracy, data_train, data_test, class_train, class_test, vectorizer, model

data = readAndFormatData("spam.csv")
data = data.drop_duplicates(keep="first")
stopwords = getStopwords()
data_text, data_class = splitTrainTest(data)
data_text = data_text.apply(text_preprocess)
data_text = data_text.apply(stemmer)

vocab_size = 400
oov_tok = ""
max_length = 250
embedding_dim = 16
encode = ({'ham': 0, 'spam': 1} )
#new dataset with replaced values

X = data_text
Y = data_class
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X)
# convert to sequence of integers
X = tokenizer.texts_to_sequences(X)

X = np.array(X)
y = np.array(Y)
     

X = pad_sequences(X, maxlen=max_length)
     

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 50
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.20, random_state=7)
history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test,y_test), verbose=2)