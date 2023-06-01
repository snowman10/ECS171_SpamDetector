import pandas as pd
import nltk
import string

from sklearn.model_selection import train_test_split 
from nltk.stem import SnowballStemmer

from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split 

from sklearn.ensemble import RandomForestClassifier

def readAndFormatData(file):
  data = pd.read_csv(file, encoding='latin-1')
  data.drop(labels=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
  data.rename(columns={'v1':'Class','v2':'Text'},inplace=True)
  data['numClass'] = data['Class'].map({'ham':0, 'spam':1})
  return data

def getStopwords():
  # nltk.download('stopwords')
  stopwords = nltk.corpus.stopwords.words('english')
  return stopwords

def splitDataClass(data):
  x = data.drop(columns=["Class", "numClass"], axis=1)
  x = x["Text"].copy()
  y = data["numClass"]
  return x, y

def text_preprocess(text):
  stopwords = getStopwords()

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

def getModel(data_train, class_train):
  rf = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1)
  model = rf.fit(data_train, class_train)
  return model