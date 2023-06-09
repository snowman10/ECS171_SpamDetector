import pandas as pd
import nltk
import string

from sklearn.model_selection import train_test_split 
from nltk.stem import SnowballStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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

def getBayesModel(data_train, class_train):
  bayes = MultinomialNB(alpha=0.550010) # Ideal Alpha
  bayes.fit(data_train, class_train)
  return bayes

def getRFModel(data_train, class_train):
  rf = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1)
  model = rf.fit(data_train, class_train)
  return model

def getLogisticModel(data_train, class_train):
  model = LogisticRegression(solver='liblinear', penalty='l1')
  model.fit(data_train, class_train)
  return model

def getDataSplit():
  data = readAndFormatData("spam.csv")
  data = data.drop_duplicates(keep="first")

  data_text, data_class = splitDataClass(data)
  data_text = data_text.apply(text_preprocess)
  data_text = data_text.apply(stemmer)

  return data_text, data_class

def NaiveBayes(output_dict=False):
  data_text, data_class = getDataSplit()

  vectorizer = TfidfVectorizer()
  matrix = vectorizer.fit_transform(data_text)
  data_train, data_test, class_train, class_test = train_test_split(
    matrix, data_class, test_size=0.1, random_state=42
  )

  model = getBayesModel(data_train, class_train)
  pred = model.predict(data_test)

  return model, classification_report(class_test, pred, output_dict=output_dict, target_names=["ham", "spam"]), vectorizer

def RandomForest(output_dict=False):
  data_text, data_class = getDataSplit()

  vectorizer = TfidfVectorizer()
  matrix = vectorizer.fit_transform(data_text)
  data_train, data_test, class_train, class_test = train_test_split(
    matrix, data_class, test_size=0.1, random_state=42
  )

  model = getRFModel(data_train, class_train)
  pred = model.predict(data_test)

  return model, classification_report(class_test, pred, output_dict=output_dict, target_names=["ham", "spam"]), vectorizer

def Logistic(output_dict=False):
  data_text, data_class = getDataSplit()

  vectorizer = TfidfVectorizer()
  matrix = vectorizer.fit_transform(data_text)
  data_train, data_test, class_train, class_test = train_test_split(
    matrix, data_class, test_size=0.1, random_state=42
  )

  model = getLogisticModel(data_train, class_train)
  pred = model.predict(data_test)

  return model, classification_report(class_test, pred, output_dict=output_dict, target_names=["ham", "spam"]), vectorizer