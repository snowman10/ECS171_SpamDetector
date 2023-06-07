import pandas as pd
import numpy as np
import nltk
import string

from sklearn.model_selection import train_test_split 
from nltk.stem import SnowballStemmer

from sklearn import feature_extraction, model_selection, metrics
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support as score

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
  X = data.drop(columns=['Class', 'numClass'], axis = 1)
  X = X['Text'].copy()
  y = data['numClass']
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

data = readAndFormatData("spam.csv")
data = data.drop_duplicates(keep="first")

stopwords = getStopwords()
data_text, data_class = splitDataClass(data)
data_text = data_text.apply(text_preprocess)
data_text = data_text.apply(stemmer)

f = feature_extraction.text.CountVectorizer(stop_words="english")
x = f.fit_transform(data_text)

x_train, x_test, y_train, y_test = train_test_split(
  x, data_class, test_size=0.1, random_state=13
)

list_alpha = np.arange(1/100000, 20, 0.11)
score_train = np.zeros(len(list_alpha))
score_test = np.zeros(len(list_alpha))
recall_test = np.zeros(len(list_alpha))
precision_test= np.zeros(len(list_alpha))
count = 0

for alpha in list_alpha:
  bayes = MultinomialNB(alpha=alpha)
  bayes.fit(x_train, y_train)
  score_train[count] = bayes.score(x_train, y_train)
  score_test[count]= bayes.score(x_test, y_test)
  recall_test[count] = metrics.recall_score(y_test, bayes.predict(x_test))
  precision_test[count] = metrics.precision_score(y_test, bayes.predict(x_test))
  count = count + 1

matrix = np.matrix(np.c_[list_alpha, score_train, score_test, recall_test, precision_test])
models = pd.DataFrame(data = matrix, columns = 
             ['alpha', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
models.head(n=10)
best_index = models['Test Accuracy'].idxmax()
print(models.iloc[best_index, :])