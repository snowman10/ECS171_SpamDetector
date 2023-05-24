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

accuracy, data_train, data_test, class_train, class_test, vectorizer, model = runVectorizer(data_text, data_class)
print(accuracy)

# seaborn.regplot(x=matrix[1], y=data_class, logistic=True, ci=None)
print(data_test[0])
data = input()
data = vectorizer.fit_transform([data])
print(data)
pred = model.predict(data)
print(pred)

# pred = model.predict(np.array(input()).reshape(1,-1))
# print(pred)





# If we need more pages in our report (0.0001 improvement)

# length = message_data['length'].as_matrix()
# new_mat = np.hstack((message_mat.todense(),length[:, None]))
# message_train, message_test, spam_nospam_train, spam_nospam_test = train_test_split(new_mat, 
#                                                         message_data['Spam/Not_Spam'], test_size=0.3, random_state=20)
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# Spam_model = LogisticRegression(solver='liblinear', penalty='l1')
# Spam_model.fit(message_train, spam_nospam_train)
# pred = Spam_model.predict(message_test)
# accuracy_score(spam_nospam_test,pred)