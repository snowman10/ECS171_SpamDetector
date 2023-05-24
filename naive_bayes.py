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

count_Class=pd.value_counts(data["Class"], sort= True)
data = data.drop("Class", axis=1)
# count_Class.plot(kind = 'bar',color = ["green","red"])
# plt.title('Bar Plot')
# plt.show()

f = feature_extraction.text.CountVectorizer(stop_words = 'english')
X = f.fit_transform(data_text)
np.shape(X)

X_train, X_test, y_train, y_test = model_selection.train_test_split(
  X, data['numClass'], test_size=0.70, random_state=42)


list_alpha = np.arange(1/100000, 20, 0.11)
score_train = np.zeros(len(list_alpha))
score_test = np.zeros(len(list_alpha))
recall_test = np.zeros(len(list_alpha))
precision_test= np.zeros(len(list_alpha))
count = 0
for alpha in list_alpha:
    bayes = naive_bayes.MultinomialNB(alpha=alpha)
    bayes.fit(X_train, y_train)
    score_train[count] = bayes.score(X_train, y_train)
    score_test[count]= bayes.score(X_test, y_test)
    recall_test[count] = metrics.recall_score(y_test, bayes.predict(X_test))
    precision_test[count] = metrics.precision_score(y_test, bayes.predict(X_test))
    count = count + 1

matrix = np.matrix(np.c_[list_alpha, score_train, score_test, recall_test, precision_test])
models = pd.DataFrame(data = matrix, columns = 
             ['alpha', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
models.head(n=10)
best_index = models['Test Accuracy'].idxmax()
print(models.iloc[best_index, :])




rf = RandomForestClassifier(n_estimators=100,max_depth=None,n_jobs=-1)
rf_model = rf.fit(X_train,y_train)

y_pred=rf_model.predict(X_test)
precision,recall,fscore,support =score(y_test,y_pred,pos_label=1, average ='binary')
print('Precision : {} / Recall : {} / fscore : {} / Accuracy: {}'.format(round(precision,3),round(recall,3),round(fscore,3),round((y_pred==y_test).sum()/len(y_test),3)))
