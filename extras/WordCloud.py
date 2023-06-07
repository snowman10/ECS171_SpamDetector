import wordcloud
import pandas
import matplotlib

# Python program to generate WordCloud
 
# importing all necessary modules
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
 
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

# data = data[data['Class'] == 'ham'] 



data = data.drop_duplicates(keep="first")
stopwords = getStopwords()
data_text, data_class = splitTrainTest(data)
data_text = data_text.apply(text_preprocess)
data_text = data_text.apply(stemmer)

df = data_text


# print(data)
# exit(1)
# df1 = data[data['Text'] == 'ham'] 
# df = df1
 
comment_words = ''
stopwords = set(STOPWORDS)
 
# iterate through the csv file
for val in df.values:
     
    # typecaste each val to string
    val = str(val)
 
    # split the value
    tokens = val.split()
     
    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
     
    comment_words += " ".join(tokens)+" "
 
wordcloud = WordCloud(width = 1600, height = 800,
                background_color = 'white',
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)
 
# plot the WordCloud image                      
# plt.figure(figsize = (16, 8), facecolor = None)
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.tight_layout(pad = 0)


# print(data["Class"].value_counts())
# x = [100,100]

# print(data)
# x = len(df[df.Text == "ham"])
# y = len(df[df.Text == "spam"])

# x = [x,y]


plt.pie(data["Class"].value_counts(), labels = ["Not Spam", "Spam"], autopct = "%0.2f", colors=["forestgreen", "skyblue"])
plt.title("Dataset Contents")
plt.show()