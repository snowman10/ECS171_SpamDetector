import pandas as pd
import numpy as np
import seaborn
import nltk

from sklearn.model_selection import train_test_split 

data = pd.read_csv("spam.csv", encoding='latin-1')
data.drop(labels=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
data.rename(columns={'v1':'Class','v2':'Text'},inplace=True)
data['numClass'] = data['Class'].map({'ham':0, 'spam':1})

# nltk.download('stopwords')
# stopwords = nltk.corpus.stopwords.words('english')
# print(stopwords)

# print(data)

print(data)


#importing the Stemming function from nltk library
# from nltk.stem.porter import PorterStemmer
# #defining the object for stemming
# porter_stemmer = PorterStemmer()
 
# #defining a function for stemming
# def stemming(text):
#   stem_text = [porter_stemmer.stem(word) for word in text]
#   return stem_text
 
# # applying function for stemming
# data['Text']=data['Text'].apply(lambda x: stemming(x))
# data['Text']=data['Text'].apply(lambda x: ''.join(x))
# print(data)

# To do: STEMMING

# Split the data into train and test
X = data.drop('Class', axis = 1)
y = data['Class']

data_train, data_test, class_train, class_test = train_test_split(X, y, test_size=0.1)

