from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer("english")



porter_stemmer = PorterStemmer()
# #defining a function for stemming
def stemming(text):
  stem_text = [porter_stemmer.stem(word) for word in text]
  return stem_text
 
# # applying function for stemming
  data['Text']=data['Text'].apply(lambda x: stemming(x))
  data['Text']=data['Text'].apply(lambda x: ''.join(x))