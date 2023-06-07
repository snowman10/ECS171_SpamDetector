# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer("english")



# porter_stemmer = PorterStemmer()
# # #defining a function for stemming
# def stemming(text):
#   stem_text = [porter_stemmer.stem(word) for word in text]
#   return stem_text
 
# # # applying function for stemming
#   data['Text']=data['Text'].apply(lambda x: stemming(x))
#   data['Text']=data['Text'].apply(lambda x: ''.join(x))


import pandas as pd

dict = {
  "Text": "Wowie this is cool"
}
df = pd.DataFrame([dict])
print(df)



# dictStr='{"firstName": "John", "lastName": "Doe", "email": "john.doe@example.com", "age": 32}'
# print("The dictionary string is:")
# print(dictStr)
# myDict=eval(dictStr)
# df=pd.DataFrame([myDict])
# print("The output dataframe is:")
# print(df)



def wrapString(string, n=10):
  words = string.split()
  partial = [(' '.join(words[i:i+n])) for i in range(0, len(words), n)]
  partial = [a.center(150) for a in partial]
  
  return '\n'.join(partial)



x = ''
if x == '':
  print(f"|{x.center(150)}|")
else:
  print(f"|{wrapString(x)}|")


print(" a".split())