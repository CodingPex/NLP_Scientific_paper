import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')
from sklearn.linear_model import SGDClassifier


# preprocess the text data, including keep only letters, lowercase, removed stopwords and stemm or lemmatise words
# reference: https://github.com/lisanka93/text_analysis_python_101/blob/master/Dummy%20movie%20dataset.ipynb
stop_words = set(stopwords.words('english'))

def preprocess(raw_text):
    
    #regular expression keeping only letters 
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)

    # convert to lower case and split into words -> convert string into list ( 'hello world' -> ['hello', 'world'])
    words = letters_only_text.lower().split()

    cleaned_words = []
    lemmatizer = PorterStemmer() #plug in here any other stemmer or lemmatiser you want to try out
    
    # remove stopwords
    for word in words:
        if word not in stop_words:
            cleaned_words.append(word)
    
    # stemm or lemmatise words
    stemmed_words = []
    for word in cleaned_words:
        word = lemmatizer.stem(word)   #dont forget to change stem to lemmatize if you are using a lemmatizer
        stemmed_words.append(word)
    
    # converting list back to string
    return " ".join(stemmed_words)


# feature engineering
# use only title, abstract and year
def feature_engin(df):
    df['prep_title'] = df['title'].apply(preprocess)
    df['prep_abstract'] = df['abstract'].apply(preprocess)
    df['prep_whole'] = df['prep_title']  + " " + df['prep_abstract'] + " " + df['year'].apply(str)


# write prediction into json file
def save_pred(prediction,df_test):
    ids=df_test["paperId"].tolist()
    res=[]
    
    for i in range(0,len(ids)):
        dic = dict()
        dic["paperId"]=ids[i]
        dic["authorId"]=str(prediction[i])
        res.append(dic)
        
    with open('predicted.json', 'w') as js:
        json.dump(res, js)

    js.close()


# load the data
data = json.load(open('train.json'))
df_train = pd.DataFrame(data)

data_test = json.load(open('test.json'))
df_test = pd.DataFrame(data_test)


# feature engineering both train and test dataset
feature_engin(df_train)
feature_engin(df_test)


#use train data to fit and transform to sparse matrix, and then just transform test data
vectorizer = TfidfVectorizer(ngram_range=(1,2),min_df=2, sublinear_tf=True)

X = vectorizer.fit_transform(df_train["prep_whole"])
X_test = vectorizer.transform(df_test["prep_whole"])
y = df_train['authorId']


# split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1234)


# feed SGDClassifier
sgd = SGDClassifier(loss='epsilon_insensitive',max_iter=800, n_iter_no_change=15)
sgd.fit(X_train, y_train)

# evaluate the model
score_train = sgd.score(X_train, y_train)
print('Train accuracy:', score_train)
     
score = sgd.score(X_val, y_val)
print('Test accuracy:', score)


# predict test data
prediction = sgd.predict(X_test)
save_pred(prediction,df_test)