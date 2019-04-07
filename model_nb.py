#Import Libraries
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import re
import matplotlib.pyplot as plt

np.random.seed(500)
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

#Read CSV File
Corpus = pd.read_csv("Travel Blogs.csv")

#Drop Extra Columns
Corpus.drop('Video id', axis=1, inplace=True)
Corpus.drop('Title', axis=1, inplace=True)

# Remove blank rows if any.
Corpus.dropna(inplace=True)
Corpus['Description'].dropna(inplace=True)

#Remove Website Links
Corpus['Description'] = Corpus['Description'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
Corpus['Description'] = Corpus['Description'].str.replace('http\S+|www.\S+', '', case=False)
Corpus['Description'] = Corpus['Description'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)
Corpus['Description'] = Corpus['Description'].str.replace('http\S+|www.\S+|S+.com', '', case=False)

# Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
Corpus['Description'] = [entry.lower() for entry in Corpus['Description']]

# Tokenization : In this each entry in the corpus will be broken into set of words
Corpus['Description']= [word_tokenize(entry) for entry in Corpus['Description']]

# Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.

# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(Corpus['Description']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    Corpus.loc[index,'text_final'] = str(Final_words)


print(Corpus.head())

#Train Test Split
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['Category'],test_size=0.3)

#Encoding and Transforming Data
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y.astype(str))
Test_Y = Encoder.fit_transform(Test_Y)

#Vectorizering Data
Tfidf_vect = TfidfVectorizer()
Tfidf_vect.fit(Corpus['text_final'].astype(str))
Train_X_Tfidf = Tfidf_vect.transform(Train_X.astype(str))
Test_X_Tfidf = Tfidf_vect.transform(Test_X.astype(str))

# Classifier - Algorithm - Naive Bayes

# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)

# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)

# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)

#Precision , Recall, F1 Score
print("Precision: %f "%precision_score(Test_Y, predictions_NB))
print("Recall: %f "%recall_score(Test_Y, predictions_NB))
print("F1: %f"% f1_score(Test_Y, predictions_NB))

y_scores = Naive.predict_proba(Train_X_Tfidf)

prec, rec, tre = precision_recall_curve(Train_Y, y_scores[:,1], )

#Plot Precison, Recall
def plot_prec_recall_vs_tresh(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label='precision')
    plt.plot(thresholds, recalls[:-1], 'g--', label = 'recall')
    plt.xlabel('Threshold')
    plt.title('Naive Bayes')
    plt.legend(loc='upper left')
    plt.ylim([0,1])

plot_prec_recall_vs_tresh(prec, rec, tre)
plt.show()
