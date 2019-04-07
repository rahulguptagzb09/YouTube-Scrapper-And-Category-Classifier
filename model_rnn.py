#Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import nltk
import re

from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text
from keras import utils
from keras.models import Model
from keras.layers import LSTM, Input, Embedding

from keras import backend as K

np.random.seed(500)
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

#Read CSV File
Corpus = pd.read_csv("Final1.csv")

#Drop Extra Columns
Corpus.drop('Video id', axis=1, inplace=True)
Corpus.drop('Title', axis=1, inplace=True)

# Step - a : Remove blank rows if any.
Corpus.dropna(inplace=True)
Corpus['Description'].dropna(inplace=True)

#Remove Website Links
Corpus['Description'] = Corpus['Description'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
Corpus['Description'] = Corpus['Description'].str.replace('http\S+|www.\S+', '', case=False)
Corpus['Description'] = Corpus['Description'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)
Corpus['Description'] = Corpus['Description'].str.replace('http\S+|www.\S+|S+.com', '', case=False)

# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
Corpus['Description'] = [entry.lower() for entry in Corpus['Description']]

# RNN

#Train Test Split
train_size = int(len(Corpus) * .7)
train_posts = Corpus['Description'][:train_size]
train_tags = Corpus['Category'][:train_size]

test_posts = Corpus['Description'][train_size:]
test_tags = Corpus['Category'][train_size:]

#Vectorizering Data
max_words = 1000
max_len = 150
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(train_posts) # only fit on train

x_train = tokenize.texts_to_matrix(train_posts)
x_test = tokenize.texts_to_matrix(test_posts.astype(str))

#Encoding and Transforming Data
encoder = LabelEncoder()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags.astype(str))
y_test = encoder.transform(test_tags.astype(str))

num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

batch_size = 32
epochs = 1

# Build the model

inputs = Input(name='inputs',shape=[max_words, ])
layer = Embedding(max_words,50,input_length=max_len)(inputs)
layer = LSTM(64)(layer)
layer = Dense(256,name='FC1')(layer)
layer = Activation('relu')(layer)
layer = Dropout(0.5)(layer)
layer = Dense(6,name='out_layer')(layer)
layer = Activation('sigmoid')(layer)
model = Model(inputs=inputs,outputs=layer)

model.summary()

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', f1_m,precision_m, recall_m])

# fit the training dataset on the Shallow NN                    
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)



#Precision , Recall, F1 Score, Accuracy
loss, score, f1_score, precision, recall = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f} \n  \n  F1 Score: {:0.3f} \n  Precision: {:0.3f} Recall: {:0.3f}'.format(loss, score, f1_score, precision, recall))

