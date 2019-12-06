### Character-based Generation
import os
import numpy as np
import keras
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import reuters, imdb
from keras.models import Sequential
from keras.layers import LSTM, SimpleRNN, GRU, Dense, Dropout, Activation, Embedding, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import pandas as pd

EMBEDDING_DIM = 50
# load in training/test set
data = pd.read_csv('tweets.160k.random.csv', encoding='utf-8')
data.head()

data['label'].value_counts()

vocab_size = 20000
tokenizer = Tokenizer(num_words= vocab_size, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ', char_level=True)
tokenizer.fit_on_texts(data['text'])
sequences = tokenizer.texts_to_sequences(data['text'])
word_index = tokenizer.word_index
tweets = sequence.pad_sequences(sequences, padding='post', maxlen=50)


labels = data['label']
labels = labels.replace(4,1) # replace label '4' with '1' to facilitate one-hot encoding
x_train, x_test, y_train, y_test = train_test_split(tweets, labels, test_size=0.2)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

y_train = keras.utils.to_categorical(y_train) # 2 classes
y_test = keras.utils.to_categorical(y_test)


# define documents

# string = 'abcdefghijklmnoprstuvwxyz0123456789'
# docs = [i for i in string]
# labels = np.array([i for i in range(len(string))])
# print(labels)
# # define class labels
# # labels = array([1,1,1,1,1,0,0,0,0,0])
# # # integer encode the documents
# vocab_size = 50
# encoded_docs = [one_hot(d, vocab_size) for d in docs]
# print(encoded_docs)
# # pad documents to a max length of 4 words
# max_length = 1
# padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# print(padded_docs)

# embeddings_index = {}







# embeddings_index = {}
# GLOVE_DIR = ".\\"
# f = open(os.path.join(GLOVE_DIR, 'glove.6B.50d.txt'), encoding='utf8')
# for line in f:
#     values = line.split()
#     word = values[0]
#     print(word)
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# f.close()




# print('Found %s word vectors.' % len(embeddings_index))

# embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
# for word, i in word_index.items():
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector

model = Sequential()
model.add(Embedding(len(word_index)+1, EMBEDDING_DIM, trainable=True))
model.add(Conv1D(100, 2, activation='relu'))
model.add(Conv1D(100, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(160, 4, activation='relu'))
model.add(Conv1D(160, 5, activation='relu'))
model.add(LSTM(128))

# model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))

model.add(Dense(2))
model.add(Activation('softmax'))

print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=128, epochs=3, verbose=1, validation_split=0.2)

score = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

