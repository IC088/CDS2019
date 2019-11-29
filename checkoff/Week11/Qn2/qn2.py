from numpy import array
from pickle import dump
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout

filename = 'C:/Users/chris/Documents/GitHub/CDS2019/checkoff/Week11/Qn2/shakespeare.txt'

def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    # text = [next(file) for x in range(4000)]
    text = file.read()
    # close the file
    file.close()
    return text


raw_text = load_doc(filename).lower()
tokens = raw_text.split()
raw_text = ' '.join(tokens[:8000])

print(raw_text)

# print(tokens)

#organize into sequences of characters
length = 50
sequences = list()
for i in range(length, len(raw_text)):
    # select sequence of tokens
    seq = raw_text[i-length:i+1]
    # store
    sequences.append(seq)
    print('Total Sequences: %d' % len(sequences))#Total Sequences: 597# save tokens to file, one dialog per line

def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()    
# save sequences to file
out_filename = 'char_sequences.txt'
save_doc(sequences, out_filename)

in_filename = 'char_sequences.txt'
raw_text = load_doc(in_filename)
lines = raw_text.split('\n')
chars = sorted(list(set(raw_text)))
mapping = dict((c, i) for i, c in enumerate(chars))

sequences = list()
for line in lines:
    #integer encode line
    encoded_seq = [mapping[char] for char in line]
    #store
    sequences.append(encoded_seq)#vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)


sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = array(sequences)
y = to_categorical(y, num_classes=vocab_size)

print(X.shape)

def define_model(X):
    model = Sequential()
    model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.8))
    model.add(Dense(vocab_size, activation='softmax'))
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize defined model
    model.summary()
    return model


# define model
model = define_model(X)


# fit model
model.fit(X, y, epochs=10, verbose=2, batch_size=32, validation_split=0.2)

# save the model to file
model.save('model.h5')
# save the mapping
dump(mapping, open('mapping.pkl', 'wb'))



from pickle import load
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences# load the model
model = load_model('model.h5')
# load the mapping
mapping = load(open('mapping.pkl', 'rb'))


#Input for the functions 
# model, mapping, the sequence length =50, the intial text, number of characters generated after this.

def generate_seq(model, mapping, seq_length, seed_text, n_chars):
    in_text = seed_text
    len(in_text)
    # generate a fixed number of characters
    for _ in range(n_chars):
        # encode the characters as integers
        encoded = [mapping[char] for char in in_text]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # one hot encode
        encoded = to_categorical(encoded, num_classes=len(mapping))
        encoded.shape
        #encoded = encoded.reshape( 1,encoded.shape[0], encoded.shape[1])
        # predict character
        yhat = model.predict_classes(encoded, verbose=0)
        # reverse map integer to character
        out_char = ''
        for char, index in mapping.items():
            if index == yhat:
                out_char = char
                break
            # append to input
        in_text += out_char
    return in_text

print(generate_seq(model, mapping, 50, "out of grief and impatience. answer'd neglectingly", 15)) 
#If no visible chars are generated it's ' '