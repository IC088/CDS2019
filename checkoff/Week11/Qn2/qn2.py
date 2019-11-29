from numpy import array
from pickle import dump
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

filename = 'C:/Users/Asus/Documents/GitHub/CDS2019/checkoff/Week11/Qn2/shakespeare.txt'

def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    # text = [next(file) for x in range(4000)]
    text = file.read()
    # close the file
    file.close()
    return text


raw_text = load_doc(filename)
tokens = raw_text.split().lower()
raw_text = ' '.join(tokens[:4000])

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