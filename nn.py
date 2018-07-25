import collections
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
import pandas as pd

a = pd.read_csv('./data/topic_xinchetongji_all_seg8.txt', header=None, sep='\t', encoding='utf-8')
b = pd.read_csv('./data/topic_xinchetongji_all_seg10.txt', header=None, sep='\t', encoding='utf-8')

def tokenize(x):
    x_tok = Tokenizer(char_level=False)
    x_tok.fit_on_texts(x)
    return x_tok.texts_to_sequences(x), x_tok
    
def tokenize(x):
    x_tok = Tokenizer(char_level=False)
    x_tok.fit_on_texts(x)
    return x_tok.texts_to_sequences(x), x_tok
    
def preprocessing(x, y):
    preprocess_x, x_tok = tokenize(x)
    preprocess_y, y_tok = tokenize(y)
    preprocess_x = padding(preprocess_x)
    preprocess_y = padding(preprocess_y)
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)
    return preprocess_x, preprocess_y, x_tok, y_tok
    
def logits_to_text(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
#     index_to_words[0] = '<PAD>'
    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])
    
d = pd.merge(b, a, on=0, how='left')
e = [x.strip().split(',') for x in d.loc[:5000, '1_x']]
f = [x.strip().split(',') for x in d.loc[:5000, '1_y']]
pre_in, pre_out, in_tok, out_tok = preprocessing(f, e)
max_pre_in = pre_in.shape[1]
max_pre_out = pre_out.shape[1]
in_len = len(in_tok.word_index)
out_len = len(out_tok.word_index)
from keras.models import Sequential
def embed_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    learning_rate = 1e-5
    rnn = GRU(64, return_sequences=True, activation="tanh")
#     rnn = GRU(16, return_sequences=True, activation="relu")

    embedding = Embedding(french_vocab_size, 20, input_length=input_shape[1]) 
    logits = TimeDistributed(Dense(french_vocab_size, activation="softmax"))

    model = Sequential()
    #em can only be used in first layer --> Keras Documentation
#     model.add(embedding)
#     model.add(rnn)
#     model.add(logits)
    model.add(Embedding(input_dim=english_vocab_size,output_dim=128,input_length=input_shape[1]))
    model.add(Bidirectional(GRU(256,return_sequences=False)))
    model.add(RepeatVector(output_sequence_length))
    model.add(Bidirectional(GRU(256,return_sequences=True)))
    model.add(TimeDistributed(Dense(french_vocab_size,activation='softmax')))
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])

    return model

tmp_x = padding(pre_in, max_pre_out)
tmp_x = tmp_x.reshape((-1, pre_out.shape[-2]))
embeded_model = embed_model(tmp_x.shape,
                            max_pre_out,
                            in_len+1,
                            out_len+1)
embeded_model.fit(tmp_x, pre_out, batch_size=500, epochs=5, validation_split=0.2)
print(logits_to_text(embeded_model.predict(tmp_x[:3])[0], out_tok))
