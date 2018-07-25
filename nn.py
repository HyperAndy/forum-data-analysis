# coding:utf-8  
"""
Author: WangZijian
Concat: wangzijian@autohome.com.cn
Time: 2018/7/20
Describle: seq2seq nn
"""
import collections
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import GRU, Input, TimeDistributed, Activation, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
import pandas as pd

a = pd.read_csv('./data/topic_xinchetongji_all_seg8.txt', header=None, sep='\t', encoding='utf-8')

print(a.head())


# import pandas as pd
# import tensorflow as tf
# import numpy as np
# import os
# import sys
# import keras
# from keras.layers import *
# import seq2seq
# from seq2seq.models import SimpleSeq2Seq

# df = pd.read_csv('./data/forum_topic0718_20170601.txt', header=None, sep='\t', encoding='utf-8')  #评论中存在'',读取df时会将其识别为字符串
df_seg = pd.read_csv('./segment_result/topic_xinchetongji_all_seg7.txt', header=None, sep='\t', encoding='utf-8')

# model = SimpleSeq2Seq(input_dim=5, hidden_dim=10, output_length=8, output_dim=8, depth=3)
# model.compile(loss='mse', optimizer='rmsprop')
#
# from keras.models import Sequential
# from keras.layers.recurrent import LSTM
# from keras.layers.wrappers import TimeDistributed
# from keras.layers.core import Dense, RepeatVector
#
#
# def build_model(input_size, seq_len, hidden_size):
#     """建立一个 sequence to sequence 模型"""
#     model = Sequential()
#     model.add(GRU(input_dim=input_size, output_dim=hidden_size, return_sequences=False))
#     model.add(Dense(hidden_size, activation="relu"))
#     model.add(RepeatVector(seq_len))
#     model.add(GRU(hidden_size, return_sequences=True))
#     model.add(TimeDistributed(Dense(output_dim=input_size, activation="linear")))
#     model.compile(loss="mse", optimizer='adam')
#
#     return model

