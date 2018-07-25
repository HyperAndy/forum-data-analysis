# coding:utf-8  
"""
Author: WangZijian
Concat: wangzijian@autohome.com.cn
Time: 2018/7/24
Describle:  ltp处理分词，词性标注，NER，语义分析
"""

import os
import sys
import re
import codecs
import pandas as pd
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import NamedEntityRecognizer
from pyltp import Parser

LTP_DATA_DIR = './ltp_data_v3.4.0'  # ltp模型目录的路径
dic_path = './dict'  # 字典文件路径
dic_list = [os.path.join(dic_path, file) for file in os.listdir(dic_path)]
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # ner模型
pas_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型，parser

stop_words_list = []
p = codecs.open('./dict/stopwords_all.txt', 'r', 'utf-8')
for i in p:
    stop_words_list.append(i.strip('\r\n'))
p.close()

# 词性列表
# allow_speech_tags = ['an', 'i', 'j', 'l', 'n', 'nr', 'nrfg', 'ns', 'nt', 'nz', 't', 'v', 'vd', 'vn', 'eng', 'domain', 'brand', 'series', 'manu', 'car']
allow_speech_tags = ['n', 'nr', 'nrfg', 'nt', 'nz', 'eng', 'domain', 'brand', 'series', 'manu', 'car']

class preprosessing(object):
    def __init__(self, stop_words=stop_words_list, tags=allow_speech_tags):
        self.LTP_path = LTP_DATA_DIR
        self.cws_model_path = cws_model_path
        self.pos_model_path = pos_model_path
        self.ner_model_path = ner_model_path
        self.pas_model_path = pas_model_path
        self.dic_list = dic_list
        self.stop_words = set(stop_words)
        self.tags_filter = tags

    def segment(self, texts, postag=True, recognize=True, parse=True):
        # 初始化实例
        segmentor = Segmentor()
        segmentor.load_with_lexicon(cws_model_path, self.dic_list)  # 加载模型，参数lexicon是自定义词典的文件路径

        postagger = Postagger()
        postagger.load(self.pos_model_path)

        recognizer = NamedEntityRecognizer()
        recognizer.load(ner_model_path)

        parser = Parser()
        parser.load(pas_model_path)

        for text in texts:
            text = text.lower()

            word_list = segmentor.segment(text)
            word_list = [word for word in word_list if len(word) > 1]
            word_list = [word for word in word_list if re.match("[\u0041-\u005a\u4e00-\u9fa5]+", word) != None]  # .decode('utf8') 保留中英文
            word_list = [word.strip() for word in word_list if word.strip() not in self.stop_words]  # 去除停用词

            # 词性标注
            if postag:
                posttags = postagger.postag(word_list)
                postags = list(posttags)

            # NER识别
            if recognize:
                netags = recognizer.recognize(word_list, postags)

            # 句法分析
            if parse:
                arcs = parser.parse(word_list, postags)
                rely_id = [arc.head for arc in arcs]  # 提取依存父节点id
                relation = [arc.relation for arc in arcs]  # 提取依存关系
                heads = ['Root' if id == 0 else word_list[id - 1] for id in rely_id]  # 匹配依存父节点词语

        segmentor.release()
        postagger.release()
        recognizer.release()
        parser.release()
        return word_list, postags, netags, relation, heads

