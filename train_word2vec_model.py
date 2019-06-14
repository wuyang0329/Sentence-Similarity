#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import logging
import os
import sys
import multiprocessing
 
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence,PathLineSentences
 
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
 
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
 
    # check and process input arguments

    inp = './data/train_summary_jieba.txt'#语料文件地址
    outp1 = './model/finance/word2vec_summaries_v3.model'#模型文件存储地址
    outp2 = './model/finance/word2vec_summaries_v3.vector'#向量文件存储地址
 
    model = Word2Vec(LineSentence(inp), size=200, window=5, min_count=5,iter=30,sg=0,
                     workers=multiprocessing.cpu_count())

    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)
