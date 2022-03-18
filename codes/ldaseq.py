import numpy as np
import pandas as pd
import gensim
from gensim.models import ldaseqmodel


if __name__ == "__main__":
    print("Read Corpus")
    corpus = gensim.corpora.MmCorpus('../data/covid.mm')
    print("Read Dictionary")
    dictionary = gensim.corpora.dictionary.Dictionary.load('../models/lda_dict')
    docs_per_time_slice = [13758, 9460, 13417, 12683, 12219, 9535, 7805, 8488, 7064, 5518, 5625, 8656, 13615]
    print("Start!")
    ldaseq = ldaseqmodel.LdaSeqModel(corpus=corpus, id2word=dictionary, time_slice=docs_per_time_slice, num_topics=4)
    ldaseq.save('../models/lda_seq')
