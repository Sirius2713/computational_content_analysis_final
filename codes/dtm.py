#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import gensim
import re


# In[2]:
if __name__ == "__main__":

    print("Read data!")
    data = pd.read_csv('../data/csv_data/dtm_data.csv')
    data.head()


    # In[3]:


    data['year_month'] = data['date'].apply(lambda x: x[: 5])
    data.sort_values(by=['year_month'], inplace=True)
    docs_per_time_slice = data['year_month'].value_counts()


    # In[5]:


    data['body_tokens_reduced'] = data['body_tokens_reduced'].apply(lambda x: [re.sub('[^0-9a-zA-Z]+', '', k) for k in x.split(',')])


    # In[8]:


    dictionary = gensim.corpora.Dictionary(data['body_tokens_reduced'])


    # In[9]:


    corpus = [dictionary.doc2bow(text) for text in data['body_tokens_reduced']]


    # In[10]:
    print("Start modeling!")

    ldaseq = gensim.models.ldaseqmodel.LdaSeqModel(corpus=corpus, id2word=dictionary, time_slice=docs_per_time_slice, num_topics=4)
    ldaseq.save('../models/ldaseq')
