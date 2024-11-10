import re
import json
import nltk
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from afinn import Afinn
from transformers import pipeline
from sklearn.metrics.pairwise import euclidean_distances


def Embedding(sentence, batch_size=5000, model_path='https://tfhub.dev/google/elmo/3', key='elmo'):
    tf_sent=tf.stack([tf.constant(x) for x in sentence], axis=0)
    embed= hub.KerasLayer(model_path, output_key=key, trainable=False)
    result=[]
    for start in range(0, len(tf_sent), batch_size):
        end=start+batch_size  
        embed_sent=embed(tf_sent[start:end])
        result.extend(embed_sent)
    return tf.stack(result) 


def classify(sentence):
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    afn=Afinn()
    score=afn.score(sentence)
    sentiment= 1 if score > 0 else 1 if score < 0 else 2
    if sentiment== 2:
        return sentiment
    if sentiment == 1:
        sentiment_analyzer = pipeline("sentiment-analysis", model=model_name, device=0)
        label=sentiment_analyzer(sentence)[0]['label']
        sentiment= 1 if label =='POSITIVE' else 0
    return sentiment

def single_embed(sentence):
    new=[]
    for sent in sentence:
        n_sent=sent.split()
        while len(n_sent)>6:
            sen=n_sent[:6]
            new.append(' '.join(sen))
            n_sent=sent[6:]
        if len(n_sent)!=0:
            new.append(' '.join(n_sent))
    embed=Embedding(new)
    result=[]
    for em in embed:
        if em.shape[0]!=6:
            p=6-em.shape[0]
            pad_em=np.pad(em,((0,p),(0,0)))
            result.append(tf.constant(pad_em))
        else:
            result.append(em)
            
    return tf.stack(result)
