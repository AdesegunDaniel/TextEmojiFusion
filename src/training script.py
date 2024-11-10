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
from src.exception import CustomException
from src.my_logging import logging
from src.utils import Embedding, single_embed, classify
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import euclidean_distances

logging.info('getting the training dataset')
try:
    with open('src\Sentiment Group.json', 'r') as file:
        groups=json.load(file)
except Exception as e:
    raise CustomException(e, sys)


logging.info('getting the sentiment groups')
try:
    positive=pd.DataFrame(groups['POSITIVE'])
    negative=pd.DataFrame(groups['NEGATIVE'])
    neutral=pd.DataFrame(groups['NEUTRAL'])
    object_=pd.DataFrame(groups['OBJECT'])
except Exception as e:
    raise CustomException(e, sys)


          

logging.info('embedding the description of the emojis')
try:
    pos_arr=single_embed(list(positive['name'])).numpy()
    neg_arr=single_embed(list(negative['name'])).numpy()
    neu_arr=single_embed(list(neutral['name'])).numpy()
    logging.info('encoding name to emoji map')
    obj_dic={name:emoji for name, emoji in zip(list(object_['name']),list(object_['emoji']))}
    pos_dic={idx:emoji for idx, emoji in enumerate(list(positive['emoji']))}
    neg_dic={idx:emoji for idx, emoji in enumerate(list(negative['emoji']))}
    neu_dic={idx:emoji for idx, emoji in enumerate(list(neutral['emoji']))}
    logging.info('mapping the decode')
    decode={0:'Negative', 1:'Positive', 2:'Neutral'}
except Exception as e:
    raise CustomException(e, sys)


def extract(sent_arr, group):
    if group==0:
        emo= np.argsort(euclidean_distances(neg_arr.reshape(neg_arr.shape[0],-1), 
                                          sent_arr.reshape(1,-1)), axis=0).reshape(1,-1)[0][:20]
        emoji=neg_dic[np.random.choice(emo)]
    elif group==1:
        emo= np.argsort(euclidean_distances(pos_arr.reshape(pos_arr.shape[0], -1),
                                          sent_arr.reshape(1,-1)), axis=0).reshape(1,-1)[0][:20]
        emoji=pos_dic[np.random.choice(emo)]
        
    else:
        emo=np.argsort(euclidean_distances(neu_arr.reshape(neu_arr.shape[0], -1),
                                        sent_arr.reshape(1,-1)), axis=0).reshape(1,-1)[0][:20]
        emoji=neu_dic[np.random.choice(emo)]        
    
    return emoji


def predict(sentences):
    prediction=[]
    for sentence in sentences:
        sent=sentence.split()
        work=[]
        while len(sent)>6:
            wrk=sent[:6]
            work.append(' '.join(wrk))
            sent=sent[6:]
        if sent!=0:
            work.append(' '.join(sent))
    
        embed=Embedding(work).numpy()
        if embed.shape[1]!=6:
            p=6-embed.shape[1]
            embed=np.pad(embed,((0,0),(0,p),(0,0)))
            
        grp=[classify(x) for x in work]
        grp_emo=[extract(arr,group) for arr,group in zip(embed,grp)]
        result=' '.join([x+' '+re.sub(r'\u200d','',y) for x,y in zip(work,grp_emo)]).split()
        
        
        for i,x in enumerate(result):
            if x in obj_dic.keys():
                result[i]=x+obj_dic[x]
            else:
                pass
        
        prediction.append([' '.join(result), decode[grp[0]]])
    return  prediction, 



if __name__=="__main__":
    spicy=predict(["is a powerful and intuitive tool designed to enhance your text with the perfect emojis. Whether you're looking to add a touch of emotion, humor, or emphasis, TextEmojiFusion is not just a tool; it's your creative partner in the digital realm, transforming mundane messages into vibrant conversations. Imagine a platform that understands the nuances of your emotions and the subtleties of your humor, then curates a selection of emojis that perfectly align with your intent. TextEmojiFusion does just that, bridging the gap between plain text and the rich expressiveness of emojis. With its intuitive design, it's incredibly user-friendly, inviting you to elevate your communication effortlessly. Whether you're conveying heartfelt sentiments, injecting a dose of wit, or emphasizing key points, TextEmojiFusion integrates emojis with such finesse that your messages will capture hearts and imaginations alike. It's not merely about enhancing text; it's about enriching connections and sharing experiences that words alone can't fully express. With TextEmojiFusion, every message becomes a canvas, and every conversation is an opportunity to create something memorable. So why settle for ordinary when you can communicate with color, emotion, and life? Embrace the power of TextEmojiFusion and watch your digital interactions blossom into engaging, expressive, and extraordinary exchanges."])
    print(spicy)