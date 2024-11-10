from flask import Flask, request, jsonify, render_template, session
import re 
import numpy as np
from afinn import Afinn
import tensorflow as tf
import tensorflow_hub as hub
from transformers import pipeline
from sklearn.metrics.pairwise import euclidean_distances
import pickle 
import os




app=Flask(__name__)
app.secret_key =os.urandom(24)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/test_page')
def test_page():
    return render_template('test_page.html')


with open('src\Standard Array.pkl', 'rb') as file:
    std_arr=pickle.load(file)
with open('src\Ref_dic.pkl', 'rb') as file:
    ref_dic=pickle.load(file)

pos_arr=std_arr['pos_arr']
neg_arr=std_arr['neg_arr']
neu_arr=std_arr['neu_arr']

obj_dic=ref_dic['obj_dic']
pos_dic=ref_dic['pos_dic']
neg_dic=ref_dic['neg_dic']
neu_dic=ref_dic['neu_dic']
decode=ref_dic['decode']

model_path='src\elmo-tensorflow1-v3'
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
elmo_model=hub.KerasLayer(model_path, output_key='elmo', trainable=False)
hug_model=pipeline("sentiment-analysis", model=model_name)

embed, work, grp= None, None, None


def Embedding(sentence, batch_size=500, model=elmo_model):
    tf_sent=tf.constant(sentence)
    embed=model(tf_sent)
    return embed



def classify(sentence, model=hug_model):   
    afn=Afinn()
    score=afn.score(sentence)
    sentiment= 1 if score > 0 else 1 if score < 0 else 2
    if sentiment== 2:
        return sentiment
    if sentiment == 1:
        label=model(sentence)[0]['label']
        sentiment= 1 if label =='POSITIVE' else 0
    return sentiment



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


@app.route('/predict', methods=['POST'])
def predict():
    global embed, work, grp
    users_input=request.form['user_text']
    sentences=[users_input]
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
        #session['embed']=embed.tolist()
        #session['grp']=grp 
        #session['work']=work
        grp_emo=[extract(arr,group) for arr,group in zip(embed,grp)]
        result=' '.join([x+' '+re.sub(r'\u200d','',y) for x,y in zip(work,grp_emo)]).split()
        
        
        for i,x in enumerate(result):
            if x in obj_dic.keys():
                result[i]=x+obj_dic[x]
            else:
                pass
        
        prediction.append([' '.join(result)])
    return  render_template('result.html', results=prediction[0][0])


@app.route('/change', methods=['POST'])
def change_emoji():
    global embed, work, grp
    #embed=np.array(session.get('embed'))
    #work=session.get('work')
    #grp=session.get('grp')
    prediction=[]
    grp_emo=[extract(arr,group) for arr,group in zip(embed,grp)]
    result=' '.join([x+' '+re.sub(r'\u200d','',y) for x,y in zip(work,grp_emo)]).split()
    for i,x in enumerate(result):
        if x in obj_dic.keys():
            result[i]=x+obj_dic[x]
        else:
                pass
        
        prediction.append([' '.join(result)])
    return  render_template('result.html', results=prediction[0][0])



if __name__=="__main__":
    app.run(host='0.0.0.0', debug=True)