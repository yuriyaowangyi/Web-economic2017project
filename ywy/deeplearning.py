from sklearn.linear_model import LogisticRegression
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import string
import math
import csv
import random
import time
import pandas as pd
from sklearn import metrics
from keras.layers import Dense, Dropout, activations
from keras.models import Sequential
import gc


def train_deeplearning(X,y):
    dimx = X.shape[1]
    model = Sequential()
    model.add(Dense(128,input_dim= dimx,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

    model.fit(X, y)

    return model




def params_estimator(path):
    df=pd.read_csv(path)
    avgCTR=(df.click.sum()/df.shape[0])*100
    base_bid=df.payprice.mean()
    return avgCTR,base_bid

def load_data(path,training=True,testing=False):
    processed_data=list()
    processed_labels=list()
    features=['weekday', 'hour', 'region','city', 'adexchange', 'slotwidth', 'slotheight','slotvisibility', 'slotformat', 'slotprice','advertiser']
    other_features=['useragent','usertag']
    if training==False:
        df=pd.read_csv(path, skipinitialspace=True,usecols=['payprice'])
        payprices=df['payprice']
        payprices=list(payprices.values)
    if testing==False:
        df = pd.read_csv(path, skipinitialspace=True,usecols=['click'])
        labels=df['click']
        labels=list(labels.values)
    converter_dict={'weekday':str,'hour':str,'region':str,'city':str,'adexchange':str,'slotwidth':str,'slotheight':str,'slotvisibility':str,'slotformat':str,'advertiser':str}
    datas = pd.read_csv(path, skipinitialspace=True,usecols=features,converters=converter_dict)
    print('Start prepocessing\n')
    datas['slotprice']=datas['slotprice']/std_of_slotprice
    print('price standardization\n')
    df = pd.read_csv(path, skipinitialspace=True,usecols=other_features)
    print('Start deep prepocessing\n')
    for i in list(range(0,len(df['usertag']))):
        instance=dict()
        instance.update(dict(datas.iloc[i]))
        op_sys, browser =df.iloc[i]['useragent'].split('_')
        instance.update({op_sys: True, browser: True})
        #后期需要修改
        usertags=df.iloc[i]['usertag'].split(',')
        temp_dict = {}
        for tag in usertags:
            temp_dict["tag " + tag] = True
        instance.update(temp_dict)
        processed_data.append(instance)
        if testing==False:
            processed_labels.append(int(labels[i]))
        if i%10000==0:
            print('Have processed '+str(i)+' data \n')
    del datas
    if testing==True:
        df = pd.read_csv(path, skipinitialspace=True,usecols=['bidid'])
        ind=list(df['bidid'].values)
        return processed_data,ind
    if training==False:
        return processed_data,processed_labels,payprices
    if training==True:
        return processed_data,processed_labels



def train(training_data, labels):
    label_encoder = LabelEncoder()
    vectorizer = DictVectorizer()
    train_event_x = vectorizer.fit_transform(training_data)
    train_event_y = label_encoder.fit_transform(labels)
    # Create and train the model.
    pctr_estimator = train_deeplearning(train_event_x.toarray(),train_event_y)
    model = (pctr_estimator, label_encoder, vectorizer)
    print('Training done')
    return model


def predict_event_labels(instance, model): # models:dict
    pctr_estimator = model[0]
    # Transform event:
    vectorizer = model[2]
    event = [instance]
    event_x = vectorizer.transform(event)
    #event_y = label_encoder.inverse_transform(lr.predict(event_x))
    event_y = pctr_estimator.predict_proba(event_x.toarray())
    return event_y

def predict_events_PCTR(model,processed_data):
    predict_label=list()
    PCTR=list()
    for i in list(range(0,len(processed_data))):
        # parsing event:
        instance = processed_data[i]
        # Predicting CTR:
        PCTR.append(predict_event_labels(instance, model))
        predict_label.append(int((PCTR[-1]>0.5)))
    return PCTR,predict_label

def model_evaluator(model,processed_staff,start_budget,random_iter=5):
    clicks=0
    impressions=0
    budget=start_budget
    CTR=0
    CPC=0
    processed_data=processed_staff[0]
    processed_labels=processed_staff[1]
    payprices=processed_staff[2]
    PCTR,predict_label=predict_events_PCTR(model,processed_data)
    total_CTR_scores=0
    total_CPC_scores=0
    for i in range(random_iter):
        y=random.sample(list(range(0,len(PCTR))),200000)
        data=[processed_data[x] for x in y]
        labels=[processed_labels[x] for x in y]
        payprice=[payprices[x] for x in y]
        sample_PCTR=[PCTR[x] for x in y]
        for i in list(range(0,len(data))):
            pCTR = sample_PCTR[i]
            #你的定价模型写在这里！！！
            current_bid = base_bid * pCTR / avgCTR
            # Check if we still have budget:
            if budget > current_bid:
                # Get the market price:
                payprice =payprices[i]
                # Check if we win the bid:
                if current_bid > payprice:
                    impressions += 1
                    budget -= payprice
                    # Check if the person clicks:
                    if labels[i] == 1:
                        clicks += 1
            else:
                break
        temp_CTR_score=clicks/impressions
        temp_CPC_score=((start_budget-budget)/clicks)/1000
        total_CTR_scores=total_CTR_scores+temp_CTR_score
        total_CPC_scores=total_CPC_scores+temp_CPC_score
    CTR=total_CTR_scores/random_iter
    CPC=total_CPC_scores/random_iter
    return CTR,CPC


def predict_event_bidprice(PCTR,processed_staff,start_budget):
    bid_price=list()
    processed_data=processed_staff
    for i in list(range(0,len(processed_data))):
        pCTR = PCTR[i]
        current_bid = base_bid * pCTR / avgCTR
        # Check if we still have budget:
        bid_price.append(current_bid)
    return bid_price


st=time.time()
training_path = r"dataset/train.csv"
validation_path = r"dataset/validation.csv"
test_path=r"dataset/test.csv"
start_budget=6250000
slotprices = pd.read_csv(training_path, skipinitialspace=True,usecols=['slotprice'])
global std_of_slotprice
std_of_slotprice=int(slotprices['slotprice'].values.std())
# Extracting data:
training_events, labels = load_data(training_path)
avgCTR,base_bid=params_estimator(training_path)

# training model
model = train(training_events, labels)

processed_staff=load_data(validation_path,training=False)

CTR,CPC=model_evaluator(model,processed_staff,start_budget,random_iter=5)

test_events,ind=load_data(test_path,training=True,testing=True)
PCTR,test_predict_label=predict_events_PCTR(model,test_events)
predict_bid_price=predict_event_bidprice(PCTR,test_events,start_budget)
out_file = open("deepl.csv", "w", newline='')
writer = csv.writer(out_file)
writer.writerow(['id','bid_price'])
for i in range(len(predict_bid_price)):
    writer.writerow([ind[i],predict_bid_price[i]])
out_file.close()
print (time.time()-st)
