#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 19:43:42 2017

@author: dupingyuan
"""

from pyfm import pylibfm
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import string
import math
import csv
import random
import time
import pandas as pd
from sklearn import metrics


def params_estimator(path):
    df=pd.read_csv(path)
    avgCTR=(df.click.sum()/df.shape[0])*100
    base_bid=df.payprice.mean()-df.payprice.std()
    #base_bid=optimal bid by ctr or cpc from model_evaluation
    return avgCTR,base_bid

def load_data(path,training=True,testing=False):
    processed_data=list()
    processed_labels=list()  
    #feature selection manually
    features=['weekday', 'hour', 'region','city', 'adexchange', 'slotwidth', 'slotheight','slotvisibility', 'slotformat', 'slotprice','advertiser']
    #special features to split them
    other_features=['useragent','usertag']
    
    #read validation.csv
    if training==False:
        df=pd.read_csv(path, skipinitialspace=True,usecols=['payprice'])
        payprices=df['payprice']
        payprices=list(payprices.values)  
    #read train.csv    
    if testing==False:
        df = pd.read_csv(path, skipinitialspace=True,usecols=['click'])
        labels=df['click']
        labels=list(labels.values)
        
    converter_dict={'weekday':str,'hour':str,'region':str,'city':str,'adexchange':str,'slotwidth':str,'slotheight':str,'slotvisibility':str,'slotformat':str,'advertiser':str}
    datas = pd.read_csv(path, skipinitialspace=True,usecols=features,converters=converter_dict)
    print('Process begin\n')
    #standard deviation payprice process to limit the range
    datas['slotprice']=datas['slotprice']/std_of_slotprice
    print('standard deviation payprice process finished\n')
    df = pd.read_csv(path, skipinitialspace=True,usecols=other_features)
    print('Deep process begin\n')
    
    for i in list(range(0,len(df['usertag']))):
        instance=dict()
        instance.update(dict(datas.iloc[i]))
        #Split of 'useragent' & 'usertag'
        op_sys, browser =df.iloc[i]['useragent'].split('_')
        instance.update({op_sys: True, browser: True})
        usertags=df.iloc[i]['usertag'].split(',')
        temp_dict = {}
        
        for tag in usertags:
            temp_dict["tag " + tag] = True
        instance.update(temp_dict)
        processed_data.append(instance)
        #train.csv process
        if testing==False:
            processed_labels.append(int(labels[i]))
        if i%10000==0:
            print('having processed'+str(i)+'data\n')
    #delete datas to save memory
    del datas
    
    if testing==True:
        df = pd.read_csv(path, skipinitialspace=True,usecols=['bidid'])
        ind=list(df['bidid'].values)
        return processed_data,ind
    #for validation data
    if training==False:
        return processed_data,processed_labels,payprices
    if training==True:
        return processed_data,processed_labels


def train(training_data, labels):
    #label encoder features to get prediction
    label_encoder = LabelEncoder()
    vectorizer = DictVectorizer()
    train_event_x = vectorizer.fit_transform(training_data)
    train_event_y = label_encoder.fit_transform(labels)
    #definition of balanced weights 
    global weights
    weights=len(labels)/(sum(labels)*2)
    
    # Create and train the model using Factorization Machine Algorithm
    pctr_estimator = pylibfm.FM()
    pctr_estimator.fit(train_event_x, train_event_y)
    model = (pctr_estimator, label_encoder, vectorizer)
    print('Training done')
    return model

def predict_event_labels(instance, model):
    #get a pctr of an item
    pctr_estimator = model[0]
    # Transform event:
    vectorizer = model[2]
    event = [instance]
    event_x = vectorizer.transform(event)
    event_y = pctr_estimator.predict(event_x)
    return event_y[0]

def predict_events_PCTR(model,processed_data):
    #get all pctrs
    predict_label=list()
    PCTR=list()
    for i in list(range(0,len(processed_data))):
        # parsing event:
        instance = processed_data[i]
        # Predicting CTR:
        PCTR.append(predict_event_labels(instance, model))
        predict_label.append(int((PCTR[-1]>0.5)))
    return PCTR,predict_label

def model_evaluator(model,processed_staff,start_budget,random_iter=10):
    #randomly selecting 200000 data from validation.csv and iterating 10 times
    #final CTR & CPC are the average data of 5 times
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
        
        #bidding process begin with budget limitation 
        for i in list(range(0,len(data))):
            pCTR = sample_PCTR[i]*weights
            if pCTR > 0.8:
                pCTR=0.80000000000000
            #print "pctr: "+str(pCTR)
            predict_label.append(int((pCTR>0.5)))
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

def val_assessment(validation_path,model):
    #the evaluation of bidding strategy of the whole validation set
    #return CTR,CPC,clicks,impression,money left after the auction
    val_events,ind=load_data(validation_path,training=True,testing=True)
    PCTR,val_predict_label=predict_events_PCTR(model,val_events)
    predict_bid_price=predict_event_bidprice(PCTR,val_events,start_budget)
    df_val=pd.read_csv(validation_path)
    budget=6250000
    impression=0
    clicks=0
    for i in list(range(0,len(df_val))):
        if budget<0:
            break
        if df_val.iloc[i].payprice<predict_bid_price[i]:
            #here budget is the left money after auction
            #it might be positive, it might be negative
            budget=budget-df_val.iloc[i].payprice
            impression=impression+1
            if df_val.iloc[i].click==1:
                clicks=clicks+1
    CTR=(clicks/impression)*100
    CPC=(6250000-budget)/clicks/1000
    return CTR,CPC,clicks,impression,budget


def predict_event_bidprice(PCTR,processed_staff,start_budget):
    bid_price=list()
    processed_data=processed_staff
    for i in list(range(0,len(processed_data))):
        pCTR = PCTR[i]*weights
        if pCTR > 0.8:
            pCTR=0.80000000000000           
        current_bid = base_bid * pCTR / avgCTR
        # Check if we still have budget:
        bid_price.append(current_bid)
    return bid_price

if __name__=="__main__":
    # MAIN:
    
    st=time.time()
    training_path = r"train.csv"
    validation_path = r"validation.csv"
    test_path=r"test.csv"
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
    
    
    #the range of base_bid
    #base_bids = list(range(5,25))
    #val_ctrs=list()
    #val_cpcs=list()
    
    #get relative optimal base_bid in the range
    #for base_bid in base_bids:
        #val_ctr,val_cpc=model_evaluator(model,processed_staff,start_budget,random_iter=10)
        #val_ctrs.append(val_ctr)
        #val_cpcs.append(val_cpc)
    #according to cpc_optimal to select optimal base_bid
    #cpc_index=val_cpcs.index(max(val_cpcs))
    #base_bid=base_bids[cpc_index]
    #according to ctr_optimal to select optimal base_bid
    #ctr_index=val_ctrs.index(max(val_ctrs))
    #base_bid=base_bids[ctr_index]
    
    
    CTR,CPC=model_evaluator(model,processed_staff,start_budget,random_iter=10)
    
    final_CTR,final_CPC,clicks,impression,budget=val_assessment(validation_path,model)

    
    test_events,ind=load_data(test_path,training=True,testing=True)
    PCTR,test_predict_label=predict_events_PCTR(model,test_events)
    predict_bid_price=predict_event_bidprice(PCTR,test_events,start_budget)

    #final results output as a csv
    out_file = open("fmalg.csv", "w", newline='') 
    writer = csv.writer(out_file)
    writer.writerow(['id','bid_price'])
    for i in range(len(predict_bid_price)):
        writer.writerow([ind[i],predict_bid_price[i]])
    out_file.close()    
    print (time.time()-st)