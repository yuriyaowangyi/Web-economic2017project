# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 22:20:20 2017

@author: Administrator
"""
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
import matplotlib.pyplot as plt
from scipy.stats import mode


 #Compute parameter C of two bidding functions 
def param_c(filepath,start_bid=50,end_bid=400,win_rate_model=2):
    payprice_list=list()
    temp_c=list()
    win=0
    count=0
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)
        for row in reader:
            payprice_list.append(int(row[22]))
    for bid_price in np.arange(start_bid,end_bid,50):
        for payprice in payprice_list:
                # Get the market price
                # Check if we win the bid
              if bid_price>payprice:
                 win=win+1  
                    # Check if the person clicks
              count=count+1
        win_rate=win/count
        if win_rate_model==1:
            temp_c.append((bid_price/win_rate)-bid_price)
        if win_rate_model==2:
            temp_c.append(np.sqrt((np.square(bid_price)/win_rate)-np.square(bid_price)))
    c=np.average(temp_c)
    return c        

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
    #Transform to String
    converter_dict={'weekday':str,'hour':str,'region':str,'city':str,'adexchange':str,'slotwidth':str,'slotheight':str,'slotvisibility':str,'slotformat':str,'advertiser':str}
    datas = pd.read_csv(path, skipinitialspace=True,usecols=features,converters=converter_dict)
    print('Preprocessing is starting\n')
    #Normalize slotprice
    datas['slotprice']=datas['slotprice']/std_of_slotprice
    print('Normalization of price finished\n')
    df = pd.read_csv(path, skipinitialspace=True,usecols=other_features)
    print('Deep preprocessing is starting\n')
    #Preprocessing useragent and usertag
    for i in list(range(0,len(df['usertag']))):
        instance=dict()
        instance.update(dict(datas.iloc[i]))
        op_sys, browser =df.iloc[i]['useragent'].split('_')
        instance.update({op_sys: True, browser: True})
        usertags=df.iloc[i]['usertag'].split(',')
        temp_dict = {}
        for tag in usertags:
            temp_dict["tag " + tag] = True
        instance.update(temp_dict)
        processed_data.append(instance)
        if testing==False:
            processed_labels.append(int(labels[i]))
        if i%10000==0:
            print(str(i)+'data has been preprocessed\n')
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
    # Use one hot-encoding
    label_encoder = LabelEncoder()
    vectorizer = DictVectorizer()
    train_event_x = vectorizer.fit_transform(training_data)
    train_event_y = label_encoder.fit_transform(labels)
    # Create and train the model.
    c = 1
    pctr_estimator = LogisticRegression(C=c, class_weight='balanced')
    pctr_estimator.fit(train_event_x, train_event_y)
    model = (pctr_estimator, label_encoder, vectorizer)
    print('Training done')
    return model

# Predict single PCTR
def predict_event_labels(instance, model): 
    pctr_estimator = model[0]
    # Transform event
    vectorizer = model[2]
    event = [instance]
    event_x = vectorizer.transform(event)
    event_y = pctr_estimator.predict_proba(event_x)
    return event_y[0][1]

# Predict whole labels and PCTR
def predict_events_PCTR(model,processed_data):
    predict_label=list()
    PCTR=list()
    for i in list(range(0,len(processed_data))):
        instance = processed_data[i]
        PCTR.append(predict_event_labels(instance, model))
        predict_label.append(int((PCTR[-1]>0.5)))
    return PCTR,predict_label

def predict_event_bidprice(p_c,p_lambda,PCTR,processed_staff,start_budget=6250000,testing=False,win_rate_model=2):
    clicks=0
    impressions=0
    budget=start_budget
    bid_price=list()
    CTR=0
    # Compute two bidding functions
    if testing==True:
        processed_data=processed_staff
        for i in list(range(0,len(processed_data))):
            pCTR = PCTR[i]
            if win_rate_model==1:
                current_bid =np.sqrt(((p_c/p_lambda)*pCTR)+p_c**2)-p_c 
            if win_rate_model==2:
                p1=pCTR+np.sqrt((p_c**2)*(p_lambda)**2+pCTR**2)
                p2=p_c*p_lambda
                current_bid=p_c*((p1/p2)**(1/3)-(p2/p1)**(1/3))
            bid_price.append(current_bid)
        return bid_price
    if testing==False:
        processed_data=processed_staff[0]
        processed_labels=processed_staff[1]
        payprices=processed_staff[2]   
        for i in list(range(0,len(processed_data))):
            pCTR = PCTR[i]
            if win_rate_model==1:
                current_bid =np.sqrt(((p_c/p_lambda)*pCTR)+p_c**2)-p_c 
            if win_rate_model==2:
                p1=pCTR+np.sqrt((p_c**2)*(p_lambda)**2+pCTR**2)
                p2=p_c*p_lambda
                current_bid=p_c*((p1/p2)**(1/3)-(p2/p1)**(1/3))
            # Check if we still have budget
            if budget > current_bid:
                # Get the market price
                payprice =payprices[i]
                # Check if we win the bid
                if current_bid > payprice:
                    impressions += 1
                    budget -= payprice
                    # Check if the person clicks
                    if testing==False:
                        if processed_labels[i] == 1:
                            clicks += 1
            else:
                break
        if impressions!=0:
            CTR=clicks/impressions
            CPC=((start_budget-budget)/clicks)/1000
        else:
            CTR=0
            CPC=0
        return CTR,CPC,clicks

# Compute best lambda
def Best_Lambda_Selector(model,p_c,validation_path, start_budget=6250000,win_rate_model=2):  
    true_label=list()
    lambdas=[10**(-x) for x in np.arange(2,8,0.1)]
    lambda_scores=list()
    processed_data,processed_labels,payprices=load_data(validation_path,training=False)
    true_label=processed_labels
    PCTR,predict_label=predict_events_PCTR(model,processed_data)
    total_scores=np.zeros(len(lambdas))
    total_CPC_scores=np.zeros(len(lambdas))
    total_clicks=np.zeros(len(lambdas))
    for i in range(100):
        y=random.sample(list(range(0,len(PCTR))),200000)
        data=[processed_data[x] for x in y]
        labels=[processed_labels[x] for x in y]
        payprice=[payprices[x] for x in y]
        sample_PCTR=[PCTR[x] for x in y]
        j=0
        for p_lambda in lambdas:
            processed_staff=(data,labels,payprice)
            temp_CTR,temp_CPC,clicks=predict_event_bidprice(p_c,p_lambda,sample_PCTR,processed_staff,start_budget=6250000,testing=False,win_rate_model=2)
            total_scores[j]=total_scores[j]+temp_CTR
            total_CPC_scores[j]=total_CPC_scores[j]+temp_CPC
            total_clicks[j]=total_clicks[j]+clicks
            j=j+1
        print(i)
    lambda_scores=list(total_scores/100)
    lambda_CPC_scores=list(total_CPC_scores/100)
    lambda_clicks=list(total_clicks/100)
    p_lambda=lambdas[lambda_scores.index(max(lambda_scores))]
    CTR=lambda_scores[lambda_scores.index(max(lambda_scores))]
    click=lambda_clicks[lambda_scores.index(max(lambda_scores))]
    #CPC=lambda_CPC_scores[lambda_CPC_scores.index(min(lambda_CPC_scores))]
    CPC=lambda_CPC_scores[lambda_scores.index(max(lambda_scores))]
    return CTR*100,CPC,click,p_lambda,predict_label,true_label,lambda_scores,lambda_CPC_scores

def val_assessment(validation_path,model,c,p_lambda):
    val_events,ind=load_data(validation_path,training=True,testing=True)
    PCTR,val_predict_label=predict_events_PCTR(model,val_events)
    predict_bid_price=predict_event_bidprice(c,p_lambda,PCTR,val_events,start_budget = 6250000,testing=True,win_rate_model=2)
    df_val=pd.read_csv(validation_path)
    budget=6250000
    impression=0
    clicks=0
    for i in list(range(0,len(df_val))):
        if budget<0:
            break
        if df_val.iloc[i].payprice<predict_bid_price[i]:
            budget=budget-df_val.iloc[i].payprice
            impression=impression+1
            if df_val.iloc[i].click==1:
                clicks=clicks+1
    CTR=(clicks/impression)*100
    CPC=(6250000-budget)/clicks
    return CTR,CPC/1000,clicks,impression,budget
    
    
if __name__=="__main__":
    # MAIN:
    '''
    st=time.time()
    training_path = r"dataset/train.csv"
    validation_path = r"dataset/validation.csv"
    test_path=r"dataset/test.csv"
    slotprices = pd.read_csv(training_path, skipinitialspace=True,usecols=['slotprice'])
    global std_of_slotprice
    std_of_slotprice=int(slotprices['slotprice'].values.std())
    # Extracting data:
    training_events, labels= load_data(training_path)
    # training model
    
    c=param_c(training_path,start_bid=50,end_bid=400,win_rate_model=2)
    '''
    model= train(training_events, labels)
    '''
    val_CTR,val_CPC,click,p_lambda,predict_l,true_l,CTR_scores,CPC_scores=Best_Lambda_Selector(model,c,validation_path,start_budget = 6250000,win_rate_model=2)
    '''
    final_CTR,final_CPC,clicks,impression,budget=val_assessment(validation_path,model,c,p_lambda)
    '''
    test_events,ind=load_data(test_path,training=True,testing=True)
    PCTR,test_predict_label=predict_events_PCTR(model,test_events)
    predict_bid_price=predict_event_bidprice(c,p_lambda,PCTR,test_events,start_budget = 6250000,testing=True,win_rate_model=2)
    out_file = open("/Users/apple/Downloads/winningrate2.csv", "w", newline='') 
    writer = csv.writer(out_file)
    writer.writerow(['id','bid_price'])
    for i in range(len(predict_bid_price)):
        writer.writerow([ind[i],predict_bid_price[i]])
    out_file.close()    
    print (time.time()-st)
    
    fpr, tpr, thresholds = metrics.roc_curve(true_l, predict_l, pos_label=1)
    metrics.auc(fpr, tpr)
    bid_std=np.std(predict_bid_price)
    bid_mean=np.mean(predict_bid_price)
    plt.hist(predict_bid_price, 200, normed=True)
    plt.xlim([1,130])
    plt.xlabel('predict_bid_price')
    plt.ylabel('Count')
    plt.title('predict_bid_price count')'''