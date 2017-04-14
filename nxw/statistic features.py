#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 21:48:06 2017

@author: apple
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def statstics_summary(df,advertiser):
    total_num=len(df)
    temp_df=df[df['advertiser']==advertiser]
    impressions=len(temp_df['click'])
    clicks=sum(temp_df['click'])
    cost=sum(temp_df['payprice'])/1000
    bid=sum(temp_df['bidprice'])/1000
    win_rate=impressions/total_num
    CTR=(clicks/impressions)*100
    avgCPM=(cost/impressions)*1000
    eCPC=(cost/clicks)
    return [advertiser,impressions,clicks,cost,bid,win_rate,CTR,avgCPM,eCPC]

def os_browser_slotsize(df_ad):
    df=df_ad
    os_sys_list=list()
    browser_list=list()
    slotsize_list=list()
    for i in list(range(0,len(df['useragent']))):
        op_sys, browser =df.iloc[i]['useragent'].split('_')
        os_sys_list.append(op_sys)
        browser_list.append(browser)
        width=str(df.iloc[i]['slotwidth'])
        height=str(df.iloc[i]['slotheight'])
        slotsize_list.append(width+'*'+height)
    return os_sys_list,browser_list,slotsize_list

def CTR_features(df_ad,feature):
    df=df_ad
    feature_list=sorted(list(set(df[feature])))
    CTRs=[sum(df[df[feature]==x]['click'])*100/len(df[df[feature]==x]) for x in feature_list]
    return feature_list,CTRs
    
def plt_2_ad(feature,feature_list,CTR1,CTR2):
    if type(feature_list[0])!=str:
        plt.figure()
        plt.errorbar(feature_list,CTR1,fmt='-o')
        plt.errorbar(feature_list,CTR2,fmt='-o')
        plt.xlabel(feature)
        plt.ylabel('CTR')
        plt.title('CTR '+'and '+feature)
        plt.legend([1458,3358],loc='upper right')
    else:
        plt.figure()
        num_list=feature_list
        num_list=range(len(num_list))
        plt.errorbar(num_list,CTR1,fmt='-o')
        plt.errorbar(num_list,CTR2,fmt='-o')
        plt.xticks(num_list,feature_list)
        if len(feature_list)<4:
            plt.xlim([-2,4])
        plt.xlabel(feature)
        plt.ylabel('CTR')
        plt.title('CTR '+'and '+feature)
        plt.legend([1458,3358],loc='upper right')
    return 0

def pay_feature(df,feature):
    feature_list=sorted(list(set(df[feature])))
    feature_std_list=list()
    feature_mean_list=list()
    for fea in feature_list:
        temp_df=df[df[feature]==fea]
        temp_std=temp_df['payprice'].values.std()
        temp_mean=temp_df['payprice'].values.mean()
        feature_std_list.append(temp_std)
        feature_mean_list.append(temp_mean)
    return feature_list,feature_mean_list,feature_std_list

def plt_2_ad_pay(feature,feature_list,mean1,std1,mean2,std2):
    if type(feature_list[0])!=str:
        plt.figure()
        plt.errorbar(feature_list,mean1,yerr=std1,fmt='-o')
        plt.errorbar(feature_list,mean2,yerr=std2,fmt='-o')
        plt.xlabel(feature)
        plt.ylabel('Payprice')
        plt.title('Payprice '+'and '+feature)
        plt.legend([1458,3358],loc='upper right')
    else:
        plt.figure()
        num_list=feature_list
        num_list=range(len(num_list))
        plt.errorbar(num_list,mean1,yerr=std1,fmt='o')
        plt.errorbar(num_list,mean2,yerr=std2,fmt='o')
        plt.xticks(num_list,feature_list)
        if len(feature_list)<4:
            plt.xlim([-2,4])
        plt.xlabel(feature)
        plt.ylabel('Payprice')
        plt.title('Payprice '+'and '+feature)
        plt.legend([1458,3358],loc='upper right')
    return 0

def quick_plot_1(df1,df2,feature):
    feature_list,CTRs1=CTR_features(df1,feature)
    feature_list,CTRs2=CTR_features(df2,feature)
    plt_2_ad(feature,feature_list,CTRs1,CTRs2)
    return 0
def quick_plot_2(df1,df2,feature):
    feature_list,mean1,std1=pay_feature(df1,feature)
    feature_list,mean2,std2=pay_feature(df2,feature)
    plt_2_ad_pay(feature,feature_list,mean1,std1,mean2,std2)
    return 0
'''
training_path = r"dataset/train.csv"
validation_path = r"dataset/validation.csv"
test_path=r"dataset/test.csv"
features=['weekday', 'hour','region','adexchange','city','slotwidth', 'slotheight','useragent','bidprice','payprice','click','advertiser']
df_train=pd.read_csv(training_path,skipinitialspace=True,usecols=features)

advertisers_list=list(set(df_train['advertiser'].values))
summarys=[statstics_summary(df_train,advertiser) for advertiser in advertisers_list]
fm=pd.DataFrame(summarys,columns=['advertiser','impressions','clicks','cost','bid','win_rate','CTR','avgCPM','eCPC'])

#1458 AND 3358
train_1458=df_train[df_train['advertiser']==1458].copy(deep=True)
os_sys_list,browser_list,slotsize_list=os_browser_slotsize(train_1458)
train_1458.is_copy=False
train_1458['OS']=os_sys_list
train_1458['Browser']=browser_list
train_1458['slotsize']=slotsize_list

train_3358=df_train[df_train['advertiser']==3358].copy(deep=True)
os_sys_list,browser_list,slotsize_list=os_browser_slotsize(train_3358)
train_3358.is_copy=False
train_3358['OS']=os_sys_list
train_3358['Browser']=browser_list
train_3358['slotsize']=slotsize_list
'''
quick_plot_1(train_1458,train_3358,'weekday')
quick_plot_1(train_1458,train_3358,'hour')
quick_plot_1(train_1458,train_3358,'OS')
quick_plot_1(train_1458,train_3358,'Browser')
quick_plot_1(train_1458,train_3358,'region')
quick_plot_1(train_1458,train_3358,'adexchange')

quick_plot_2(train_1458,train_3358,'weekday')
quick_plot_2(train_1458,train_3358,'hour')
quick_plot_2(train_1458,train_3358,'OS')
quick_plot_2(train_1458,train_3358,'Browser')
quick_plot_2(train_1458,train_3358,'region')
quick_plot_2(train_1458,train_3358,'adexchange')

"""bid_std=predict_bid_price.std()
bid_mean=predict_bid_price.mean()
plt.hist(predict_bid_price, 20, normed=True, histtype='step', cumulative=True)
plt.xlabel('predict_bid_price')
plt.ylabel('Count')
plt.title('predict_bid_price count')"""

