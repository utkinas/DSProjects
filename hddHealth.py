
import os
import datetime
import json
import matplotlib.pyplot as plt

import plotly
import sys
#from urllib2 import urlopen
import urllib3
import bs4
import html5lib
import html.parser
from html.parser import HTMLParser
import requests
import pandas as pd
import re
import csv
from os import listdir
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler





def readCSVFiles(d1,path):
    #d1 = pd.DataFrame()

    for f in listdir(path):

        filename = path + f
        ds = pd.read_csv(filename)
        #d1 = d1.append(ds).fillna(0)
        pd.concat([d1, ds], axis=0)

        return d1



path = "/Users/sveta/python/InsightDataScience/data/"

def getHDDStat():
    os.system("smartctl -aj -s on /dev/disk0 > /Users/sveta/python/test/test2.js")
    jsFile = "/Users/sveta/python/test/test2.js"
    data = json.load(open(jsFile))
    arr=[]

    df = pd.DataFrame(data["ata_smart_attributes"])
    for p in df['table']:
        arr.append( [p['id'],p['name'],p['value']])


    dfstat = pd.DataFrame(arr)
    dfstat1 = dfstat.rename(columns={0:'id',1:'name',2:'value'})
    dfstat1.to_csv("/Users/sveta/python/test/dfstat.csv",index = False)


filename0 = '2019-09-30.csv'
#initial file
d0 = pd.read_csv(path+filename0)[:0]


yearArr = [2019]
qtrArr = [3,4]

def loadFiles(d0,year,qtr):


    dfinal = d0
    #for qtr in qtrArr:
    pathDownloads = "/Users/sveta/Downloads/data_Q"+str(qtr)+"_"+str(year)+"/"

    if os.path.exists(pathDownloads):

        for f in listdir(pathDownloads):
            if '.csv' in f:
                filename = pathDownloads + f
                #print(filename)

                ds = pd.read_csv(filename)
                # d1 = d1.append(ds).fillna(0)
                dfinal = pd.concat([dfinal, ds], axis=0)
                print(filename, len(dfinal),len(ds))

    dfinal.to_pickle("/Users/sveta/python/InsightDataScience/data/hdd_"+str(qtr)+"_"+str(year)+".pkl")
    print("finished loading " +str(year) + ", "+ str(qtr))
    #return dfinal

print (datetime.datetime)

def loadingQtrData():
    for year in yearArr:
        for qtr in qtrArr:
            loadFiles(d0,year,qtr)




#dfinal = readCSVFiles(d1,path)

#dfinal.to_pickle("/Users/sveta/python/InsightDataScience/data/hdd.pkl")



dfinal = pd.read_pickle("/Users/sveta/python/InsightDataScience/data/hdd_3_2019.pkl")
#print(list(dfinal.columns))
#print(len(dfinal))
#input()



#########################################################
# Exploratory data analysis
#########################################################




dfSorted = dfinal.sort_values(by = ['date','model'])
#dfFailures=dfSorted[['date','model','failure']]
dfFailures = dfSorted[dfSorted.failure ==1]
dfSuccess = dfSorted[dfSorted.failure ==0]

df_fail = dfFailures[['capacity_bytes', 'smart_1_raw', 'smart_5_raw', 'smart_7_raw',
        'smart_9_raw']]

df_success = dfSuccess[['capacity_bytes','smart_1_raw', 'smart_5_raw', 'smart_7_raw',
        'smart_9_raw']]



df_success.hist()
df_fail.hist()
plt.show()


input()




cols = ['date', 'model', 'serial_number', 'capacity_bytes','failure',  'smart_1_raw', 'smart_5_raw', 'smart_7_raw',
        'smart_9_raw',  'smart_12_raw', 'smart_173_raw', 'smart_174_raw',
        'smart_192_raw', 'smart_194_raw', 'smart_195_raw','smart_197_raw', 'smart_199_raw', 'smart_240_raw','smart_241_raw']

dfSorted['modelName']=dfSorted['model']

dfModels = dfinal['model'].drop_duplicates().reset_index()
dfModels['modelType'] = dfModels.index

dfSorted = pd.merge(left = dfSorted,right = dfModels, how='left', left_on='model',right_on = 'model')

print(dfSorted)
input()
#dfModels.to_csv("/Users/sveta/python/InsightDataScience/data/hddModels.csv")
#print(dfModels)


#df0 = dfSorted[['serial_number','failure']].groupby(by = ['serial_number','failure']).count()
df1 = dfSorted[['date','model','failure']].groupby(by = ['model']).count()
df2 = dfSorted[['model','serial_number']].groupby(by = ['model']).count()

df3 = dfSorted[['model','smart_9_raw']]
df3.to_csv("/Users/sveta/python/InsightDataScience/data/hdd/avgModelHour.csv")
df4 = dfFailures[['model','failure']].groupby(by = ['model']).count()
df4.to_csv("/Users/sveta/python/InsightDataScience/data/hdd/failureCnt.csv")

#plt.hist(df3.smart_9_raw, bins='auto', color='#0504aa')

plt.hist(df4.failure, bins='auto', color='red')
plt.xlabel("HDD model")
plt.ylabel("Failure count")
plt.show()


#histogram by hours working
#

#df_fail = dfFailures[['capacity_bytes','failure',  'smart_1_raw', 'smart_5_raw', 'smart_7_raw',
#        'smart_9_raw',  'smart_12_raw', 'smart_173_raw', 'smart_174_raw',
#        'smart_192_raw', 'smart_194_raw', 'smart_195_raw','smart_197_raw', 'smart_199_raw', 'smart_240_raw','smart_241_raw']]

#df_success = dfSuccess[['capacity_bytes','failure',  'smart_1_raw', 'smart_5_raw', 'smart_7_raw',
#        'smart_9_raw',  'smart_12_raw', 'smart_173_raw', 'smart_174_raw',
#        'smart_192_raw', 'smart_194_raw', 'smart_195_raw','smart_197_raw', 'smart_199_raw', 'smart_240_raw','smart_241_raw']]


df2.to_csv("/Users/sveta/python/InsightDataScience/data/hdd/modelCnt.csv")
#df0.to_csv("/Users/sveta/python/InsightDataScience/data/hdd/modelCntFailure.csv")

failedID = list(dfFailures['serial_number'])


dfFailures[cols].to_csv("/Users/sveta/python/InsightDataScience/data/hdd/failures.csv")


df1sorted = df1.sort_values(by='date')
print(df1sorted)
print('done')

input()


#filename = '2019-04-01.csv'
#filenameTest = '2019-06-28.csv'

ds = dfSorted
#dsTest0 = pd.read_csv(path + filenameTest)
dsTest0 = pd.read_pickle("/Users/sveta/python/InsightDataScience/data/hdd_4_2016.pkl")

dsTest = dsTest0[dsTest0.capacity_bytes>0]

dsTest = pd.merge(left = dsTest,right = dfModels, how='left', left_on='model',right_on = 'model')


arrCol = ds.columns

#simpleFeatures = ['capacity_bytes','smart_1_raw', 'smart_5_raw', 'smart_9_raw', 'smart_10_raw','smart_12_raw',  'smart_173_raw',
#                  'smart_174_raw', 'smart_192_raw', 'smart_194_raw', 'smart_197_raw', 'smart_199_raw', 'smart_240_raw']

simpleFeatures = ['modelType','capacity_bytes','smart_1_raw', 'smart_5_raw', 'smart_7_raw','smart_9_raw',  'smart_12_raw', 'smart_173_raw', 'smart_174_raw',  'smart_192_raw', 'smart_194_raw', 'smart_195_raw','smart_197_raw', 'smart_199_raw', 'smart_240_raw','smart_241_raw']

#simpleFeatures = ['modelType','capacity_bytes','smart_1_normalized', 'smart_5_normalized', 'smart_9_normalized','smart_12_normalized', 'smart_192_normalized', 'smart_194_normalized', 'smart_240_normalized']



#'smart_169_raw','smart_175_raw' -  not in dataset
#features=['serial_number','model','capacity_bytes']
features=['capacity_bytes']


for item in arrCol:
    if '_raw' in item:
        features.append(item)

#print(features)
#
dstrain = ds[ds.capacity_bytes>0]

#ds0 =
Y = dstrain[['failure']]

X = dstrain[simpleFeatures]



Xtest = dsTest[simpleFeatures]
Ytest = dsTest[['failure']]



#cols = ['date', 'model', 'serial_number', 'capacity_bytes','failure',    'smart_1_normalized', 'smart_5_normalized', 'smart_9_normalized','smart_12_normalized', 'smart_192_normalized', 'smart_194_normalized', 'smart_240_normalized']


XtestValidate = dsTest[cols]

#print(Ytest)
#input()


scaler = StandardScaler()


X1 = X.fillna(0)
#X1=X.dropna()

X_std = scaler.fit_transform(X1)

Xtest1 = Xtest.fillna(0)
#Xtest1 =Xtest.dropna()

#scaler = StandardScaler()
#X_std = scaler.fit_transform(X1)

clf = LogisticRegression(random_state=0, class_weight='balanced').fit(X1, Y)
df = pd.DataFrame(clf.predict(Xtest1))

df.rename(columns={0 : "predictedFailure"})
df.to_csv("/Users/sveta/python/InsightDataScience/data/hdd/testmodel.csv")





#print(df[df[0]==1])
print(Ytest[Ytest.failure==1])






print(clf.score(X1, Y))

dsValidate = pd.merge(left = XtestValidate,right = df, how = 'left',left_index=True, right_index=True)

dsValidate1=dsValidate[dsValidate.failure==1]

dsValidate1.to_csv("/Users/sveta/python/InsightDataScience/data/hdd/validate.csv")
input()
