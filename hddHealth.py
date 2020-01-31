
import os
import datetime
import json
import matplotlib.pyplot as plt
import gc
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
import numpy as np
import re
import csv
from os import listdir
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from imblearn import FunctionSampler
from imblearn.pipeline import make_pipeline

from matplotlib import pyplot
from datetime import datetime
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE # doctest: +NORMALIZE_WHITESPACE
from zipfile import ZipFile
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import ClusterCentroids
from sklearn import linear_model
import pickle


def hddHealth():
    path = "data/"

    def getHDDStat():
        #os.system("smartctl -aj -s on /dev/disk0 > /Users/sveta/python/test/stat.js")
        jsFile = "data/stat.js"
        data = json.load(open(jsFile))
        arr=[]

        df = pd.DataFrame(data["ata_smart_attributes"])
        for p in df['table']:

            arr.append( [p['id'],p['name'],p['raw']['value']])


        dfstat = pd.DataFrame(arr)
        dfstat1 = dfstat.rename(columns={0:'id',1:'name',2:'value'})
        dfstat1.to_csv("data/dfstat.csv",index = False)



    #getHDDStat()

    filename0 = 'testFile.csv'
    #initial file
    d0 = pd.read_csv(path+filename0)[:0]


    yearArr = [2017]
    qtrArr = [1,2,3,4]

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

    def loadingQtrData():
        for year in yearArr:
            for qtr in qtrArr:
                loadFiles(d0,year,qtr)


    def pickleToHdfConvert(filepathPkl,filepathHdf):
        df = pd.read_pickle(filepathPkl)
        df.to_hdf(filepathHdf, key='dsTest0', mode='w')



    def dataPreprocessing(filepathHdf, preprocessedDataPath):

        print('reading hdf5')
        dfinal = pd.read_hdf(filepathHdf)

        print("Current Time =", datetime.now().strftime("%H:%M:%S"))


        dfSorted = dfinal.sort_values(by = ['date','model'])


        cols = ['date', 'model', 'serial_number', 'capacity_bytes','failure',  'smart_1_raw', 'smart_5_raw', 'smart_7_raw',
                'smart_9_raw',  'smart_12_raw', 'smart_173_raw', 'smart_174_raw',
                'smart_192_raw', 'smart_194_raw', 'smart_195_raw','smart_197_raw', 'smart_199_raw', 'smart_240_raw','smart_241_raw']


        smartArr =['capacity_bytes','smart_1_raw', 'smart_5_raw', 'smart_7_raw',
                'smart_9_raw',
                'smart_192_raw', 'smart_193_raw','smart_194_raw', 'smart_195_raw', 'smart_240_raw','smart_241_raw','smart_242_raw']


        dfSorted = dfinal.sort_values(by = ['date','model'])



        dfSorted['modelName']=dfSorted['model']

        dfModels = dfinal['model'].drop_duplicates().reset_index()
        dfModels['modelType'] = dfModels.index

        dfSorted = pd.merge(left = dfSorted,right = dfModels, how='left', left_on='model',right_on = 'model')

        del(dfinal)
        gc.collect()




        dfFailures = dfSorted[dfSorted.failure ==1]
        dfSuccess = dfSorted[dfSorted.failure ==0]


        df_fail = dfFailures[['model','modelType','failure','capacity_bytes', 'smart_1_raw', 'smart_5_raw', 'smart_7_raw',
                'smart_9_raw',  'smart_12_raw', 'smart_173_raw', 'smart_174_raw',
                'smart_192_raw', 'smart_193_raw','smart_194_raw', 'smart_195_raw','smart_197_raw', 'smart_199_raw', 'smart_240_raw','smart_241_raw','smart_242_raw']]

        df_success = dfSuccess[['model','modelType','failure','capacity_bytes','smart_1_raw', 'smart_5_raw', 'smart_7_raw',
                'smart_9_raw',  'smart_12_raw', 'smart_173_raw', 'smart_174_raw',
                'smart_192_raw', 'smart_193_raw','smart_194_raw', 'smart_195_raw','smart_197_raw', 'smart_199_raw', 'smart_240_raw','smart_241_raw','smart_242_raw']]

        cols = ['model','modelType','failure','capacity_bytes','smart_1_raw', 'smart_5_raw', 'smart_7_raw',
                'smart_9_raw',  'smart_12_raw', 'smart_173_raw', 'smart_174_raw',
                'smart_192_raw', 'smart_193_raw','smart_194_raw', 'smart_195_raw','smart_197_raw', 'smart_199_raw',
                'smart_240_raw','smart_241_raw','smart_242_raw']



        models = list(dfModels['model'])

        newModelsList=[]

        #filter out Models without any failure

        def filterModels():

            for item in models:
                df_model_fail = df_fail[df_fail.model==item]
                #df_model_success = df_success[df_success.model == item]
                if len(df_model_fail)>0:
                    newModelsList.append(item)


        newModelsList = ['ST4000DM000', 'ST12000NM0007', 'HGST HMS5C4040ALE640', 'ST8000NM0055', 'ST8000DM002', 'HGST HMS5C4040BLE640', 'TOSHIBA MG07ACA14TA', 'HGST HUH721212ALN604', 'HGST HUH721212ALE600', 'TOSHIBA MQ01ABF050', 'ST500LM030', 'ST6000DX000', 'ST10000NM0086', 'TOSHIBA MQ01ABF050M', 'WDC WD5000LPVX', 'ST500LM012 HN', 'HGST HUH728080ALE600', 'ST8000DM005', 'Seagate BarraCuda SSD ZA500CM10002', 'ST12000NM0117']

        print(newModelsList)


        df_updated = dfSorted[dfSorted['model'].isin(newModelsList)]

        print(df_updated.head(),len(df_updated) )




        df_updated.to_hdf(preprocessedDataPath, key='dfinal', mode='w')

        print('done ')
        print("Current Time =", datetime.now().strftime("%H:%M:%S"))


        del (dfSorted)
        gc.collect()

        return df_updated

    def EDA(df, dfFailures,dfSuccess):

        #########################################################
        # Exploratory data analysis
        #########################################################
        print('Starting EDA ...')
        #df0 = dfSorted[['serial_number','failure']].groupby(by = ['serial_number','failure']).count()
        #df1 = df[['date','model','failure']].groupby(by = ['model']).count()
        #df2 = df[['model','serial_number']].groupby(by = ['model']).count()

        df_pairs = dfFailures[['model','serial_number','failure']].groupby(['model','serial_number']).count()
        print(len(df_pairs))
        print(df_pairs)


        df_pairsNonFail = dfSuccess[['model','serial_number','failure']].groupby(['model','serial_number']).count()

        df_pairs.to_csv("/Users/sveta/python/InsightDataScience/data/uniquePairFailed.csv")
        df_pairsNonFail.to_csv("/Users/sveta/python/InsightDataScience/data/uniquePairNonFailed.csv")

        df_pairs = pd.read_csv("/Users/sveta/python/InsightDataScience/data/uniquePairFailed.csv")
        df_pairsNonFail = pd.read_csv("/Users/sveta/python/InsightDataScience/data/uniquePairNonFailed.csv")

        serNumberFailedArr = list(df_pairs['serial_number'])

        serNumberNeverFailed = df_pairsNonFail[~df_pairsNonFail.serial_number.isin(serNumberFailedArr)]
        serNumberNeverFailed.to_csv("/Users/sveta/python/InsightDataScience/data/serNumberNeverFailed.csv")


        serNumberNeverFailedArr = list(serNumberNeverFailed['serial_number'])

        #saving list of serial numbers of failed devices
        pd.DataFrame(serNumberFailedArr).to_csv("/Users/sveta/python/InsightDataScience/data/serNumberFailed.csv")


        df_serNumber = df[ ['date','serial_number']].groupby(by = ['serial_number']).count().rename(columns = {'date': 'dayCnt'})
        df_serNumber.to_csv("/Users/sveta/python/InsightDataScience/data/serNumberDays.csv")
        df_serNumber = pd.read_csv("/Users/sveta/python/InsightDataScience/data/serNumberDays.csv")

        #print('df_serNumber')
        print(df_serNumber.columns)

        df_maxDaysCnt = df_serNumber[['serial_number','dayCnt']].max().rename(columns = {'dayCnt':'maxDaysCnt'})
        df_maxDaysCnt.to_csv("/Users/sveta/python/InsightDataScience/data/maxDaysCnt.csv")

        #df_serNumberFailed = dfFailures[['serial_number','failure']].groupby(by = ['serial_number']).count()

        #df_serNumberFailed['serial_number'].to_csv("/Users/sveta/python/InsightDataScience/data/serNumberFailed.csv")

        #df_serNumber.to_csv("/Users/sveta/python/InsightDataScience/data/serNumber.csv")

        df_updated = pd.merge( left = df, right = df_serNumber, how = 'left', left_on = 'serial_number', right_on = 'serial_number')

        #df_updated['failDay'] = np.NaN
        #df_updated['failDay'] = [x if df_updated.failure ==1 else np.NaN for x in df_updated['failDay'] ]
        print (df_updated)
        #input()

        df_ModelSerNumber = df_updated[['model', 'serial_number','date']].groupby(by=['model','serial_number']).count()
        df_ModelSerNumber.to_csv("/Users/sveta/python/InsightDataScience/data/serNumber.csv")

        #df3 = df_updated[['model','smart_9_raw']]

        #df3avgHour = df_updated[['model','smart_9_raw']].groupby(by = ['model']).mean()
        #df3stdHour = df_updated[['model','smart_9_raw']].groupby(by = ['model']).std()


        #df_workHours = df_updated[['model','smart_9_raw']].groupby(by = ['model']).sum()

        #Saving
        #df3.to_csv("/Users/sveta/python/InsightDataScience/data/hdd/avgModelHour.csv")

        #df4 = dfFailures[['model','failure']].groupby(by = ['model']).count()
        #df4.to_csv("/Users/sveta/python/InsightDataScience/data/hdd/failureCnt.csv")
        print('done EDA')
        return serNumberFailedArr,serNumberNeverFailedArr


    def readingTestDataset(filepath, dfModels, features):
        dsTest0 = pd.read_hdf(filepath)
        dsTest = dsTest0[dsTest0.capacity_bytes > 0]
        dsTest = pd.merge(left=dsTest, right=dfModels, how='left', left_on='model', right_on='model')

        # validate for another quarter
        Xtest = dsTest[features]
        Ytest = dsTest[['failure']]
        X1test = Xtest.fillna(0)

        return X1test, Ytest

    print("Current Time =", datetime.now().strftime("%H:%M:%S"))
    #df_updated = dataPreprocessing()
    simpleFeatures = ['capacity_bytes','smart_1_raw', 'smart_5_raw', 'smart_7_raw','smart_9_raw', 'smart_12_raw','smart_192_raw', 'smart_194_raw', 'smart_195_raw','smart_197_raw', 'smart_199_raw', 'smart_240_raw','smart_241_raw']

    print('Read preprocessed data')

    df_updated = pd.read_hdf("data/df_Updated.h5")
    dfModels = pd.read_csv("data/modelList.csv")



    #print(df_updated.columns)

    dfFailures = df_updated[df_updated.failure==1]
    dfSuccess = df_updated[df_updated.failure==0]

    def calculateMean():
        dfmean = pd.DataFrame(dfFailures[simpleFeatures].mean()).rename(columns={0: 'Fail'})
        dfmean = pd.DataFrame(list(dfmean['Fail'])).rename(columns={0: 'Fail'})
        dfmedian = dfFailures[simpleFeatures].median()

        dfmeanSuccess = pd.DataFrame(dfSuccess[simpleFeatures].sample(n=600).mean()).rename(columns={0: 'nonFail'})
        dfmeanSuccess = pd.DataFrame(list(dfmeanSuccess['nonFail'])).rename(columns={0: 'nonFail'})

        dfcolumns = pd.DataFrame(simpleFeatures).rename(columns={0:'name'})




        dfallMean = pd.merge(left = dfcolumns,right=dfmean, how = 'left', right_index=True,left_index=True)
        dfallMean = pd.merge(left = dfallMean,right=dfmeanSuccess, how = 'left', right_index=True,left_index=True)

        dfallMean.to_csv("data/allmean.csv")




    def sample_batch(indices: list, batch_size: int, probs: list):
        return np.random.choice(indices, batch_size, probs, replace=True)


    print(len(dfFailures),len(dfSuccess))

    print("Current Time =", datetime.now().strftime("%H:%M:%S"))


    #EDA(df_updated,dfFailures,dfSuccess)


    #read serial number of devices which never failed  - and exclude them
    df_neverfailed = pd.read_csv("data/serNumberNeverFailed.csv")

    serNumberNeverFailed = df_neverfailed[df_neverfailed.failure>=50]




    print("whole ds ",len(df_updated))
    ds = df_updated[~df_updated.serial_number.isin(list(serNumberNeverFailed.serial_number))]




    #simpleFeatures = ['capacity_bytes','smart_1_raw', 'smart_5_raw', 'smart_7_raw','smart_9_raw', 'smart_12_raw','smart_192_raw', 'smart_194_raw', 'smart_195_raw','smart_197_raw', 'smart_199_raw', 'smart_240_raw','smart_241_raw']
    #simpleFeatures = ['capacity_bytes','smart_5_raw', 'smart_7_raw','smart_9_raw', 'smart_12_raw','smart_173_raw','smart_192_raw', 'smart_194_raw','smart_240_raw']
    #                 [0.02656122     0.03499287     0.18742727     0.04481702    0.43668853     0.00379877      0.00102822       0.14376292       0.12092319]
    #simpleFeatures = ['modelType','capacity_bytes','smart_5_raw','smart_9_raw','smart_240_raw']

    features1=['capacity_bytes']

    #ds = df_updated



    print("after exclusion",len(ds))

    arrCol = ds.columns
    def getRawCols():
        for item in arrCol:
            if '_raw' in item:
                features1.append(item)

    #print(features)
    #

    dstrain = ds[ds.capacity_bytes>0]


    print(len(dstrain[dstrain.failure==0]),len(dstrain[dstrain.failure==1]),len(dstrain) )

    Y = dstrain[['failure']]

    X = dstrain[simpleFeatures]

    print('Original dataset shape %s' % Counter(Y))

    #split 80:20
    #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)


    # Set up training dataset
    X_train0 = X
    y_train0 = Y

    print("Number transactions X_train dataset: ", X_train0.shape)
    print("Number transactions y_train dataset: ", y_train0.shape)


    print("Current Time =", datetime.now().strftime("%H:%M:%S"))

    X_train = X_train0.fillna(0)


    #clfSDG = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
    #clf = LogisticRegression(random_state=0, class_weight='balanced').fit(X1, Y)
    #clf1 = LogisticRegression(random_state=0, class_weight='balanced').fit(X_train1, y_train)



    print('Handling imbalanced data - starting SMOTE ...')


    def overSamplingWithSMOTE(X_train,y_train):
        sm = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)

        #convert pandas to array
        y_train = np.array(y_train0.failure)

        #Create final training dataset
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

        return X_train_res, y_train_res


    X_train_res, y_train_res = overSamplingWithSMOTE(X_train,y_train0)

    df_sm = pd.DataFrame(y_train_res)
    #print(len(df_sm[df_sm[0]==1]),len(df_sm[df_sm[0]==0]))

    print('After OverSampling, the shape of train_X: ',X_train_res.shape)
    print('After OverSampling, the shape of train_y: ',y_train_res.shape)



    #print(clf2.feature_importances_)


    #Prediction using logistic regression
    Xtest,Ytest = readingTestDataset("data/hdd_4_2016.h5",dfModels, simpleFeatures)


    def logisticRegression(X_train_res, y_train_res, Xtest, Ytest):
        clf2 = LogisticRegression().fit(X_train_res, y_train_res)
        #predictions2 = clf2.predict(Xtest)

        filename = 'finalized_model.sav'
        pickle.dump(clf2, open(filename, 'wb'))


    #logisticRegression(X_train_res, y_train_res, Xtest, Ytest)

    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    predictions3 = loaded_model.predict(Xtest)
    print('results after over sampling, logistic regressioin, no device filtering')

    print(classification_report(Ytest, predictions3))
    print(confusion_matrix(Ytest, predictions3))

    def randomForest(X_train_res, y_train_res):
        #Random forests
        print('starting random forest')
        clfRF = RandomForestClassifier(max_depth=2, random_state=0).fit(X_train_res, y_train_res)
        print('feature impoartance for RF')
        print(clfRF.feature_importances_)
        input()



    #for i in range
    #prediction_sgd = clfSDG.partial_fit(X_batch, y_batch)



    #for i in range(0, epoch):
     #   x batch, ybatch=sample
        #np.random.choice(aamilnearr, 5, p=[0.5, 0.1, 0.1, 0.3])
        #clf.partialfit(Xminibatch, y_minibatch)

    def sample_batch(indices: list, batch_size: int, probs: list):
        return np.random.choice(indices, batch_size, probs, replace=True)


    def batchSample(dfsampleSuccess,dfsampleFailure,features):

        dfSuccessbatch = dfsampleSuccess.sample(n = 150)
        dfFailurebatch = dfsampleFailure.sample(n = 150)


        dfBatch = pd.concat([dfSuccessbatch, dfFailurebatch], axis=0)

        print(len(dfBatch[dfBatch['failure']]==1))

        X_batch = dfBatch[features].fillna(0)
        y_batch = dfBatch['failure']

        return X_batch,y_batch


    dfsampleSuccess = dfSuccess.sample(n = 200000-len(dfFailures))

    dfsampleFailure = dfFailures
    print(len(dfsampleFailure[dfsampleFailure.failure==1]))

    def SDG(dfsampleSuccess,dfsampleFailure,features,iter):
        SDGmodel = linear_model.SGDClassifier(max_iter=10, loss='log')
        for i in range(0, iter):
            x_batch, y_batch=batchSample(dfsampleSuccess,dfsampleFailure,features)
            #print(y_batch)
            clf = SDGmodel.partial_fit(x_batch, y_batch,classes=np.unique(y_batch))
        return clf

    print('Starting SGD...')
    print("Current Time =", datetime.now().strftime("%H:%M:%S"))
    clf = SDG(dfsampleSuccess,dfsampleFailure,simpleFeatures,20)


    prediction = clf.predict(Xtest)
    print(confusion_matrix(Ytest,prediction))
    print(classification_report(Ytest, prediction))

    print("Current Time =", datetime.now().strftime("%H:%M:%S"))

    def regressionUnderSampling(X, y):
        print('calculate under-sampling')

        cc = ClusterCentroids(random_state=0)
        X_resampled, y_resampled = cc.fit_resample(X, y)

        print('Results with under-sampling')


def readModel(modelFile):
    ds = pd.read_csv("data/diskStat3.csv").fillna(0)
    dfModels = pd.read_csv("data/modelList.csv")

    dsTest = ds[ds.capacity_bytes > 0]
    dsTest = pd.merge(left=dsTest, right=dfModels, how='left', left_on='model', right_on='model')

    simpleFeatures = ['modelType','capacity_bytes', 'smart_1_raw', 'smart_5_raw', 'smart_7_raw', 'smart_9_raw',
                      'smart_12_raw', 'smart_192_raw', 'smart_194_raw', 'smart_195_raw', 'smart_197_raw',
                      'smart_199_raw',
                      'smart_240_raw', 'smart_241_raw']
    # validate for another quarter
    Xtest = dsTest[simpleFeatures]
    Ytest = dsTest[['failure']]
    X1test = Xtest.fillna(0)





    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    predictions3 = loaded_model.predict(Xtest)
    probability = loaded_model.predict_proba(Xtest)
    print('results after over sampling, logistic regressioin, no device filtering')
    print(100*round(probability[0][1],2))
    print(classification_report(Ytest, predictions3))
    print(confusion_matrix(Ytest, predictions3))






if __name__ == '__main__':
    hddHealth()
    #readModel("data/modelList.csv")




