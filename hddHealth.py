
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
from sklearn import svm
import seaborn



def hddHealth():
    path = "data/"

    #Get
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


    yearArr = [2019]
    qtrArr = [1,2]

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


        dfinal.to_hdf("/Users/sveta/python/InsightDataScience/data/hdd_"+str(qtr)+"_"+str(year)+".h5", key='dfinal', mode='w')

        print("finished loading " +str(year) + ", "+ str(qtr))
        #return dfinal

    def loadingQtrData(yearArr,qtrArr):
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

    print('start loading data')

    loadingQtrData()

    print('loading done')
    input()

    XYcols = ['modelType','serial_number','capacity_bytes','failure','smart_1_raw', 'smart_5_raw', 'smart_7_raw','smart_9_raw', 'smart_12_raw','smart_192_raw', 'smart_194_raw', 'smart_195_raw','smart_197_raw', 'smart_199_raw', 'smart_240_raw','smart_241_raw']

    featuresArr = ['modelType','capacity_bytes','smart_1_raw', 'smart_5_raw', 'smart_7_raw','smart_9_raw', 'smart_12_raw','smart_192_raw', 'smart_194_raw', 'smart_195_raw','smart_197_raw', 'smart_199_raw', 'smart_240_raw','smart_241_raw']

    SGDFeatures = [ 'smart_1_raw', 'smart_5_raw', 'smart_9_raw','smart_192_raw', 'smart_194_raw',
                   'smart_240_raw']


    print('Read preprocessed data')





    df_updated = pd.read_hdf("data/df_Updated.h5")
    dfModels = pd.read_csv("data/modelList.csv")

    features1 = ['capacity_bytes']
    arrCol = df_updated.columns
    def getRawCols(arrCol):
        for item in arrCol:
            if '_raw' in item:
                features1.append(item)
        return features1



    #simpleFeatures = SGDFeatures
    simpleFeatures = getRawCols(arrCol)
    #print(df_updated.columns)


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


    #print(len(dfFailures),len(dfSuccess))

    print("Current Time =", datetime.now().strftime("%H:%M:%S"))


    #EDA(df_updated,dfFailures,dfSuccess)


    #read serial number of devices which never failed  - and exclude them
    df_neverfailed = pd.read_csv("data/serNumberNeverFailed.csv")

    serNumberNeverFailed = df_neverfailed[df_neverfailed.failure>=50]

    start_mem = df_updated.memory_usage().sum()/ 1024**2
    print('current memory usage: ',start_mem)


    print("whole ds ",len(df_updated))
    ds = df_updated[~df_updated.serial_number.isin(list(serNumberNeverFailed.serial_number))]

    #ds = df_updated
    dfFailures = ds[ds.failure==1]
    dfSuccess = ds[ds.failure==0]

    start_mem = df_updated.memory_usage().sum() / 1024 ** 2
    print('current memory usage: ', start_mem)


    #simpleFeatures = ['capacity_bytes','smart_1_raw', 'smart_5_raw', 'smart_7_raw','smart_9_raw', 'smart_12_raw','smart_192_raw', 'smart_194_raw', 'smart_195_raw','smart_197_raw', 'smart_199_raw', 'smart_240_raw','smart_241_raw']
    #simpleFeatures = ['capacity_bytes','smart_5_raw', 'smart_7_raw','smart_9_raw', 'smart_12_raw','smart_173_raw','smart_192_raw', 'smart_194_raw','smart_240_raw']
    #                 [0.02656122     0.03499287     0.18742727     0.04481702    0.43668853     0.00379877      0.00102822       0.14376292       0.12092319]
    #simpleFeatures = ['modelType','capacity_bytes','smart_5_raw','smart_9_raw','smart_240_raw']

    simpleFeatures = ['capacity_bytes', 'modelType', 'smart_1_raw', 'smart_5_raw', 'smart_7_raw', 'smart_9_raw',
                      'smart_12_raw', 'smart_188_raw', 'smart_189_raw', 'smart_190_raw', 'smart_192_raw',
                      'smart_194_raw', 'smart_195_raw', 'smart_197_raw', 'smart_199_raw', 'smart_200_raw',
                      'smart_240_raw', 'smart_241_raw']



    #ds = df_updated

    del (df_updated)
    gc.collect()


    print("after exclusion",len(ds))



    #print(features)
    #

    dstrain = ds[ds.capacity_bytes>0]

    del (ds)
    gc.collect()

    #print(len(dstrain[dstrain.failure==0]),len(dstrain[dstrain.failure==1]),len(dstrain) )

    print(len(dstrain))
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




    print('Handling imbalanced data - starting SMOTE ...')


    def overSamplingWithSMOTE(X_train,y_train0):
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
#        clf2 = LogisticRegression(penalty = 'l2', solver = 'saga',tol = 1e-6).fit(X_train_res, y_train_res)
        clf2 = LogisticRegression(class_weight='balanced',  C=0.5).fit(X_train_res, y_train_res)

        #(penalty = 'l1', solver = 'liblinear',tol = 1e-6, max_iter = int(1e6),warm_start = True,intercept_scaling = 10000.)

        #predictions2 = clf2.predict(Xtest)

        filename = 'finalized_model.sav'
        pickle.dump(clf2, open(filename, 'wb'))
        return clf2


    #clf2 = logisticRegression(X_train_res, y_train_res, Xtest, Ytest)

    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    predictions3 = loaded_model.predict(Xtest)

    print (Xtest.head())




    #predictions3 = clf2.predict(Xtest)
    print('results after over sampling, logistic regressioin, no device filtering')

    print(classification_report(Ytest, predictions3))
    print(confusion_matrix(Ytest, predictions3))

    def randomForest(X_train_res, y_train_res):
        #Random forests
        print('starting random forest')
        clfRF = RandomForestClassifier(max_depth=2, random_state=0).fit(X_train_res, y_train_res)
        print('feature impoartance for RF')
        print(clfRF.feature_importances_)
        pd.DataFrame(clfRF.feature_importances_).to_csv("data/featuesSelection.csv")
        return clfRF


    clfRF = randomForest(X_train_res, y_train_res)
    predictionRF = clfRF.predict(Xtest)
    print(classification_report(Ytest, predictionRF))
    print(confusion_matrix(Ytest, predictionRF))


    #for i in range
    #prediction_sgd = clfSDG.partial_fit(X_batch, y_batch)

    def runSVM(X_train_res, y_train_res):
        clf = svm.SVC().fit(X_train_res, y_train_res)
        return clf


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

        #print(dfBatch[['failure']])


        X_batch = dfBatch[features].fillna(0)
        y_batch = dfBatch['failure']


        return X_batch,y_batch


    dfsampleSuccess = dfSuccess.sample(n = 100000-len(dfFailures))
    #dfsampleSuccess = dfSuccess

    dfsampleFailure = dfFailures
    #print(len(dfsampleFailure[dfsampleFailure.failure==1]))



    def SDG(dfsampleSuccess,dfsampleFailure,features,iter):
        print(features)
        SDGmodel = linear_model.SGDClassifier(max_iter=1000, tol=1e-6, loss='log')
        for i in range(0, iter):
            x_batch, y_batch=batchSample(dfsampleSuccess,dfsampleFailure,features)
            #print(y_batch)
            clf = SDGmodel.partial_fit(x_batch, y_batch,classes=np.unique(y_batch))
        return clf

    print('Starting SGD...')
    print("Current Time =", datetime.now().strftime("%H:%M:%S"))

    clf = SDG(dfsampleSuccess,dfsampleFailure,SGDFeatures,20)




    prediction = clf.predict(Xtest)
    print(confusion_matrix(Ytest,prediction))
    print(classification_report(Ytest, prediction, zero_division=1))

    print("Current Time =", datetime.now().strftime("%H:%M:%S"))

    def regressionUnderSampling(X, y):
        print('calculate under-sampling')

        cc = ClusterCentroids(random_state=0)
        X_resampled, y_resampled = cc.fit_resample(X, y)

        print('Results with under-sampling')


def readModel():

    #read user test file
    ds = pd.read_csv("data/diskStat_dc.csv").fillna(0)

    ds = ds[ds.capacity_bytes > 0]

    simpleFeatures = ['capacity_bytes', 'smart_1_raw', 'smart_5_raw', 'smart_7_raw', 'smart_9_raw', 'smart_12_raw',
                      'smart_188_raw', 'smart_189_raw', 'smart_190_raw', 'smart_192_raw', 'smart_194_raw',
                      'smart_195_raw', 'smart_197_raw', 'smart_199_raw',
                      'smart_200_raw', 'smart_240_raw', 'smart_241_raw']

    # validate for user data
    Xtest = ds[simpleFeatures]
    Ytest = ds[['failure']]
    X1test = Xtest.fillna(0)
    info = ds[['date', 'model', 'serial_number', 'capacity_bytes']]

    filename_RF = 'finalized_model_RF1.sav'
    filename_LR = 'finalized_model.sav'

    loaded_model = pickle.load(open(filename_RF, 'rb'))

    predictions = loaded_model.predict(X1test)
    probability = loaded_model.predict_proba(X1test)

    user_prob = pd.DataFrame(probability).rename(columns={1: 'probability'})
    result = pd.merge(Ytest, user_prob[['probability']], right_index=True, left_index=True)


    predRF = pd.DataFrame(predictions).rename(columns={0: 'predictions'})
    result = pd.merge(result, predRF, right_index=True, left_index=True)

    result = pd.merge(result, info[['model', 'serial_number', 'capacity_bytes']], right_index=True,
                      left_index=True).sort_values(by='probability', ascending=False)
    # result.to_csv("~/data/dashboardResults.csv")

    serialNumbersOfFailing = list(result[result.predictions == 1]['serial_number'])


    disksLikelyToFail = result[result.predictions == 1].head()
    print(disksLikelyToFail)

    #print(serialNumbersOfFailing[0])
    #get user data for the device with highest probability of failure
    dfUser = ds[ds.serial_number == serialNumbersOfFailing[0]]

    topFeatures = pd.read_csv("data/top5Features.csv").head()[['name', 'fullName']]
    topFeaturesArr = topFeatures[['name']]
    topFeaturesList = list(topFeaturesArr['name'])
    topFeatures['name_y'] = topFeatures['name']

    dfFailStat = pd.read_csv("data/failStat.csv")
    dfSuccessStat = pd.read_csv("data/successStat.csv")
    # dfFail = dfFailStat[[topFeaturesArr]]

    # print(dfFailStat)
    # print(topFeatures)

    dfmean = pd.read_csv("data/allmean.csv")

    dfmeanTopFeatures = dfmean[dfmean.name.isin(topFeaturesList)]
    dfmeanUser = pd.DataFrame(dfUser[simpleFeatures].mean()).rename(columns={0: 'userData'})
    dfmeanUser['name_x'] = dfmeanUser.index
    dfmeanUserTopFeatures = dfmeanUser[dfmeanUser.name_x.isin(topFeaturesList)]

    dfmeanTopFeatures = pd.merge(left=dfmeanTopFeatures, right=dfmeanUserTopFeatures, how='left', right_on='name_x',
                                 left_on='name')


    dfmeanTopFeatures = pd.merge(left=dfmeanTopFeatures, right=topFeatures, how='left', right_on='name_y',
                                 left_on='name_x')



    dfmeanTopFeaturesHours = dfmeanTopFeatures[dfmeanTopFeatures.fullName == 'Power-On Hours in days']
    dfmeanTopFeaturesNotHours = dfmeanTopFeatures[dfmeanTopFeatures.fullName != 'Power-On Hours in days']

    dfmeanTopFeaturesHours['Fail'] = [x / 24 for x in dfmeanTopFeaturesHours['Fail']]
    dfmeanTopFeaturesHours['nonFail'] = [x / 24 for x in dfmeanTopFeaturesHours['nonFail']]
    dfmeanTopFeaturesHours['userData'] = [x / 24 for x in dfmeanTopFeaturesHours['userData']]

    dfmeanTopFeatures = pd.concat([dfmeanTopFeaturesHours, dfmeanTopFeaturesNotHours], axis=0)

    print (dfmeanTopFeatures)
    # graph to display
    print(dfmeanTopFeatures[['fullName', 'Fail', 'nonFail', 'userData']])
    # print(dfmeanUser)

    # print('results after over sampling, logistic regressioin, no device filtering')
    # print(probability)
    # print(classification_report(Ytest, predictions3))
    # print(confusion_matrix(Ytest, predictions3))

    #print(round(100 * round(probability[0][1], 2)))
    # return round(100*round(probability[0][1],2))






if __name__ == '__main__':
    #hddHealth()
    readModel()




