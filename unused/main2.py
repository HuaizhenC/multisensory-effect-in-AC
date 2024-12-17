import time
start_time = time.monotonic()
from datetime import timedelta

from eegUtilities import loadTimWindowedEEG,genLabels,BalanceSamples
import numpy as np
import math
from decoders import svmdecoder, applysvm
import json



# Monkey = 'Elay'
# date = ['221128','221129','221130','221205','221207','221208'] # two channel sessions
# # date = ['221109','221111','221114','221116','221121'] # one frontal channel sessions
# eegfilePathway = '/data/by-user/Huaizhen/'
# outputPathway = '/data/by-user/Huaizhen/Fitresults/'

Monkey = 'Wu'
date = ['221123']
eegfilePathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/EEGanalyze/'
outputPathway = './'

AlleegalignStr = ['align2js','align2coo']# 'align2coo' 'align2js'
AllchannelStr = ['align2coo_medial1']  #'align2coo_medial1' 'align2coo_frontal3' 

extraStr = '_Audhit'
filterCond = {'respLabel':[0],'trialMod':['a']} # have multipe keys, each key can only has one value
# catCond = {'trialMod':['a','v','av']} # only one cat condition key, can have multiple values
catCond = {'snr':[10,5,0,-5,-10]}
catNames = catCond[list(catCond.keys())[0]]

timeRange = [-0.7,0.5]
binTim = 0.1 #sec, time window for svm
freBand = {'none':[],'delta':[1,4],'theta':[4,8],\
            'alpha':[8,12],'beta':[12,25],'lowgamma':[25,58]}

for indd,frekeys in enumerate(freBand):
    for eegalignStr in AlleegalignStr:
        for channelStr in AllchannelStr:
            jsonnameStr = Monkey+'_'+eegalignStr+'_'+channelStr+'_'+frekeys+extraStr 
            xx = np.array([])
            yy = np.array([])
            for dd in date:
                print('cutting eeg in session: '+dd)
                eegSeg = loadTimWindowedEEG(eegfilePathway+Monkey+'_'+dd+'_preprocEEGtimeseries.mat',eegalignStr,channelStr,timeRange,freBand[frekeys],[]) # trials X features(time) array 
                Behavdata = genLabels(eegfilePathway+Monkey+'_'+dd+'_labviewNsynapseBehavStruct.mat',filterCond,catCond) # trials X 1 array
                #find valid trials index
                BehavalInd = [i for i,v in enumerate(Behavdata) if v==v] # trials fullfill behave filterCondition
                xx_temp = eegSeg[BehavalInd,:]
                yy_temp = Behavdata[BehavalInd]
                # dignity check 
                StimvalInd = [j for j,vec in enumerate(xx_temp) if (vec!=vec).any()] # filtered trials stimulus not played, should always empty if filterCond['respLabel'] is 0 or 1
                if len(StimvalInd) !=0:
                    print('nan remained in filtered eeg segments in' + Monkey + ' ' + date[dd])
                try:
                    xx = np.append(xx,xx_temp,axis = 0)
                    yy = np.append(yy,yy_temp,axis = 0)
                except:
                    xx = np.empty((0,xx_temp.shape[1]))
                    yy = np.empty((0,yy_temp.shape[1]))
                    xx = np.append(xx,xx_temp,axis = 0)
                    yy = np.append(yy,yy_temp,axis = 0)
            # apply svm on dataset
            applysvm(xx,yy,timeRange,binTim,catNames,outputPathway,jsonnameStr)


  
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))








