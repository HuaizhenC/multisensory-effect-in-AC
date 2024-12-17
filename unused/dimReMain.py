import time
start_time = time.monotonic()
from datetime import timedelta

from eegUtilities import loadTimWindowedEEG,genLabels,BalanceSamples
import numpy as np
import pandas as pd
import math
from decoders import svmdecoder, applysvm
import json

Monkey = 'Elay'
date = ['221128','221129','221130','221205','221207','221208','221212','221213','221214','221226','221227'] # two channel sessions
# date = ['221109','221111','221114','221116','221121'] # one frontal channel sessions
eegfilePathway = '/data/by-user/Huaizhen/'
outputPathway = '/data/by-user/Huaizhen/Fitresults/'

# Monkey = 'Wu'
# date = ['1']
# eegfilePathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/EEGanalyze/'
# outputPathway = './'

AlleegalignStr = ['align2js']# 'align2coo' 'align2js'
AllchannelStr = ['align2coo_medial1' ]  #'align2coo_medial1' 'align2coo_frontal3' 

extraStr = '_hit_bootstrapeDR'
filterCond = {'respLabel':[0]} # have multipe keys, each key can only has one value
catCond = {'trialMod':['a','av']} # only one cat condition key, can have multiple values
# catCond = {'snr':[10,5,0,-5,-10]}
catNames = catCond[list(catCond.keys())[0]]
colorstr = ['m','c']

timeRange = [-0.7,0.5]
binTim = 0.1 #sec, time window for svm

freBand = {'none':[],'delta':[1,4],'theta':[4,8],\
            'alpha':[8,12],'beta':[12,25],'lowgamma':[25,58]}

# DRmethod = string, dimensionality reduction method to use; options include:
#             "tsne", "mds", "lle", "mlle", "isomap", "umap", "pca"
method = {'method':'bootstrapDR','DRmethod':'umap','B':10,'num_jobs':8,'colorstr':colorstr}

for indd,frekeys in enumerate(freBand):
    for eegalignStr in AlleegalignStr:
        for channelStr in AllchannelStr:
            jsonnameStr = Monkey+'_'+eegalignStr+'_'+channelStr+'_'+frekeys+extraStr 
            print(jsonnameStr)
            for dd in date:
                print('\ncutting eeg in session: '+dd)
                eegSeg = loadTimWindowedEEG(eegfilePathway+Monkey+'_'+dd+'_preprocEEGtimeseries.mat',eegalignStr,channelStr,timeRange,freBand[frekeys],[]) # trials X features(time) array 
                Behavdata = genLabels(eegfilePathway+Monkey+'_'+dd+'_labviewNsynapseBehavStruct.mat',filterCond,catCond) # trials X 1 array
                #find valid trials index
                BehavalInd = [i for i,v in enumerate(Behavdata) if v==v] # trials fullfill behave filterCondition
                xx_temp = eegSeg[BehavalInd,:]
                yy_temp = Behavdata[BehavalInd]
                # dignity check: filtered trials stimulus not played, should always empty if filterCond['respLabel'] is 0 or 1
                StimvalInd = [j for j,vec in enumerate(xx_temp) if (vec!=vec).any()] 
                if len(StimvalInd) !=0:
                    print('nan remained in filtered eeg segments in ' + Monkey + ' ' + dd)
                    xx_temp = np.delete(xx_temp,StimvalInd,axis=0)
                    yy_temp = np.delete(yy_temp,StimvalInd,axis=0)

                try:
                    xx = np.append(xx,xx_temp,axis = 0)
                    yy = np.append(yy,yy_temp,axis = 0)
                except NameError:
                    xx = xx_temp  
                    yy = yy_temp  
                                        
            # apply svm on dataset
            print('START SVM TRAINING................')
            applysvm(xx,yy,timeRange,binTim,catNames,outputPathway,jsonnameStr,method)
            print('SVM TRAINING DONE.................')
            del xx, yy



end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
