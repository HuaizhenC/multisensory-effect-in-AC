import time
from datetime import timedelta
import numpy as np
import seaborn as sns
import mat73
import os
import pickle
import pandas as pd
from matplotlib import pyplot as plt 
from spikeUtilities import getClsRasterMultiprocess,countTrialSPKs,loadPreprocessMat
start_time = time.monotonic()

MonkeyDate_all = {'Elay':['230420']} #

Pathway='/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/data'
ResPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/noisecorrelation/'

# MonkeyDate_all = {'Elay':['230420','230503','230509','230525','230531','230602','230606','230608'],\
#                   'Wu':['230508','230512','230515','230517','230522','230530','230601','230605','230607']} #

# Pathway='/data/by-user/Huaizhen/preprocNeuralMatfiles/EEG+Vprobe/'
# ResPathway = '/data/by-user/Huaizhen/Fitresults/noisecorrelation/'

# extrastr_all = ['align2coo','align2js','align2chorus']
# alignkeys_all = ['cooOnsetIndwithVirtual','joystickOnsetInd','chorusOnsetInd']
# NaneventsIndtarg_all = ['[]','labelDictFilter["cooOnsetIndwithVirtual"]','[]']
# timwinStart = np.arange(-1,1,0.2) # 
# winlen = 0.3
extrastr_all = ['align2coo']
alignkeys_all = ['cooOnsetIndwithVirtual']
NaneventsIndtarg_all = ['[]']
timwinStart = np.arange(-1,1,0.2) # 
winlen = 0.3

for extrastr,alignkeys,NaneventsIndtarg in zip(extrastr_all,alignkeys_all,NaneventsIndtarg_all):
    for Monkey,Date in MonkeyDate_all.items():
        for Date_temp in Date:  
            AllSinUnits_dict = {}
            print('...............'+'estimating spk for noisecorr in '+Monkey+'-'+Date_temp+'...............')

            spikeTimeDict,labelDictFilter, \
                timeSamp2Chorus,spikefs,behavefs= loadPreprocessMat(Monkey,Date_temp,Pathway)

            # save sessions have more than 2 clusters
            if len([substr for substr in list(spikeTimeDict.keys()) if any(d in substr for d in ['good'])])>1:
                for cls in list(spikeTimeDict.keys()):
                    if any(substr in cls for substr in ['good']):                    
                        print('cluster '+cls+' in progress............')
                        # get trial by trial raster in each cluster
                        labelDict_sub = {}
                        ntrials = 50
                        for key,value in labelDictFilter.items():
                            labelDict_sub[key] = value[-ntrials:]
                        labelDictFilter = labelDict_sub.copy()
                        spikeTime_temp = spikeTimeDict[cls][-ntrials:]   
                                            
                        # spikeTime_temp = spikeTimeDict[cls]

                        spikeTimedf_temp = getClsRasterMultiprocess(timeSamp2Chorus,spikeTime_temp,spikefs,behavefs,\
                                                            labelDictFilter['chorusOnsetInd'],\
                                                            labelDictFilter[alignkeys] ,\
                                                            labelDictFilter['joystickOnsetInd'],\
                                                            [timwinStart[0],timwinStart[-1]+winlen],\
                                                            labelDictFilter,eval(NaneventsIndtarg))

                        for tt, tStart in enumerate(timwinStart):
                            spikeNumdf_temp=countTrialSPKs(spikeTimedf_temp,estwin='subwinoff',fs = behavefs,winTim=[tStart,tStart+winlen])
                            if tt==0:
                                spikeNumdf = spikeNumdf_temp.iloc[:,:-1].copy()
                            spikeNumdf['spkT'+str(np.round(tStart,decimals=1))] = spikeNumdf_temp['spkRate']                        
                        AllSinUnits_dict[cls] = spikeNumdf                
                        pickle.dump(AllSinUnits_dict,open(ResPathway+Monkey+'_'+Date_temp+'_'+extrastr+'_spkinwin.pkl','wb'))            
                        print('done with cluster '+cls)
                        end_time = time.monotonic()
                        print(timedelta(seconds=end_time - start_time))

print('done')
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))

