import time
from datetime import timedelta
import numpy as np
import seaborn as sns
import mat73
import os
import pickle
import pandas as pd
from matplotlib import pyplot as plt 
from spikeUtilities import getClsRaster,countTrialSPKs,SortfilterDF,loadPreprocessMat,AVmodulation
# estimate Visual modulation, FR between Vidonset and Cooonset-100ms was used 
start_time = time.monotonic()

# MonkeyDate_all = {'Elay':['230420']} #
# Pathway='/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/data'
# ResPathway = os.getcwd()+'/'
# figSavePath = os.getcwd()+'/'

MonkeyDate_all = {'Elay':['230420','230503','230509','230525','230531','230602','230606','230608'],\
                  'Wu':['230508','230512','230515','230517','230522','230530','230601','230605','230607']} #

Pathway='/data/by-user/Huaizhen/preprocNeuralMatfiles/EEG+Vprobe/'
ResPathway = '/data/by-user/Huaizhen/Fitresults/AVmodIndex/'
figSavePath = '/data/by-user/Huaizhen/Figures/AVmodIndex/'

timwin = [-0.53,-0.1] # window for getcluster, when estimate spk between vidOnset to cooOnset-100ms

df_avMod_all = pd.DataFrame()
for Monkey,Date in MonkeyDate_all.items():
    for Date_temp in Date:  
        print('...............'+'estimating AV modulation in '+Monkey+'-'+Date_temp+'...............')
        neuroMetricDict_sess = {}
        neuroMetricDict_sess['Monkey'] = Monkey
        neuroMetricDict_sess['session'] = Date_temp                  

        spikeTimeDict,labelDictFilter, \
            timeSamp2Chorus,spikefs,behavefs= loadPreprocessMat(Monkey,Date_temp,Pathway)
      
        for keys in list(spikeTimeDict.keys()):
            if 'cls' in keys:
                if any(substr in keys for substr in ['mua','good']):
                # if any(substr in keys for substr in ['cls140_ch13_mua']):
                # if 'cls126_ch10' in keys:
                    print('cluster '+keys+' in progress............')
                    # # get trial by trial raster in each cluster
                    spikeTime_temp = spikeTimeDict[keys]
                    spikeTimedf_temp = getClsRaster(timeSamp2Chorus,spikeTime_temp,spikefs,behavefs,\
                                                        labelDictFilter['chorusOnsetInd'],\
                                                        labelDictFilter['cooOnsetIndwithVirtual'] ,\
                                                        labelDictFilter['joystickOnsetInd'],\
                                                        timwin,labelDictFilter)

                    # labelDict_sub = {}
                    # ntrials = 50
                    # for key,value in labelDictFilter.items():
                    #     labelDict_sub[key] = value[-ntrials:]
                    # spikeTime_temp = spikeTimeDict[keys][-ntrials:]
                    # spikeTimedf_temp = getClsRaster(timeSamp2Chorus,spikeTime_temp,spikefs,behavefs,\
                    #                                     labelDictFilter['chorusOnsetInd'][-ntrials:],labelDictFilter['cooOnsetIndwithVirtual'][-ntrials:],\
                    #                                     labelDictFilter['joystickOnsetInd'][-ntrials:],timwin,labelDict_sub)

                    spikeNumdf_temp=countTrialSPKs(spikeTimedf_temp,estwin='off',fs = behavefs)

                    ## drop trials no sound played at all, spkrate =NAN
                    spikeNumdf_temp.dropna(subset=['spkRate'],inplace=True)
                    ## filter out only hit or cr trials
                    spikeNumdf_temp = SortfilterDF(spikeNumdf_temp,filterlable = {'respLabel':[0,10,100]})
                    
                    ## estimate AV modulation X snrs
                    bsSamp = 600 #number of samples to pick for each condition in each Bootstrap operation
                    df_avMod_temp = AVmodulation(spikeNumdf_temp,bsSamp=bsSamp,cond = ['trialMod'])
                    df_avMod_temp['cluster'] = keys
                    df_avMod_temp['Monkey'] = Monkey
                    df_avMod_temp['session'] = Date_temp
                    df_avMod_all = pd.concat([df_avMod_all,df_avMod_temp])
              
        pickle.dump(df_avMod_all,open(ResPathway+'AVmodIndexDF_beforeCooAud.pkl','wb'))            

## plot 
df_avMod_all['avMod_sign'] = df_avMod_all['enhance/inhibit'].values*df_avMod_all['avMod'].values
datasig = df_avMod_all[df_avMod_all['avModSig']==1] 

fig, axess = plt.subplots(1,1,figsize=(8,10),sharex=True)
sns.stripplot(datasig.reset_index(),x='Monkey',y='avMod_sign',hue='session',dodge=True)
plt.tight_layout()
plt.savefig(figSavePath+'AVmodIndex_byMonkey_beforeCooAud.png')
plt.close()


end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))


