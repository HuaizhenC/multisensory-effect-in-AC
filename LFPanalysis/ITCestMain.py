import time
start_time = time.monotonic()
from datetime import timedelta
from eegUtilities import loadTimWindowedEEG,genLabelsComb,genLabels,ITCest_resample,ITCstats,ITCest
import numpy as np
import pandas as pd
import os 

import math
import json

Monkey = 'Elay'
# date = ['221128','221129','221130','221205','221207','221208'] # two channel sessions
date = ['221128','221129','221130','221205','221207','221208','221212','221213','221214','221226','221227','221228','221229','230102','230106','230109','230111'] # two channel sessions
# date = ['221109','221111','221114','221116','221121'] # one frontal channel sessions
eegfilePathway = '/data/by-user/Huaizhen/'
outputFolder = 'Fitresults/ITCest/'
outputPathway = os.path.join(eegfilePathway,outputFolder)
try:
    os.mkdir(outputPathway)
except FileExistsError:
    pass

# Monkey = 'Elay'
# date = ['221128']
# eegfilePathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/'
# outputFolder = 'Fitresults/ITCest/'
# outputPathway = os.path.join(eegfilePathway,outputFolder)
# try:
#     os.mkdir(outputPathway)
# except FileExistsError:
#     pass

n_jobs = 6
AlleegalignStr = ['align2js','align2coo']# 'align2coo' 'align2js'
AllchannelStr = ['align2coo_medial1' ]  #'align2coo_medial1' 'align2coo_frontal3' 
timeRange = [-0.7,0.5]
tfAnlyz = {'tf_phase':'phase'}
freBand = []
seriesFeature = []

extraStr = '_ITC_hit'
# filterCond = {'respLabel':[0]} # have multipe keys, each key can only has one value
# catCond_comb = {'snr':list(np.arange(-15,15,5)),'trialMod':['a','av']} # several cat condition key, can have multiple values
# factor_levels = [len(list(catCond_comb.values())[0]), len(list(catCond_comb.values())[1])]  # number of levels in each factor
# effects = 'A*B'  # this is the default signature for computing all effects in ANOVA
# effect_labels = ['snr', 'mod', 'snrXmod']
# effects_all =['A','B','A:B'] 

filterCondall = [{'respLabel':[0],'snr':[10]},\
                {'respLabel':[0],'snr':[5]},\
                    {'respLabel':[0],'snr':[0]},\
                        {'respLabel':[0],'snr':[-5]},\
                            {'respLabel':[0],'snr':[-10]},\
                                {'respLabel':[0],'snr':[-15]}]
catCond = {'trialMod':['a','av']} # catCond: only one cat condition key, can have multiple values               

catNames = catCond[list(catCond.keys())[0]]

for filterCond in filterCondall:
    for indd2,tfkeys in enumerate(tfAnlyz):
        for eegalignStr in AlleegalignStr:
            for channelStr in AllchannelStr:
                jsonnameStr = Monkey+'_'+eegalignStr+'_'+channelStr+'_'+tfkeys+'_snr'+str(filterCond['snr'][0])+extraStr 
                print(jsonnameStr)

                for dd in date:
                    print('cutting eeg in session: '+dd)
                    eegSeg = loadTimWindowedEEG(eegfilePathway+Monkey+'_'+dd+'_preprocEEGtimeseries.mat',eegalignStr,channelStr,timeRange,freBand,tfAnlyz[tfkeys],seriesFeature,norm='applyStandardScaler',baselineCorrect=True) # #1 X n_freqs X n_times  array
                    # Behavdata_df = genLabelsComb(eegfilePathway+Monkey+'_'+dd+'_labviewNsynapseBehavStruct.mat',filterCond,catCond_comb) # trials X cats dataframe
                    Behavdata_df = genLabels(eegfilePathway+Monkey+'_'+dd+'_labviewNsynapseBehavStruct.mat',filterCond,catCond)
                    xx_temp = eegSeg 
                    yy_temp = Behavdata_df

                    try:
                        xx = np.append(xx,xx_temp,axis = 0)
                        # yy = pd.concat([yy,yy_temp])
                        yy = np.append(yy,yy_temp,axis=0)
                    except NameError:
                        xx = xx_temp  
                        yy = yy_temp  
                
                # # calculate ITC in each condition
                # totalTrials = len(yy)
                # validTrialsMin = min(totalTrials-yy.isnull().sum())
                # print('validTrialsMin'+str(validTrialsMin))
                # subsamNum = 100 # number of subsample trials needed 
                # subsamSize = 30 # number of trials to generate one ITC
                
                # n_permutations = 1000
                # # xx_bootstr (trials X conditions X frequency X time)
                # xx_bootstr = ITCest_resample(xx,yy,list(tfAnlyz.values())[0],subsamSize,subsamNum)
                # ITCstats(xx_bootstr,timeRange,outputPathway,jsonnameStr,factor_levels,effects,effects_all,effect_labels,n_jobs,n_permutations)
                ITCest(xx,yy,catNames,outputPathway,timeRange,jsonnameStr)
                
                del xx, yy

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))








