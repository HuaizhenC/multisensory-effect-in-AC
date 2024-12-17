import time
start_time = time.monotonic()
from datetime import timedelta
from eegUtilities import loadTimWindowedEEG,genLabelsComb,ITCest_resample,ITCstats
import numpy as np
import pandas as pd

import math
import json

Monkey = 'Elay'
date = ['221128','221129','221130'] # two channel sessions
# date = ['221109','221111','221114','221116','221121'] # one frontal channel sessions
eegfilePathway = '/data/by-user/Huaizhen/'
outputPathway = '/data/by-user/Huaizhen/Fitresults/'

# Monkey = 'Wu'
# date = ['1']
# eegfilePathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/'
# outputPathway = './'

n_jobs = 6
AlleegalignStr = ['align2js','align2coo']# 'align2coo' 'align2js'
AllchannelStr = ['align2coo_medial1' ]  #'align2coo_medial1' 'align2coo_frontal3' 
timeRange = [-0.7,0.5]
tfAnlyz = {'tf_phase':'power'}
freBand = []
seriesFeature = []

extraStr = '_power_hit_3sessions'
filterCond = {'respLabel':[0]} # have multipe keys, each key can only has one value
catCond_comb = {'snr':list(np.arange(-10,15,5)),'trialMod':['a','av']} # several cat condition key, can have multiple values
factor_levels = [len(list(catCond_comb.values())[0]), len(list(catCond_comb.values())[1])]  # number of levels in each factor
effects = 'A*B'  # this is the default signature for computing all effects in ANOVA
effect_labels = ['snr', 'mod', 'snrXmod']
effects_all =['A','B','A:B'] 
# catNames = catCond[list(catCond.keys())[0]]


for indd2,tfkeys in enumerate(tfAnlyz):
    for eegalignStr in AlleegalignStr:
        for channelStr in AllchannelStr:
            jsonnameStr = Monkey+'_'+eegalignStr+'_'+channelStr+'_'+tfkeys+extraStr 
            print(jsonnameStr)

            for dd in date:
                print('cutting eeg in session: '+dd)
                eegSeg = loadTimWindowedEEG(eegfilePathway+Monkey+'_'+dd+'_preprocEEGtimeseries.mat',eegalignStr,channelStr,timeRange,freBand,tfAnlyz[tfkeys],seriesFeature,norm='applyStandardScaler',baselineCorrect=True) # #1 X n_freqs X n_times  array
                Behavdata_df = genLabelsComb(eegfilePathway+Monkey+'_'+dd+'_labviewNsynapseBehavStruct.mat',filterCond,catCond_comb) # trials X cats dataframe

                xx_temp = eegSeg 
                yy_temp = Behavdata_df

                try:
                    xx = np.append(xx,xx_temp,axis = 0)
                    yy = pd.concat([yy,yy_temp])
                except NameError:
                    xx = xx_temp  
                    yy = yy_temp  
            
            # calculate ITC in each condition
            totalTrials = len(yy)
            validTrialsMin = min(totalTrials-yy.isnull().sum())
            print('validTrialsMin '+str(validTrialsMin))
            subsamSize = validTrialsMin # number of trials to balance across categories
            
            n_permutations = 1000
            # xx_bootstr (trials X conditions X frequency X time)
            xx_bootstr = ITCest_resample(xx,yy,list(tfAnlyz.values())[0],subsamSize,[])
            ITCstats(xx_bootstr,timeRange,outputPathway,jsonnameStr,factor_levels,effects,effects_all,effect_labels,n_jobs,n_permutations)
            
            del xx, yy

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))








