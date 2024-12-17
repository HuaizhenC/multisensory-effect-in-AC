import time
start_time = time.monotonic()
from datetime import timedelta
from eegUtilities import loadTimWindowedEEG,genLabelsComb,ITCest_resample,ITCstats,phasDist
import numpy as np
import pandas as pd
import os 
import math
import json

# Monkey = 'Elay'
# date = ['230301','230303','230306','230322','230324','230328','230329','230330','230404','230405','230410','230412']
# # date = ['221128','221129','221130','221205','221207','221208','221212','221213','221214','221226','221227','221228','221229','230102','230106','230109','230111'] # two channel sessions
# # Monkey = 'Wu'
# # date = ['230301','230302','230303','230306','230320','230321','230323','230327','230328','230329','230330','230403','230404','230405','230407','230410']
# filePathway = '/data/by-user/Huaizhen/'
# eegfilePathway = os.path.join(filePathway,'preprocNeuralMatfiles/')
# outputPathway = os.path.join(filePathway,'Fitresults/PhaseDist/')
# try:
#     os.mkdir(outputPathway)
# except FileExistsError:
#     pass

Monkey = 'Elay'
date = ['230330']
eegfilePathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/data/'
outputFolder = 'Fitresults/phaseDist/'
outputPathway = os.path.join('/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG',outputFolder)
try:
    os.mkdir(outputPathway)
except FileExistsError:
    pass


AlleegalignStr = ['align2chorus','align2js','align2coo']# 'align2coo' 'align2js'
AllchannelStr = ['align2chorus_medial1']  #'align2chorus_medial1' 'align2chorus_frontal3' 
timeRange = [-0.7,0.5]
freBand = {'none':[],'delta':[1,4],'theta':[4,8],\
            'alpha':[8,12],'beta':[12,25],'lowgamma':[25,58]}
seriesFeature = 'phase' # ['raw','power','phase','complex']
tfAnlyz = []

extraStr = '_phaseDist_hit'
filterCond = {'respLabel':[0]} # have multipe keys, each key can only has one value
catCond_comb = {'snr':list(np.arange(-10,15,5)),'trialMod':['a','av']} # several cat condition key, can have multiple values
# catCond_comb = {'trialMod':['v']}
# factor_levels = [len(list(catCond_comb.values())[0]), len(list(catCond_comb.values())[1])]  # number of levels in each factor
# effects = 'A*B'  # this is the default signature for computing all effects in ANOVA
# effect_labels = ['snr', 'mod', 'snrXmod']
# effects_all =['A','B','A:B'] 
# catNames = catCond[list(catCond.keys())[0]]

for indd,frekeys in enumerate(freBand):

    for eegalignStr in AlleegalignStr:
        for channelStr in AllchannelStr:
            jsonnameStr = Monkey+'_'+eegalignStr+'_'+channelStr+'_'+frekeys+'_'+seriesFeature+'_'+extraStr 
            print(jsonnameStr)

            for dd in date:
                print('cutting eeg in session: '+dd)
                eegSeg = loadTimWindowedEEG(eegfilePathway+Monkey+'_'+dd+'_preprocEEGtimeseries.mat',eegalignStr,channelStr,timeRange,freBand[frekeys],tfAnlyz,seriesFeature,norm='applyStandardScaler',baselineCorrect=True) # #1 X n_freqs X n_times  array
                Behavdata_df = genLabelsComb(eegfilePathway+Monkey+'_'+dd+'_labviewNsynapseBehavStruct.mat',filterCond,catCond_comb) # trials X cats dataframe

                try:
                    xx = np.append(xx,eegSeg,axis = 0)
                    yy = pd.concat([yy,Behavdata_df],ignore_index=True)
                except NameError:
                    xx = eegSeg.copy()  
                    yy = Behavdata_df.copy()  
            
            # plot phase distribution over time in each condition
            phasDist(xx,yy,timeRange,outputPathway,jsonnameStr,catCond_comb,bins=16,densityLogi=True)
 
           
            del xx, yy

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))








