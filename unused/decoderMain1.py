import time
start_time = time.monotonic()
from datetime import timedelta
import os

from eegUtilities import loadTimWindowedEEG,genLabels,BalanceSamples
import numpy as np
import math
from decoders import svmdecoder, applysvm
import json

Monkey = 'Elay'
# date = ['221128','221129','221130','221205','221207','221208'] # two channel sessions
date = ['221128','221129','221130','221205','221207','221208','221212','221213','221214','221226','221227'] # two channel sessions
# date = ['221109','221111','221114','221116','221121'] # one frontal channel sessions
eegfilePathway = '/data/by-user/Huaizhen/'
outputFolder = 'Fitresults/decoderDimRed/'
outputPathway = os.path.join(eegfilePathway,outputFolder)
try:
    os.mkdir(outputPathway)
except FileExistsError:
    pass

# Monkey = 'Elay'
# date = ['221128']
# eegfilePathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/'
# outputFolder = 'Fitresults/decoderDimRed/'
# outputPathway = os.path.join(eegfilePathway,outputFolder)
# try:
#     os.mkdir(outputPathway)
# except FileExistsError:
#     pass

timeRange = [-0.7,0.5]
binTim = 0.1 #sec, time window for svm
freBand = {'none':[],'delta':[1,4],'theta':[4,8],\
            'alpha':[8,12],'beta':[12,25],'lowgamma':[25,58]}
# freBand = {'none':[]}
AllseriesFeature = ['raw'] # ['raw','power','phase']
tfAnlyz = []

AlleegalignStr = ['align2js','align2coo']# 'align2coo' 'align2js'
AllchannelStr = ['align2coo_medial1']  #'align2coo_medial1' 'align2coo_frontal3' 
# filterCond: has multipe keys ['respLabel','trialMod','snr'], each key can only has one value
# 'respLabel' [0,1] --> [hit, miss]
# filterCondall = [{'respLabel':[0],'trialMod':['a']},\
#                 {'respLabel':[0],'trialMod':['av']}]
# catCond = {'snr':list(np.arange(-10,15,5))} # catCond: only one cat condition key, can have multiple values               

filterCondall = [{'respLabel':[0],'snr':[-10]}]
catCond = {'trialMod':['a','av']} # catCond: only one cat condition key, can have multiple values               

# filterCondall = [{'trialMod':['a'],'Stim':[1]},\
#                  {'trialMod':['av'],'Stim':[1]}]
# catCond = {'respLabel':[0,2]} # catCond: only one cat condition key, can have multiple values               

catNames_int = catCond[list(catCond.keys())[0]]
catNames = [str(i) for i in catNames_int]
print('catCond ')
print(catCond)

# DRmethod = string, dimensionality reduction method to use; options include:
#             "tsne", "mds", "lle", "mlle", "isomap", "umap", "pca"
# method = {'method':'bootstrapDR','DRmethod':'umap','B':10,'num_jobs':8,'colorstr':colorstr}
method = {'method':'DR_pca'}

for seriesFeature in AllseriesFeature:
    for filterCond in filterCondall:
        extraStr = '_respLabel'+str(filterCond['respLabel'][0])+'_PCA'+seriesFeature+'decode_sensoryMod'

        # extraStr = '_'+filterCond['trialMod'][0]+'_PCA'+seriesFeature+'_hit_'+'decode_snr'

        for indd,frekeys in enumerate(freBand):
            for eegalignStr in AlleegalignStr:
                for channelStr in AllchannelStr:
                    jsonnameStr = Monkey+'_'+eegalignStr+'_'+channelStr+'_'+frekeys+extraStr 
                    print(jsonnameStr)
                    for dd in date:
                        print('\ncutting eeg in session: '+dd)
                        eegSeg = loadTimWindowedEEG(eegfilePathway+Monkey+'_'+dd+'_preprocEEGtimeseries.mat',eegalignStr,channelStr,timeRange,freBand[frekeys],tfAnlyz,seriesFeature,norm='applyStandardScaler',baselineCorrect=True) # trials X features(time) array 
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
                            xx = xx_temp.copy() 
                            yy = yy_temp.copy()  
                                            
                    # apply svm on dataset
                    print('START SVM TRAINING................')
                    applysvm(xx,yy,timeRange,binTim,catNames,outputPathway,jsonnameStr,method,seriesFeature,singleSess= False)
                    print('SVM TRAINING DONE.................')
                    del xx, yy
    
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))








