import time
start_time = time.monotonic()
from datetime import timedelta
from eegUtilities import loadTimWindowedEEG,TimeseriesAlignbyRT,genLabelsDF
import numpy as np
import pandas as pd
import os 
import math
import json



MonkeyDate_all = {'Elay':['230301','230303','230306','230322','230324','230328','230329','230330','230404','230405','230410','230412'],\
'Wu':['230301','230302','230303','230306','230320','230321','230323','230327','230328','230329','230330','230403','230404','230405','230407','230410']}

# MonkeyDate_all = {'Wu':['230301','230302','230303','230306','230320','230321','230323','230327','230328','230329','230330','230403','230404','230405','230407','230410']}
# MonkeyDate_all = {'Elay':['230301','230303','230306','230322','230324','230328','230329','230330','230404','230405','230410','230412']}

# MonkeyDate_all = {'Wu':['230302']}

filePathway = '/data/by-user/Huaizhen/'
eegfilePathway = os.path.join(filePathway,'preprocNeuralMatfiles/')
outputPathway = os.path.join(filePathway,'Fitresults/timeseriesAlign/')
try:
    os.mkdir(outputPathway)
except FileExistsError:
    pass

# MonkeyDate_all  = {'Elay':['230221']}
# eegfilePathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/data/'
# outputFolder = 'Fitresults/phaseDist/'
# outputPathway = os.path.join('/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG',outputFolder)
# try:
#     os.mkdir(outputPathway)
# except FileExistsError:
#     pass

AlleegalignStr = ['align2chorus','align2coo','align2js']# 'align2coo' 'align2js','align2chorus'
AllchannelStr = ['align2chorus_medial1']  #'align2chorus_medial1' 'align2chorus_frontal3' 
timeRange = [-0.7,0.2]
# freBand = {'none':[],'delta':[1,4],'theta':[4,8],\
#             'alpha':[8,12],'beta':[12,25]}#'lowgamma':[25,58]
freBand = {'none':[],'delta':[1,4],'theta':[4,8],\
            'alpha':[8,12],'beta':[12,25]}#'lowgamma':[25,58]

seriesFeatureAll = ['phase','raw']# ['raw','power','phase','phase_unwrap','complex']
tfAnlyz = []
normstr = [] #'applyStandardScaler'
extraStr = ''

for Monkey,date in MonkeyDate_all.items():
    for seriesFeature in seriesFeatureAll:
        for indd,frekeys in enumerate(freBand):
            for eegalignStr in AlleegalignStr:
                for channelStr in AllchannelStr:
                    jsonnameStr = Monkey+'_'+eegalignStr+'_'+frekeys+'_'+seriesFeature+extraStr 
                    print(jsonnameStr)

                    for dd in date:
                        print('cutting eeg in session: '+dd)
                        Behavdata_df = genLabelsDF(eegfilePathway+Monkey+'_'+dd+'_labviewNsynapseBehavStruct.mat') # trials X cats dataframe

                        eegSeg = loadTimWindowedEEG(eegfilePathway+Monkey+'_'+dd+'_preprocEEGtimeseries.mat',\
                                                    eegalignStr,channelStr,timeRange,freBand[frekeys],tfAnlyz,seriesFeature,\
                                                    norm=normstr,baselineCorrect=True) # #1 X n_freqs X n_times  array

                        try:
                            xx = np.append(xx,eegSeg,axis = 0)
                            yy = pd.concat([yy,Behavdata_df],ignore_index=True)
                        except NameError:
                            xx = eegSeg.copy()  
                            yy = Behavdata_df.copy() 
                    
                    # # plot phase distribution at time point of interest in each condition        
                    # TimeseriesAlignbyPhase(xx,yy,outputPathway,jsonnameStr,timeRange)
                            
                    # align phase series to time point of interest in each condition        
                    TimeseriesAlignbyRT(xx,yy,outputPathway,jsonnameStr,timeRange,seriesFeature)
                    
                    del xx, yy

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))








