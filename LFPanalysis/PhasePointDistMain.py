import time
start_time = time.monotonic()
from datetime import timedelta
from eegUtilities import loadTimWindowedEEG,phasPointDist,phasPointDist2,TimeseriesAlignbyRT,genLabelsDF
import numpy as np
import pandas as pd
import os 


# MonkeyDate_all = {'Elay':['230301','230303','230306','230322','230324','230328','230329','230330','230404','230405','230410','230412'],\
# 'Wu':['230301','230302','230303','230306','230320','230321','230323','230327','230328','230329','230330','230403','230404','230405','230407','230410']}
# snr_all = [np.arange(-10,15,5),np.arange(-5,15,5)]

# MonkeyDate_all = {'Wu':['230301','230302','230303','230306','230320','230321','230323','230327','230328','230329','230330','230403','230404','230405','230407','230410']}
# snr_all = [np.arange(-5,15,5)]

MonkeyDate_all = {'Elay':['230301','230303','230306','230322','230324','230328','230329','230330','230404','230405','230410','230412']}
snr_all = [np.arange(-10,15,5)]

filePathway = '/data/by-user/Huaizhen/'
eegfilePathway = os.path.join(filePathway,'preprocNeuralMatfiles/')
outputPathway = os.path.join(filePathway,'Fitresults/phasePointDist/')
try:
    os.mkdir(outputPathway)
except FileExistsError:
    pass

# MonkeyDate_all = {'Elay':['230330']}
# snr_all = [np.arange(-10,15,5)]
# eegfilePathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/data/'
# outputFolder = 'Fitresults/phaseDist/'
# outputPathway = os.path.join('/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG',outputFolder)
# try:
#     os.mkdir(outputPathway)
# except FileExistsError:
#     pass

AlleegalignStr = ['align2chorus','align2coo','align2js']# 'align2coo' 'align2js'
AllchannelStr = ['align2chorus_medial1']  #'align2chorus_medial1' 'align2chorus_frontal3' 
timeRange = [0,0]
# freBand = {'none':[],'delta':[1,4],'theta':[4,8],\
#             'alpha':[8,12],'beta':[12,25],'lowgamma':[25,58]}
freBand = {'none':[],'delta':[1,4],'theta':[4,8],'alpha':[8,12],'beta':[12,25]}

seriesFeature = 'phase' # ['raw','power','phase','complex']
tfAnlyz = []

for (Monkey,date),snr in zip(MonkeyDate_all.items(),snr_all):
    for indd,frekeys in enumerate(freBand):
        for eegalignStr in AlleegalignStr:
            for channelStr in AllchannelStr:               
                for dd in date:
                    print('cutting eeg in session: '+dd)
                    eegSeg = loadTimWindowedEEG(eegfilePathway+Monkey+'_'+dd+'_preprocEEGtimeseries.mat',eegalignStr,channelStr,timeRange,freBand[frekeys],tfAnlyz,seriesFeature,norm='applyStandardScaler',baselineCorrect=True) # #1 X n_freqs X n_times  array
                    Behavdata_df = genLabelsDF(eegfilePathway+Monkey+'_'+dd+'_labviewNsynapseBehavStruct.mat') # trials X cats dataframe

                    try:
                        xx = np.append(xx,eegSeg,axis = 0)
                        yy = pd.concat([yy,Behavdata_df],ignore_index=True)
                    except NameError:
                        xx = eegSeg.copy()  
                        yy = Behavdata_df.copy()  
                data = yy.copy()
                data['phase']=xx.copy() # 1 column 
                # filter out nan phase trials
                data = data[~data['phase'].isnull()]
                # plot phase distribution at time point of interest in each condition
                ##phasPointDist hit trials: a VS av X snr 
                jsonnameStr = Monkey+'_'+eegalignStr+'_'+frekeys+'_phaseDistPoint_aVSav_count'
                phasPointDist(data,outputPathway,jsonnameStr,{'respLabel':0},snr,bins=16)   
                # #phasPointDist2  stimon trials: hit VS FA X trialMod 
                jsonnameStr = Monkey+'_'+eegalignStr+'_'+frekeys+'_phaseDistPoint_hitVSfa_count'
                phasPointDist2(data,outputPathway,jsonnameStr,{'Stim':1},['a','av'],bins=16)           
        
                del xx, yy


end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))








