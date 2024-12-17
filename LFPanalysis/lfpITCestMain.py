import time
start_time = time.monotonic()
from datetime import timedelta
from eegUtilities import loadTimWindowedLFP,genLabelsComb,lfpITCest
from spikeUtilities import loadBehavMat,SortfilterDF,decodrespLabel2str
import numpy as np
import pandas as pd
import pickle


MonkeyDate_all = {'Elay':['230420']} #

Pathway='/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/data/'
ResPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/glmfit/'
figSavePath = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Figures/glmfit/'

# MonkeyDate_all = {'Elay':['230420','230503','230509','230525','230531','230602','230606','230608'],\
#                   'Wu':['230508','230512','230515','230517','230522','230530','230601','230605','230607']} #

# Pathway='/data/by-user/Huaizhen/preprocNeuralMatfiles/EEG+Vprobe/'
# ResPathway = '/data/by-user/Huaizhen/Fitresults/glmfit/'
# figSavePath = '/data/by-user/Huaizhen/Figures/glmfit/'


n_jobs = 6
timeRange = [-0.7,0.5]

# tfAnlyz = {'tf_phase':'power'}
# freBand = []
# seriesFeature = []
tfAnlyz = []
freBand = {'delta':[1,4],'theta':[4,8],\
            'alpha':[8,12],'beta':[12,25]}#'none':[],'lowgamma':[25,58]
seriesFeature = 'phase' # ['raw','power','phase','complex']

AlleegalignStr = ['align2cho','align2js','align2coo']
alignkeys_all = ['chorusOnsetInd','joystickOnsetInd','cooOnsetIndwithVirtual',]
filterCond = {'trialMod':['a','av','v'],'respLabel':['hit','miss','CR','FA']} # have multipe keys, each key can only has one value
labellist = ['trialMod','respLabel','snr']


# for indd2,tfkeys in enumerate(tfAnlyz):
for alignkeys in alignkeys_all:                
    for Monkey,Date in MonkeyDate_all.items():
        for dd in Date:
                xx_date = {}
                print('cutting lfp in session: '+dd)
                Behavdata_dict,behavefs = loadBehavMat(Monkey,dd,Pathway)# trials X cats dataframe
                Behavdata_df = decodrespLabel2str(pd.DataFrame.from_dict(Behavdata_dict)).copy()
                LFPfilePathway = Pathway+Monkey+'_'+dd+'_preprocLFPtimeseries.h5'
            for indd,frekeys in enumerate(freBand):    
                # LFPSeg = loadTimWindowedLFP(LFPfilePathway,Behavdata_dict,behavefs,alignkeys,timeRange,[],tfAnlyz[tfkeys],seriesFeature,baselineCorrect=True) #trial X chan X freq X time
                LFPSeg = loadTimWindowedLFP(LFPfilePathway,Behavdata_dict,behavefs,alignkeys,timeRange,freBand[frekeys],[],seriesFeature,baselineCorrect=True) # #trial X chan X time  array
                
                # #filter trials index fullfill condition
                # Behavdata_df_filtered,filterInd = SortfilterDF(Behavdata_df,filterlable = filterCond)# trials fullfill behave filterCondition
                # xx = [LFPSeg[i] for i in filterInd] 
                # yy = Behavdata_df[labellist].loc[filterInd,:]
                xx_date[frekeys] = LFPSeg.copy()
        
  

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))








