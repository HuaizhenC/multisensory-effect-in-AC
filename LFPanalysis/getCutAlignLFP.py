import time
from datetime import timedelta
import numpy as np
import scipy
import seaborn as sns
import h5py
import pickle
import os
import pandas as pd
from spikeUtilities import loadPreprocessMat,decodrespLabel2str
from eegUtilities import loadTimWindowedLFP

# MonkeyDate_all = {'Elay':['230420']}

# matlabfilePathway='/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/data/'
# genH5Pathway ='/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/data/'
# figsavPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Figures/'
# ResPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/LFP/'
# AVmodPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/AVmodIndex/'
# glmfitPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/glmfit/'

MonkeyDate_all = {'Elay':['230927','231003','231004','231010'], 
                  'Wu':['230809','230815','230821','230830',
                        '230905','230911','230913','230918','230925',
                          '231002','231006','231009']}

matlabfilePathway='/data/by-user/Huaizhen/preprocNeuralMatfiles/Vprobe+EEG/'
genH5Pathway = '/data/by-user/Huaizhen/LFPcut/2Dpower+phase/'

# timerange parameter, need to have the same 0
timeRange_lfp = [-0.8,0.6] #fs 1220.703125

# note: cut and align lfp for all trials, except trials with NAN respLabel ( no sound played at all) 
# trials with nan alignkeys times, will be saved as a row of NAN

alignStr = 'align2coo'
alignkeys = 'cooOnsetIndwithVirtual'#'cooOnsetIndwithVirtual','VidOnsetIndwithVirtual','JSOnsetIndwithVirtual'
x0str = 'cooOn'

for Monkey,Date in MonkeyDate_all.items():
    ITClfp_all={}
    for Date_temp in Date:      
        print('...............'+'process session '+Monkey+'-'+Date_temp+'...............')
        start_time = time.monotonic()

        # complete spk info for this session
        _,labelDictFilter, \
            timeSamp2Chorus,spikefs,behavefs= loadPreprocessMat(Monkey,Date_temp,matlabfilePathway)  

        # LFP info for this session, lfp matrix (generated by loadTimWindowedLFP) already excludes trials with NaN respLabel 
        Behavdata_df = decodrespLabel2str(pd.DataFrame.from_dict(labelDictFilter)).copy()   
        LFPfilePathway = matlabfilePathway+Monkey+'_'+Date_temp+'_preprocLFPtimeseries.h5'
        filename = genH5Pathway+Monkey+Date_temp+'_2Dpowerphase_trial_'+alignStr+'_lfpXtrialCohtimrange'                          
        Behavdata_df_new= loadTimWindowedLFP(LFPfilePathway,
                                             Behavdata_df,
                                             behavefs,
                                             alignkeys,
                                             timeRange_lfp,
                                             [],'complex',[],
                                             filename,
                                             baselineCorrect=True) #[CATkey] chan X freq X time [CATkeytrials] #                        
        pickle.dump(Behavdata_df_new,open(genH5Pathway+Monkey+Date_temp+'_trial_Behavdata_'+alignStr+'.pkl','wb'))            

        end_time = time.monotonic()
        print('time spend for generating LFP power&phase .h5 files')
        print(timedelta(seconds=end_time - start_time))

    

        
                     

            
              
                    



