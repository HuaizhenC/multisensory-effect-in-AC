import time
from datetime import timedelta
import numpy as np
import pickle
import pandas as pd
from spikeUtilities import loadPreprocessMat,getPSTHdf 
from sharedparam import getMonkeyDate_all,neuronfilterDF
#############
# save psth as pkl files aligned to an event in a set window session by sessoin 
#############

# MonkeyDate_all = {'Elay':['230420']}#,
# Pathway='/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/data'
# ResPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/decoder/'
# MonkeyDate_all = {'Elay':['230406','230414','230714','230912','240419','240429','240501']}

# MonkeyDate_all = {'Elay':['230406','230414','230912','230420','230509','230525','230531','230602','230606','230608','230613',
#                         '230616','230620','230627',
#                         '230705','230711','230717','230718','230719','230726','230728',
#                         '230802','230808','230810','230814','230818','230822','230829',
#                         '230906','230908','230915','230919','230922','230927',
#                         '231003','231004','231010','231017','231019',
#                         '231128',
#                         '240109','240112','240116','240119',
#                         '240202','240206','240209','240213',
#                         '240306','240318',
#                         '240419','240429','240501'],
#                         'Wu':['240522','240524','240527','240530','240605','240610','240611','240612','240614','240618','240624','240625','240626','240628',
# '240701','240702','240703','240704','240705','240708','240709','240710','240711','240712','240713','240715','240716','240717']}

MonkeyDate_all = getMonkeyDate_all()
Pathway='/home/huaizhen/Documents/MonkeyAVproj/data/preprocNeuralMatfiles/Vprobe+EEG/'
ResPathway = '/home/huaizhen/Documents/MonkeyAVproj/data/PSTHdataframe/'

##### dont change the parameters, if change need to rerun all sessionss
# # get psth in non overlap windows for dimension reduction analysis
# bin = 50 #ms
# binmethod = 'nonoverlaptimwin'  
# baselinewin = bin
# binmethodtimstep=[] # 

## get fr in slidewindows for decoding 
bin = 50 #ms
binmethod = 'overlaptimwin'  
baselinewin = bin
binmethodtimstep=0.001 # temporal resolution: stepsize of slidewindow (s)

# timwinStart = [-1,1] # window for getcluster
# extrastr = 'align2coo'
# alignkeys = 'cooOnsetIndwithVirtual'

# timwinStart = [-1.5,0] # window for getcluster
# extrastr = 'align2js'
# alignkeys = 'JSOnsetIndwithVirtual'

alignInfo_all = [[[-1,1],'align2coo','cooOnsetIndwithVirtual'],[[-1.5,0],'align2js','JSOnsetIndwithVirtual']]

start_time = time.monotonic()
for alignInfo in alignInfo_all:
    timwinStart = alignInfo[0]
    extrastr = alignInfo[1]
    alignkeys = alignInfo[2]
    
    for Monkey,Date in MonkeyDate_all.items():   
        for Date_temp in Date:  
            print('......................'+'process session '+Monkey+'-'+Date_temp+'......................')
            start_time1 = time.monotonic()

            AllSUspk_df = pd.DataFrame()
            spikeTimeDict,labelDictFilter, \
                timeSamp2Chorus,spikefs,behavefs= loadPreprocessMat(Monkey,Date_temp,Pathway)
            
            for cc,cls in enumerate(list(spikeTimeDict.keys())):                               
                if any(substr in cls for substr in ['good','mua']):
                    print('...............'+'process cluster'+cls+'...............')
                    # # if any(substr in cls for substr in ['good','mua']):
                    # # get trial by trial raster in each cluster
                    # labelDict_sub = {}
                    # ntrials = 100
                    # for key,value in labelDictFilter.items():
                    #     labelDict_sub[key] = value[-ntrials:]
                    # labelDictFilter = labelDict_sub.copy()
                    # spikeTime_temp = spikeTimeDict[cls][-ntrials:]   
                                            
                    spikeTime_temp = spikeTimeDict[cls]

                    ######## raw spkrate over time in each trial  
                    binedFRxTimdf_temp,spknumcoloumns = getPSTHdf(timeSamp2Chorus,spikeTime_temp,spikefs,behavefs,
                                                    labelDictFilter,alignkeys,[timwinStart[0],timwinStart[-1]],bin/1000,baselinewin/1000,binmethod,binmethodtimstep)                   
                
                    # concatenate ordered trials in each cluster
                    trials = binedFRxTimdf_temp.shape[0]
                    info_temp_df = pd.DataFrame({'Monkey':[Monkey]*trials,'sess':[Date_temp]*trials,'cls':[cls]*trials})  
                    AllSUspk_df_temp = pd.concat([info_temp_df,binedFRxTimdf_temp[spknumcoloumns]],axis=1) 
                    AllSUspk_df = pd.concat([AllSUspk_df,AllSUspk_df_temp],axis=0) 
                    print(AllSUspk_df.shape)
                    print(list(AllSUspk_df.columns)[:30])
            print('time spend to get psth for this session')
            print(timedelta(seconds= time.monotonic()- start_time1)) 
            pickle.dump(AllSUspk_df,open(ResPathway+Monkey+'_'+Date_temp+'_allSU+MUA_alltri_'+extrastr+'_'+binmethod+'_'+str(bin)+'msbin_1msstepsizeRawPSTH_df.pkl','wb')) 
        print('done with ' +Monkey) 

times2 = time.monotonic()
print('total time spend to get all indivisual PSTH from 2 monkeys:')
print(timedelta(seconds= times2- start_time))        
