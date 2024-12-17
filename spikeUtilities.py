import os
import random
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
import pandas as pd
import mat73
import numpy as np
from statsmodels.nonparametric.kde import KDEUnivariate
from sklearn.neighbors import KernelDensity
from scipy.integrate import quad
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV, LinearRegression
from sklearn.metrics import roc_auc_score
from scipy.optimize import curve_fit
from scipy import stats
from scipy.stats import norm
import scipy
import re
cpus = 1
from multiprocessing import Pool
from decoders import applyweibullfit
from matplotlib import pyplot as plt 
import warnings
# try:
#     from scipy.stats import _warnings_errors
#     warnings.simplefilter("error", _warnings_errors.NearConstantInputWarning)
#     warnings.simplefilter("error", _warnings_errors.ConstantInputWarning)
# except Exception as e:
#     print(e)
#     pass
def loadDDMBehavMat(DDMmatsavepath):
    # Function to remove brackets
    def remove_brackets(x):
        # If x is a list with one element, return that element
        if isinstance(x, list) and len(x) == 1:
            return x[0]
        # If x is a string, strip the brackets
        elif isinstance(x, str):
            return x.strip('[]')
        return x

    ### get ddm fit results about RT in each session
    data = mat73.loadmat(DDMmatsavepath+'one-choice-DDMfit_struct.mat')
    fitparam_df = pd.DataFrame.from_dict(data['fitparamStruct'])   
    fitparam_df = fitparam_df.applymap(remove_brackets) 
    return fitparam_df

def loadBehavMat(Monkey,Date_temp,Pathway):
    behavefilename = Monkey+'_'+Date_temp+'_labviewNsynapseBehavStruct.mat'
    behavefilePathway = os.path.join(Pathway,behavefilename)
    behavdata = mat73.loadmat(behavefilePathway)
    eventsIndDict1 = behavdata['eventsIndStruct']
    labelDict = behavdata['LabviewBehavStruct']
    ## manually add modified target onset
    eventsIndDict = addvirtCooOnset(eventsIndDict1,labelDict)
    eventsIndDict = addvirtVidOnset(eventsIndDict,labelDict)
    eventsIndDict = addvirtJSOnset(eventsIndDict,labelDict) # needs to be after addvirtCooOnset
    eventsIndDict = addvirtDTOnset(eventsIndDict,labelDict)

    labelDictFilter = combineLabel(eventsIndDict,labelDict)
    ## shift snr so that V trials SNR is 0 instead of nan, 
    snr = np.nan_to_num(labelDictFilter['snr'].copy()+20)
    labelDictFilter['snr-shift'] = snr.copy()
    labelDictFilter['trialNum'] = np.arange(0,len(snr))

    behavefs = behavdata['synapseBehavStruct']['fs']
    return labelDictFilter,behavefs

def loadPreprocessMat(Monkey,Date_temp,Pathway,loadSPKflag='on'):
    if loadSPKflag=='on':
        spikefilename = Monkey+'_'+Date_temp+'_preprocSpiketimeseries.mat'
        spikefilePathway = os.path.join(Pathway,spikefilename)
        spikedata = mat73.loadmat(spikefilePathway)
        spikeTimeDict = spikedata['SpikepreprocTimeseries']        
        spikefs = spikeTimeDict['fs']
        timeSamp2Chorus = np.arange(int(spikeTimeDict['timeRange'][0]*spikefs),int(spikeTimeDict['timeRange'][1]*spikefs),1)
    else:
        spikeTimeDict = []
        timeSamp2Chorus = []
        spikefs = []
    labelDictFilter,behavefs = loadBehavMat(Monkey,Date_temp,Pathway)
    return spikeTimeDict,labelDictFilter, timeSamp2Chorus,spikefs,behavefs

def addvirtCooOnset(eventsIndDict,labelDict):
    #only either of coo or vid delivered, cooOnsetIndwithVirtual is not nan
    cooOnsetIndwithVirtual = []
    for tt,mod in enumerate(labelDict['trialMod']):
        # If coo was delivered: in A/AV trials, coo onset unchanged; in V trials, virtual coo onset estimated    
        if not np.isnan(eventsIndDict['cooOnsetInd'][tt][0]):
            if np.isin(mod,['a','av']):
                cooOnsetIndwithVirtual.append([eventsIndDict['cooOnsetInd'][tt][0]]) 
            if mod=='v':
                cooOnsetIndwithVirtual.append([np.array(eventsIndDict['vidOnsetInd'][tt][0]+636)]) 
        
        # if coo was not delivered, 
        elif np.isnan(eventsIndDict['cooOnsetInd'][tt][0]):   
            # V/av trials vid delivered, virtual cooOnset estimated         
            if np.isin(mod,['v','av']) and not np.isnan(eventsIndDict['vidOnsetInd'][tt][0]): #A trial could have fake vidOnsetInd
                cooOnsetIndwithVirtual.append([np.array(eventsIndDict['vidOnsetInd'][tt][0]+636)])  
            # vid not delivered             
            else:
                cooOnsetIndwithVirtual.append([np.array(np.nan)])  
                       
    eventsIndDict['cooOnsetIndwithVirtual'] = cooOnsetIndwithVirtual.copy()
    return eventsIndDict

def addvirtVidOnset(eventsIndDict,labelDict):
    #only either coo or vid delivered, VidOnsetIndwithVirtual is not nan 
    VidOnsetIndwithVirtual = []
    for tt,mod in enumerate(labelDict['trialMod']):
        # If vid was delivered: in V/AV trials, vid onset unchanged;   
        # in A trials could has 'valid' 'vidOnsetInd', 
        # becasue TDT ramp up from 0 to 2, or ramp down from 2 to 0 
        # could generate a sample valued 1, which has to be detected as a spurious vidOnsetInd  
        if not np.isnan(eventsIndDict['vidOnsetInd'][tt][0]) and np.isin(mod,['v','av']): 
            VidOnsetIndwithVirtual.append([eventsIndDict['vidOnsetInd'][tt][0]]) 
        #if coo was delivered, in A trials, virtual vid onset estimated
        elif mod=='a' and not np.isnan(eventsIndDict['cooOnsetInd'][tt][0]):
            VidOnsetIndwithVirtual.append([np.array(eventsIndDict['cooOnsetInd'][tt][0]-636)])         
        # trials no vid no coo or no sound
        else:                                  
            VidOnsetIndwithVirtual.append([np.array(np.nan)])            
 
    eventsIndDict['VidOnsetIndwithVirtual'] = VidOnsetIndwithVirtual.copy()
    return eventsIndDict    

def addvirtJSOnset(eventsIndDict,labelDict): 
    # A:  hit[0],miss/latemiss[1],FAa[2],erlyResp[88]
    # AV: hit[0],miss/latemiss[1],FAa[2],erlyResp[88],FAv[22]
    # V:  hit[0],CRv[10,100miss],FAa[2],FAv[22]
    #notes: 1) keep original jsonsetind
    #           a)hit,fa,earlyresponse in  A/AV conditions
    #           b)fa in V condition
    #       2) virtual jsonset = virtualcooonset+600ms
    #           a) miss/latemiss in A/AV conditions
    #           b) CR in V conditions

    JSOnsetIndwithVirtual = []
    for tt,mod in enumerate(labelDict['trialMod']):
        # only chorus delivered trials 
        if not np.isnan(eventsIndDict['chorusOnsetInd'][tt]):
            # (hit,fa,earlyresponse) of A/AV conditions, or (fa) of V condition 
            if (np.isin(mod,['a','av']) and np.isin(eventsIndDict['respLabel'][tt][0],[0,2,88,22]))\
                    or (mod=='v' and np.isin(eventsIndDict['respLabel'][tt][0],[2,22])):
                JSOnsetIndwithVirtual.append([eventsIndDict['joystickOnsetInd'][tt][0]]) 
            #miss/late miss trials in A/AV, CR trials in V, estimate fake jsonsetInd, virtualcooonset+600ms, 
            elif (np.isin(mod,['a','av']) and np.isin(eventsIndDict['respLabel'][tt][0],[1]))\
                  or (mod=='v' and np.isin(eventsIndDict['respLabel'][tt][0],[0,10,100])):
                JSOnsetIndwithVirtual.append([np.array(eventsIndDict['cooOnsetIndwithVirtual'][tt][0]+733)])
            else:
                JSOnsetIndwithVirtual.append([np.array(np.nan)])                     
        else:
            JSOnsetIndwithVirtual.append([np.array(np.nan)])       

    eventsIndDict['JSOnsetIndwithVirtual'] = JSOnsetIndwithVirtual.copy()
    return eventsIndDict     

def addvirtDTOnset(eventsIndDict,labelDict):
    DTOnsetIndwithVirtual = []
    for tt,mod in enumerate(labelDict['trialMod']):
        # only chorus delivered trials 
        if not np.isnan(eventsIndDict['chorusOnsetInd'][tt]):
            # (hit) of A/AV conditions
            if (np.isin(mod,['a','av']) and np.isin(eventsIndDict['respLabel'][tt][0],[0])):
                DTOnsetIndwithVirtual.append([np.array(int(eventsIndDict['cooOnsetIndwithVirtual'][tt][0]+eventsIndDict['DT'][tt][0]*eventsIndDict['fs']))]) 
            else:
                DTOnsetIndwithVirtual.append([np.array(np.nan)])                     
        else:
            DTOnsetIndwithVirtual.append([np.array(np.nan)])       

    eventsIndDict['DTOnsetIndwithVirtual'] = DTOnsetIndwithVirtual.copy()
    return eventsIndDict

def combineLabel(eventsIndDictSynap,labelDictLabv):
    keys2del = ['fs','note']
    newlabelDict_temp = {key:value for key,value in eventsIndDictSynap.items() if key not in keys2del}
    newlabelDict = {}
    for kk,vv in newlabelDict_temp.items():
        vv_temp = [vv[i][0].astype(float).tolist() for i in range(len(vv))]    
        newlabelDict[kk] = vv_temp.copy()

    keys2del2 = ['Reward','Stim','rt','joystick','respLabel']
    newlabelDict.update({key:value for key,value in labelDictLabv.items() if key not in keys2del2})
    return newlabelDict

def convertspikeTime(spikeTime_tt,spktimshift,spikefs,timeRange2save):
    #convert spikeTime_tt(samples relative to chorus onset) 
    # to time (seconds relative to target onset)
    if len(spikeTime_tt[0].shape)==0 :
        spikeTime_tt = [spikeTime_tt.copy()]  
    spikTime_array_temp = (spikeTime_tt[0]+spktimshift)/spikefs

    # get the data within the time window of interest 
    indwin = np.where((spikTime_array_temp>=timeRange2save[0])&(spikTime_array_temp<=timeRange2save[1]))       
    spikTime_list_temp = spikTime_array_temp[indwin].copy()
    ## trials without spikes will have one row of -1000 spiketime in the dataframe
    if len(spikTime_list_temp)>=1:           
        spikTime_df_temp =pd.DataFrame({'spktim':spikTime_list_temp})
    else:
        spikTime_df_temp =pd.DataFrame({'spktim':[-1000]}) 
    return spikTime_df_temp
    
def getClsSPK(tt,spikeTime_tt,rowseries,eventsIndref_tt,eventsIndtarg_tt,eventsIndJS_tt,NaneventsIndtarg,NaneventsDelay,timeSamp2Chorus,timeRange2save,spikefs,behavefs):
    rowdf = rowseries.to_frame().transpose()
        
    # for trials with valid target events 
    if not np.isnan(eventsIndref_tt) and not np.isnan(eventsIndtarg_tt):
        # shift time (relative to chorus onset) to the targetevent of each trial
        spktimshift = -spikefs*(eventsIndtarg_tt-eventsIndref_tt)/behavefs #in the unit of samples
        spikTime_df_temp = convertspikeTime(spikeTime_tt,spktimshift,spikefs,timeRange2save)
        #jsevent delayed time (s) relative to targ in each trial, 
        # (trials coo not played should be 0, trials without jsmove should be nan)
        JS2Targ_temp = (eventsIndJS_tt-eventsIndtarg_tt)/behavefs  
    # for trials with invalid target events but valid nantarget events, align spktime to nantarget events       
    elif not np.isnan(eventsIndref_tt) and np.isnan(eventsIndtarg_tt) and len(NaneventsIndtarg)>0 and not np.isnan(NaneventsIndtarg[tt]):
        # shift time (relative to chorus onset) to the targetevent of each trial
        spktimshift = -spikefs*(NaneventsIndtarg[tt]-eventsIndref_tt)/behavefs #in the unit of samples
        spikTime_df_temp = convertspikeTime(spikeTime_tt,spktimshift,spikefs,timeRange2save)
        JS2Targ_temp = 0            
    # for all other trials will have one row of nan spiketime       
    else:
        spikTime_df_temp = pd.DataFrame({'spktim':[np.nan]})
        JS2Targ_temp = np.nan   

    ## add labels info to spiketim
    for cc in rowdf.columns:
        spikTime_df_temp[cc]=list(rowdf[cc])[0]
    spikTime_df_temp['trialNum'] = tt
    spikTime_df_temp['jsOnset'] = JS2Targ_temp
    return spikTime_df_temp

def getClsRasterMultiprocess(timeSamp2Chorus,spikeTime,spikefs,behavefs,eventsIndref,eventsIndtarg,eventsIndJS,timeRange2save,labelDict,NaneventsIndtarg = [],NaneventsDelay = 0.3):
# get raster dataframe for this cluster, temporally cut according to timeTange2save 
# spikeTime originally align to the chorus onset
# will temporally align the raster relative to the eventsIndtarg
# trials have nan in eventsIndtarg(JSonset) might be forced to a valid value of NaneventsIndtarg+NaneventsDelay

    label_df = pd.DataFrame(labelDict)    
    argItems = [(tt,spikeTime_tt,rowseries,eventsIndref_tt,eventsIndtarg_tt,eventsIndJS_tt,NaneventsIndtarg,NaneventsDelay,timeSamp2Chorus,timeRange2save,spikefs,behavefs)\
                 for tt,(spikeTime_tt,(_,rowseries),eventsIndref_tt,eventsIndtarg_tt,eventsIndJS_tt) in \
                    enumerate(zip(spikeTime,label_df.iterrows(),eventsIndref,eventsIndtarg,eventsIndJS))]
    
    spikTime_df = pd.DataFrame()
    # for ii in argItems:
    #     spikTime_df_temp = getClsSPK(*ii)
    #     spikTime_df=pd.concat([spikTime_df,spikTime_df_temp]) 
    with Pool(processes=cpus) as p:
        # spikeTime is a list: trails X spikeNum
        for spikTime_df_temp in p.starmap(getClsSPK,argItems):
            # print(spikTime_df_temp)
            spikTime_df=pd.concat([spikTime_df,spikTime_df_temp])
            # print(str(spikTime_df.shape[0])+'trials done') 
    p.close()
    p.join()

    return spikTime_df

def SortGroupDF(neuraldata_df_list,labelDict,filterlable = {'respLabel':0},condlabel=['trialMod','snr']):
    #neuraldata_df_list are list of dataframes, eahc dataframe is trials X feature
    neuraldata_df_grouped_list = []
    for neuraldata_df in neuraldata_df_list:
        # add condition columns to combined dataframe
        for cc in condlabel:
            neuraldata_df[cc] = labelDict[cc]
    # apply filter condition
        ind = np.arange(0,neuraldata_df.shape[0])
        for ii,(kk,vv) in enumerate(filterlable.items()):
            ind_temp = np.where(labelDict[kk]==vv)[0]
            ind = np.intersect1d(ind,ind_temp)
        neuraldata_df_filetered = neuraldata_df.iloc[ind,:]
    # # neuraldata_df_filetered shouldn't has any nan in datacolumns after conditional filtering
        colsubset = list(neuraldata_df_filetered.columns)[0:-len(condlabel)]
        mask = neuraldata_df_filetered[colsubset].isna().any(axis=1)
        NaNrow_index = mask.index[mask].tolist()
        if len(NaNrow_index)>=1:
            print('WARNING: nan still exist in data after applying conditional filtering')
        # neuraldata_df_filetered_clean = neuraldata_df_filetered.dropna(subset=colsubset )
    # #group by modalities    
        neuraldata_df_grouped_list.append(neuraldata_df_filetered.groupby(condlabel[0]))

    return neuraldata_df_grouped_list
   
def SortfilterDF(neuraldata_df,filterlable = {'respLabel':[0]}):
    #neuraldata_df is dataframe: trials X feature
    # apply filter condition
    neuraldata_df = neuraldata_df.reset_index(drop=True)
    ind = np.array(list(neuraldata_df.index))
    for ii,(kk,vv) in enumerate(filterlable.items()):      
        # print('key:'+kk+'  val:'+str(vv))  
        # ind_temp = np.where(np.isin(neuraldata_df[kk],vv))[0]
        ind_temp = np.array(list(neuraldata_df[neuraldata_df[kk].isin(vv)].index))
        ind = np.intersect1d(ind,ind_temp)
    neuraldata_df_filetered = neuraldata_df.iloc[ind,:]
    return neuraldata_df_filetered,ind

def getPSTH(spikeTimelist,timpnts,bintimeLen,totaltrials=1,totaltrialsYaxis=1, scaleFlag='on', kernelFlag='on',triInfo=None):
    def gaussian_kernel(size, sigma):
        size = int(size) // 2
        x = np.arange(-size, size+1)
        kernel = np.exp(-(x**2) / (2*sigma**2))
        return kernel / sum(kernel)

    psth = []
    for tt in timpnts:
        psth.append(sum(1 for x in spikeTimelist if tt <= x < tt+bintimeLen)/totaltrials)
    # scale value for visual effect
    if scaleFlag=='on':    
        psth = psth.copy()
     
    # smooth psth
    if kernelFlag=='on':
        # kernel = gaussian_kernel(50, 25)
        kernel = gaussian_kernel(20, 10)
        psth = scipy.signal.convolve(psth, kernel, mode='same')
    #set output
    if triInfo is not None:
        triaOutput = triInfo
    else:
        triaOutput = None
    return psth,triaOutput


def getISI(spikTime_list):
    # ISI in the unite of ms
    ISI_list = []
    for spikTime in spikTime_list:
        if len(spikTime)>=2:
            ISI_temp = [1000*(spikTime[i+1]-spikTime[i]) for i in range(len(spikTime)-1)]
            ISI_list.append(ISI_temp)
        else:
            ISI_list.append([])
    return ISI_list

def getTrialFRpk(spikeTimedf,timwin,spikefs,bin=10):
    spikenum_df = pd.DataFrame()
    for tt in spikeTimedf['trialNum'].unique(): 
        group = spikeTimedf[spikeTimedf['trialNum']==tt]
        spikenum_df_temp = pd.DataFrame(group.iloc[0,:].copy()).transpose()
        # est fr change overtime
        fr_overtime=[]
        timeseq = np.arange(timwin[0],timwin[1]+bin/1000,1/spikefs)
        for tt in timeseq:
            fr_overtime.append(group[(group['spktim']>=tt) & (group['spktim']<tt+bin/1000)].shape[0])         
        spikenum_df_temp['frpkdelay'] = timeseq[np.where(fr_overtime==fr_overtime.max())[0]]
        spikenum_df_temp.drop('spktim',axis=1,inplace=True)
        spikenum_df = pd.concat([spikenum_df,spikenum_df_temp])
    return spikenum_df.reset_index(drop=True)

def countTrialSPKs(spikeTimedf,estwin='off',fs = 1220.703125,winTim=[0,0.3],conswin = 0.6):
    spikenum_df = pd.DataFrame()
    # if estwin=='aveRate':
        # print('warning: assume spkse align to cooOnsetIndwithVirtual when est FR!!')

    for tt in spikeTimedf['trialNum'].unique(): 
        group = spikeTimedf[spikeTimedf['trialNum']==tt]
        spikenum_df_temp = pd.DataFrame(group.iloc[0,:].copy()).transpose()
        if estwin=='off':# num of spikes in this trial
            spikenum_df_temp['spknum'] = group[group['spktim']>-1000].shape[0] 
        
        elif estwin=='aveRate': 
            # assume spks in spikeTimedf align2coowithVirtualCoo 
            # rate: spikes/sec--> num of spikes between cooonset and jsonset/virtualcoooffset
            # respLabel decode:  
            # nosoundplayed: [nan] 
            # A:  hit[0],miss/latemiss[1],FAa[2],erlyResp[88]
            # AV: hit[0],miss/latemiss[1],FAa[2],erlyResp[88],FAv[22]
            # V:  hit[0],CRv[10,100miss],FAa[2],FAv[22]
            # conswin = 0.6 #s
            #spkRateCRa2: fr in coswin before cooonset in all trials
            #spkRateCRa3: fr in coswin before cooonset in a trials, coswin before vidonset in av trials
            #spkRateCRa: only trials with AVoffset>60, fr in coswin before cooonset 
            #spkRateCRa1: only trials with AVoffset>60,fr in coswin before cooonset in a trials, coswin before vidonset in av trials
            avoffset = (group['cooOnsetIndwithVirtual'][0]-group['vidOnsetInd'][0])/fs
            # avoffset = 636/fs
            # a+av miss or v CR
            if group['respLabel'][0]==1 or np.all([group['trialMod'][0]=='v',np.isin(group['respLabel'][0],[0,10,100])]):
                spikenum_df_temp['spkRate'] = group[(group['spktim']>=0) & (group['spktim']<conswin)].shape[0]/conswin
                if group['trialMod'][0]=='a':
                    spikenum_df_temp['spkRateCRa2'] = group[(group['spktim']>=-conswin-0.1) & (group['spktim']<-0.1)].shape[0]/conswin # FR for CRa
                    spikenum_df_temp['spkRateCRa3'] = group[(group['spktim']>=-conswin-0.1) & (group['spktim']<-0.1)].shape[0]/conswin # FR for CRa
                    if group['AVoffset'][0]>61:# fr before coo onset in long coonset delay trials
                        spikenum_df_temp['spkRateCRa'] = group[(group['spktim']>=-conswin-0.1) & (group['spktim']<-0.1)].shape[0]/conswin # FR for CRa
                        spikenum_df_temp['spkRateCRa1'] = group[(group['spktim']>=-conswin-0.1) & (group['spktim']<-0.1)].shape[0]/conswin # FR for CRa
                if np.isin(group['trialMod'][0],['v','av']):
                    spikenum_df_temp['spkRateCRa2'] = group[(group['spktim']>=-conswin-0.1) & (group['spktim']<-0.1)].shape[0]/conswin # FR for CRa                       
                    spikenum_df_temp['spkRateCRa3'] = group[(group['spktim']>=-conswin-0.1-avoffset) & (group['spktim']<-0.1-avoffset)].shape[0]/conswin # FR for CRa                                           
                    if group['AVoffset'][0]>61:# fr before vid onset
                        spikenum_df_temp['spkRateCRa'] = group[(group['spktim']>=-conswin-0.1-avoffset) & (group['spktim']<-0.1-avoffset)].shape[0]/conswin # FR for CRa            
                        spikenum_df_temp['spkRateCRa1'] = group[(group['spktim']>=-conswin-0.1) & (group['spktim']<-0.1)].shape[0]/conswin # FR for CRa            
            # a+av hit or earlyresp
            elif np.isin(group['trialMod'][0],['a','av']) and np.isin(group['respLabel'][0],[0,88]):
                if group['jsOnset'][0]<=0:
                    print('ERROR: negative RT in a/av modes with hit/early response, trial'+str(tt))
                    spikenum_df_temp['spkRate'] = np.nan
                else:
                    spikenum_df_temp['spkRate'] = group[(group['spktim']>=0) & (group['spktim']<group['jsOnset'][0])].shape[0]/group['jsOnset'][0]
                if group['trialMod'][0]=='a':
                    spikenum_df_temp['spkRateCRa2'] = group[(group['spktim']>=-conswin-0.1) & (group['spktim']<-0.1)].shape[0]/conswin # FR for CRa
                    spikenum_df_temp['spkRateCRa3'] = group[(group['spktim']>=-conswin-0.1) & (group['spktim']<-0.1)].shape[0]/conswin # FR for CRa
                    if group['AVoffset'][0]>61:# fr before coo onset in long coonset delay trials
                        spikenum_df_temp['spkRateCRa'] = group[(group['spktim']>=-conswin-0.1) & (group['spktim']<-0.1)].shape[0]/conswin # FR for CRa
                        spikenum_df_temp['spkRateCRa1'] = group[(group['spktim']>=-conswin-0.1) & (group['spktim']<-0.1)].shape[0]/conswin # FR for CRa
                if group['trialMod'][0]=='av':
                    spikenum_df_temp['spkRateCRa2'] = group[(group['spktim']>=-conswin-0.1) & (group['spktim']<-0.1)].shape[0]/conswin # FR for CRa            
                    spikenum_df_temp['spkRateCRa3'] = group[(group['spktim']>=-conswin-0.1-avoffset) & (group['spktim']<-0.1-avoffset)].shape[0]/conswin # FR for CRa            

                    if group['AVoffset'][0]>61:# fr before vid onset in long virtual coonset delay trials
                        spikenum_df_temp['spkRateCRa'] = group[(group['spktim']>=-conswin-0.1-avoffset) & (group['spktim']<-0.1-avoffset)].shape[0]/conswin # FR for CRa            
                        spikenum_df_temp['spkRateCRa1'] = group[(group['spktim']>=-conswin-0.1) & (group['spktim']<-0.1)].shape[0]/conswin # FR for CRa            
            # FA trials, spkrate is estimated in conswin before jsOnset
            elif np.isin(group['respLabel'][0],[2,22]):
                spikenum_df_temp['spkRate'] = group[(group['spktim']<0) & (group['spktim']>-conswin)].shape[0]/conswin
            else:
                spikenum_df_temp['spkRate'] = np.nan
                spikenum_df_temp['spkRateCRa'] = np.nan
                spikenum_df_temp['spkRateCRa1'] = np.nan
                spikenum_df_temp['spkRateCRa2'] = np.nan
                spikenum_df_temp['spkRateCRa3'] = np.nan
        elif estwin=='aveRate_equalWin': 
            conswin = 0.6 #s
            avoffset = (group['cooOnsetIndwithVirtual'][0]-group['vidOnsetInd'][0])/fs
            # avoffset = 636/fs
            # a+av miss, hit or earlyresp or v CR
            if np.isin(group['respLabel'][0],[1,0,88]) or np.all([group['trialMod'][0]=='v',np.isin(group['respLabel'][0],[0,10,100])]):
                spikenum_df_temp['spkRate'] = group[(group['spktim']>=0) & (group['spktim']<conswin)].shape[0]/conswin
                if group['trialMod'][0]=='a' and group['AVoffset'][0]>61:# fr before coo onset
                    spikenum_df_temp['spkRateCRa'] = group[(group['spktim']>=-conswin-0.1) & (group['spktim']<-0.1)].shape[0]/conswin # FR for CRa
                if np.isin(group['trialMod'][0],['v','av']) and group['AVoffset'][0]>61:# fr before vid onset
                    spikenum_df_temp['spkRateCRa'] = group[(group['spktim']>=-conswin-0.1-avoffset) & (group['spktim']<-0.1-avoffset)].shape[0]/conswin # FR for CRa            
            else:
                spikenum_df_temp['spkRate'] = np.nan
                spikenum_df_temp['spkRateCRa'] = np.nan
        elif estwin=='aveRate_equalWin2': 
            conswin = 0.6 #s
            avoffset = (group['cooOnsetIndwithVirtual'][0]-group['vidOnsetInd'][0])/fs
            # avoffset = 636/fs
            # a+av miss, hit or earlyresp or v CR
            if np.isin(group['respLabel'][0],[1,0,88]) or np.all([group['trialMod'][0]=='v',np.isin(group['respLabel'][0],[0,10,100])]):
                spikenum_df_temp['spkRate'] = group[(group['spktim']>=(group['jsOnset'][0]-conswin)) & (group['spktim']<group['jsOnset'][0])].shape[0]/conswin
                if group['trialMod'][0]=='a' and group['AVoffset'][0]>61:# fr before coo onset
                    spikenum_df_temp['spkRateCRa'] = group[(group['spktim']>=-conswin-0.1) & (group['spktim']<-0.1)].shape[0]/conswin # FR for CRa
                if np.isin(group['trialMod'][0],['v','av']) and group['AVoffset'][0]>61:# fr before vid onset
                    spikenum_df_temp['spkRateCRa'] = group[(group['spktim']>=-conswin-0.1-avoffset) & (group['spktim']<-0.1-avoffset)].shape[0]/conswin # FR for CRa            
            else:
                spikenum_df_temp['spkRate'] = np.nan
                spikenum_df_temp['spkRateCRa'] = np.nan

        elif estwin=='BlNSig':
            spikenum_df_temp['spkRate_baseline'] = group[(group['spktim']>=winTim[0]) & (group['spktim']<=0)].shape[0]/np.abs(winTim[0])
            spikenum_df_temp['spkRate_sig'] = group[(group['spktim']>0) & (group['spktim']<=winTim[1])].shape[0]/np.abs(winTim[1])
        elif estwin=='subwinoff':
            spikenum_df_temp['spknum'] = group[(group['spktim']>=winTim[0]) &(group['spktim']<winTim[1])].shape[0]

        spikenum_df_temp.drop('spktim',axis=1,inplace=True)
        spikenum_df = pd.concat([spikenum_df,spikenum_df_temp])
    return spikenum_df.reset_index(drop=True)

def decodrespLabel2str(df):
    vv = df['respLabel'].apply(lambda x: 'hit' if x==0 \
                            else ('miss' if x==1 \
                                else ('FAa' if x==2 \
                                  else ('FAv' if x==22 \
                                     else ('CR' if any(ss==x for ss in [10,100]) \
                                      else ('elyResp' if x==88 else 'NaN')))))).values
    df['respLabel'] = vv
    df.loc[(df['trialMod']=='v') & (df['respLabel']=='hit'),'respLabel'] = 'CR'
    dfreset = df.reset_index(drop=True)
    # print(dfreset.respLabel)
    return dfreset.copy()

def addIVcol(df):
    # v snr is 0, a/av snr value are positive
    snr = df['snr-shift'].values 
    vv = df['trialMod'].apply(lambda x: 1 if any(x==ss for ss in ['v','av']) else -1).values
    aa = df['trialMod'].apply(lambda x: 1 if any(x==ss for ss in ['a','av']) else -1).values

    df = pd.concat([df.reset_index(drop=True),pd.DataFrame({'V':vv,'A':aa,'AV':vv*aa,'AV-snr':vv*snr})],axis=1)
    return df

def addIVcol2(df):
    # nan snr is 0, valid snr value are positive
    snr = df.snr.values+np.abs(np.min(df.snr.dropna().unique()))+5  
    snr[np.isnan(snr)] = 0
    vv = df['trialMod'].apply(lambda x: 1 if x=='v' else 0).values
    aa = df['trialMod'].apply(lambda x: 1 if x=='a' else 0).values
    av = df['trialMod'].apply(lambda x: 1 if x=='av' else 0).values
    df = pd.concat([df.reset_index(drop=True),pd.DataFrame({'V':vv,'A':aa,'AV':av})],axis=1)
    return df

def addIVcol3(df):
    # nan snr is 0, valid snr value are positive
    snr = df.snr.values+np.abs(np.min(df.snr.dropna().unique()))+5 
    snr[np.isnan(snr)] = 0
    vv = df['trialMod'].apply(lambda x: 1 if 'v' in x else 0).values
    aa = df['trialMod'].apply(lambda x: 1 if 'a' in x else 0).values
    av = df['trialMod'].apply(lambda x: 1 if x=='av' else 0).values
    df = pd.concat([df.reset_index(drop=True),pd.DataFrame({'V':vv,'A':aa*snr,'AV':av*snr})],axis=1)
    return df

def getspknum(timeSamp2Chorus,spikeTime_temp,spikefs,behavefs,labelDictFilter,alignkeys,timwin,filterdict, baselinewin=0.6, baselinecorrectFlag=True):
    spikeTimedf_temp_baseline = getClsRasterMultiprocess(timeSamp2Chorus,spikeTime_temp,spikefs,behavefs,\
                                        labelDictFilter['chorusOnsetInd'],\
                                        labelDictFilter['chorusOnsetInd'] ,\
                                        labelDictFilter['JSOnsetIndwithVirtual'],\
                                        [-0.05-baselinewin,-0.05],\
                                        labelDictFilter) 
    spikeNumdf_temp_baseline_raw=countTrialSPKs(spikeTimedf_temp_baseline,estwin='BlNSig',winTim=[-0.05-baselinewin,-0.05])# baseline fr estimated 300ms before chorus onset                                                 
    
    spikeTimedf_temp_ori = getClsRasterMultiprocess(timeSamp2Chorus,spikeTime_temp,spikefs,behavefs,\
                                        labelDictFilter['chorusOnsetInd'],\
                                        labelDictFilter[alignkeys] ,\
                                        labelDictFilter['JSOnsetIndwithVirtual'],\
                                        [timwin[0],timwin[1]+timwin[2]],\
                                        labelDictFilter,labelDictFilter['JSOnsetIndwithVirtual'])   # get spiketime saved in df                                                         
    # fr estimated between cooonset and jsonset, or 300ms before cooonset in long coonset delay trials
    spikeNumdf_temp=countTrialSPKs(spikeTimedf_temp_ori,estwin='aveRate',fs = behavefs,conswin = baselinewin) 
    # baseline correct spkrate, mitigate drift 
    if baselinecorrectFlag:
        spikeNumdf_temp['spkRate'] = spikeNumdf_temp['spkRate']-spikeNumdf_temp_baseline_raw['spkRate_baseline']
        spikeNumdf_temp['spkRateCRa'] = spikeNumdf_temp['spkRateCRa']-spikeNumdf_temp_baseline_raw['spkRate_baseline']
    # fr estimated in slide time windows, stepsize=0.1
    timwinStart = np.arange(timwin[0],timwin[1],0.1)
    for tt, tStart in enumerate(timwinStart):
        spikeNumdf_temp2=countTrialSPKs(spikeTimedf_temp_ori,estwin='subwinoff',fs = behavefs,winTim=[tStart,tStart+timwin[2]])
        spikeNumdf_temp['fr_'+str(np.round(tStart,decimals=2))] = spikeNumdf_temp2['spknum']/timwin[2]
        if baselinecorrectFlag:
            spikeNumdf_temp['fr_'+str(np.round(tStart,decimals=2))] =spikeNumdf_temp['fr_'+str(np.round(tStart,decimals=2))] - spikeNumdf_temp_baseline_raw['spkRate_baseline']
        
    spikeTimedf_temp1 = decodrespLabel2str(spikeNumdf_temp)
    spikeTimedf_temp_filter,_ = SortfilterDF(spikeTimedf_temp1,filterlable =filterdict)  # filter out trials 

    spikeTimedf_temp_filter = spikeTimedf_temp_filter.rename(columns={'spkRate':'spkRateOri'}).reset_index(drop=True)
    return spikeTimedf_temp_filter

def getSPK4slidewin( tStart, bin, spikeTimedf,spikeNumdf_temp):
    spikeNumdf_temp2=countTrialSPKs(spikeTimedf,estwin='subwinoff',winTim=[tStart,tStart+bin])
    if (spikeNumdf_temp2.trialNum.values==spikeNumdf_temp.trialNum.values).all():
        column_temp = pd.Series(spikeNumdf_temp2['spknum'].values/bin,name='fr_'+str(np.round(tStart,decimals=7)))
    else:
        print('mismtach between trialsequences of baselinefr and slidewindowfr!!!!!')  
    return column_temp

def getbindFRxtime(spikeTimedf,estwin,baselineFR, bin=50/1000,binmethod='nonoverlapwin',timstep=0.001, getclsInput=[]):
    if binmethod =='nonoverlaptimwin':
        # obtain firing rate over time in each trial, binned without overlap and baseline corrected, to mitigate drift 
        binedge = np.arange(estwin[0],estwin[1]+bin,bin)
        spktimeseries_df = pd.DataFrame()
        for tt in spikeTimedf['trialNum'].unique(): 
            group = spikeTimedf[spikeTimedf['trialNum']==tt]
            spikenum_df_temp = pd.DataFrame(group.iloc[0,1:].copy()).transpose() 
            spkratehist = np.histogram(group.iloc[:,0].values,bins=binedge)[0]/bin
            timpntsNum = len(spkratehist)
            bsFR_temp = baselineFR[baselineFR['trialNum']==tt]['spkRate_baseline'].values
            spikenum_df_temp1 = pd.concat([spikenum_df_temp,
                                        pd.DataFrame(bsFR_temp,columns=['baselineFR']),
                                        pd.DataFrame(spkratehist.reshape(1,-1),columns=[f'frate{i}' for i in range(len(spkratehist))])],axis=1) 
            spktimeseries_df = pd.concat([spktimeseries_df,spikenum_df_temp1])
            spknumcoloumns = ['respLabel','AVoffset', 'snr', 'trialMod', 'snr-shift', 'trialNum','baselineFR']\
                                +list(spktimeseries_df.columns)[-timpntsNum:]
    if binmethod =='overlaptimwin':
        #Get noise spk: CR fr before coo onset, baseline fr before chorus onset in each trial 
        if getclsInput[0]=='cooOnsetIndwithVirtual':
            spikeNumdf_temp=countTrialSPKs(spikeTimedf,estwin='aveRate',conswin = bin) #'spkRate' 'spkRateCRa'
        else:           
            spikeTimedf_temp_coo = getClsRasterMultiprocess(getclsInput[1],getclsInput[2],getclsInput[3],getclsInput[4],\
                                                getclsInput[5]['chorusOnsetInd'],\
                                                getclsInput[5]['cooOnsetIndwithVirtual'] ,\
                                                getclsInput[5]['JSOnsetIndwithVirtual'],\
                                                [-1,1],\
                                                getclsInput[5],getclsInput[5]['JSOnsetIndwithVirtual']) 
            spikeNumdf_temp=countTrialSPKs(spikeTimedf_temp_coo,estwin='aveRate',conswin = bin) #'spkRate' 'spkRateCRa'

        if (spikeNumdf_temp.trialNum.values==baselineFR.trialNum.values).all():
           spikeNumdf_temp['baselineFR'] = baselineFR['spkRate_baseline'].values
           spikeNumdf_temp['OnsetCRa'] = baselineFR['spkRate_sig'].values
        else:
            print('mismtach between trial sequences of baselinefr and slidewindowfr!!!!!')
        # fr estimated in slide time windows
        timwinStart = np.arange(estwin[0],estwin[1]-bin,timstep)
        timpntsNum = len(timwinStart) 
        argItems = [(tStart, bin, spikeTimedf, spikeNumdf_temp) for tStart in timwinStart]
        with Pool(processes=cpus) as p:
            # spikeTime is a list: trails X spikeNum
            for column_temp in p.starmap(getSPK4slidewin,argItems):
                spikeNumdf_temp = pd.concat([spikeNumdf_temp,column_temp],axis=1)
        p.close()
        p.join()   

        spktimeseries_df = decodrespLabel2str(spikeNumdf_temp)
        spktimeseries_df.rename(columns={'spkRate':'spkRateOri'}, inplace=True)
        spknumcoloumns = ['respLabel','AVoffset', 'snr', 'trialMod', 'snr-shift', 'trialNum','baselineFR',
                          'OnsetCRa','spkRateCRa','spkRateCRa1','spkRateCRa2','spkRateCRa3']\
                            +list(spktimeseries_df.columns)[-timpntsNum:]
    return spktimeseries_df.reset_index(drop=True),spknumcoloumns


def getPSTHdf(timeSamp2Chorus,spikeTime_temp,spikefs,behavefs,labelDictFilter,alignkeys,timwin,bin,baselinewin=0.6,binmethod = 'nonoverlapwin',binmethodtimstep=0.001):
    #Get baseline firing rate before chorus onset
    spikeTimedf_temp_baseline = getClsRasterMultiprocess(timeSamp2Chorus,spikeTime_temp,spikefs,behavefs,\
                                        labelDictFilter['chorusOnsetInd'],\
                                        labelDictFilter['chorusOnsetInd'] ,\
                                        labelDictFilter['JSOnsetIndwithVirtual'],\
                                        [-baselinewin,baselinewin],\
                                        labelDictFilter) 
    spikeNumdf_temp_baseline_raw=countTrialSPKs(spikeTimedf_temp_baseline,estwin='BlNSig',winTim=[-baselinewin,baselinewin])# baseline fr estimated 600ms before chorus onset                                                 
    #Get spktime during auditory signal
    spikeTimedf_temp_ori = getClsRasterMultiprocess(timeSamp2Chorus,spikeTime_temp,spikefs,behavefs,\
                                        labelDictFilter['chorusOnsetInd'],\
                                        labelDictFilter[alignkeys] ,\
                                        labelDictFilter['JSOnsetIndwithVirtual'],\
                                        [timwin[0],timwin[1]],\
                                        labelDictFilter,labelDictFilter['JSOnsetIndwithVirtual'])   # get spiketime saved in df                                                         

    getclsInput = [alignkeys,timeSamp2Chorus,spikeTime_temp,spikefs,behavefs,labelDictFilter]
    binedFRxTim_df,spknumcoloumns=getbindFRxtime(spikeTimedf_temp_ori,[timwin[0],timwin[1]],spikeNumdf_temp_baseline_raw,bin,binmethod,binmethodtimstep,getclsInput) # fr estimated between cooonset and jsonset, or 300ms before cooonset in long coonset delay trials

    return binedFRxTim_df,spknumcoloumns

def glmfit(spikenum_df,DVcol,IVcolList,familystr):
    # add intercet in linear regression formula
    spikenum_df = spikenum_df.sort_values(by=IVcolList)
    spikenum_df['const'] = np.ones((spikenum_df.shape[0],1))
    #dtype object to float
    for col in DVcol+IVcolList:
        if spikenum_df[col].dtype=='object':
            try:
                spikenum_df[col] = spikenum_df[col].apply(lambda x: np.array(x, dtype=float))
            except ValueError:
                spikenum_df[col] = pd.factorize(spikenum_df[col])[0]

    # Instantiate a poisson family model with the default (log) link function.
    if familystr=='poisson':
        glm_model = sm.GLM(spikenum_df[DVcol], spikenum_df[['const']+IVcolList] , family=sm.families.Poisson(),missing='drop')
    if familystr == 'gaussian':
        glm_model = sm.GLM(spikenum_df[DVcol], spikenum_df[['const']+IVcolList] , family=sm.families.Gaussian(link=sm.families.links.Identity()),missing='drop')
 
    glm_results = glm_model.fit(max_iter=1000, tol=1e-6, tol_criterion='params')

    # print(glm_results.summary())
    coef = pd.DataFrame(glm_results.params).transpose()
    colnames = ['coef_'+col for col in coef.columns]
    coef = coef.rename(columns=dict(zip(coef.columns,colnames)))
    pval = pd.DataFrame(glm_results.pvalues).transpose()
    colnames = ['pval_'+col for col in pval.columns]
    pval = pval.rename(columns=dict(zip(pval.columns,colnames))) 
    evalparam = pd.DataFrame({'aic':glm_results.aic,'bic':glm_results.bic_llf},index=[0])
    # evalparam = pd.DataFrame({'aic':[0],'bic':[0]},index=[0])
    return coef, pval, evalparam

def detrendLSR(tri,snippets):
    # use least square regression to detrend snippet
    # Step 1: Fit the linear model
    A = np.vstack([tri, np.ones(len(tri))]).T  # Design matrix
    m, c = np.linalg.lstsq(A, snippets, rcond=None)[0]  # Solve least squares
    # Step 2: Calculate the trend
    trend = m * tri + c
    # Step 3: Detrend the data
    detrended = snippets - trend
    # get SD of data in this snippets
    DTsnippetsSD = np.std(detrended,ddof=1)
    return detrended,DTsnippetsSD

def getDFAslope(trialorderedSPKrate):
#DFA to quantify the self-similarity in baseline spike counts on the scale of trials
    #cumulative sum of the spike counts during time window of interest. aka: signal profile in https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2012.00450/full
    SPKrate_cumsum = np.cumsum(trialorderedSPKrate)
    # partition signal profile to trial window sizes range from 4 to full block trials, with half window overlap
    triwinSize = np.unique(np.round(np.logspace(np.log10(4),np.log10(len(SPKrate_cumsum)),num=20,base=10)).astype(int))
    triwinsize_SDmean = []
    for triwinsize_temp in triwinSize:
        # print('trial windsize '+str(triwinsize_temp))       
        DTsnippetsSD = []
        for triSt_temp in np.arange(0,len(SPKrate_cumsum),int(triwinsize_temp/2)):
            triEnd_temp = triSt_temp+triwinsize_temp
            if triEnd_temp<=len(SPKrate_cumsum):
                snippets_wintemp = SPKrate_cumsum[triSt_temp:triEnd_temp]
                DTsnippetsSD.append(detrendLSR(np.arange(triSt_temp,triEnd_temp,1),snippets_wintemp)[1])
        triwinsize_SDmean.append(np.mean(DTsnippetsSD))
    # mean standard deviations were regressed linearly against the logarithmically scaled trial windows
    model = LinearRegression().fit(np.log10(triwinSize).reshape(-1, 1), np.log10(triwinsize_SDmean))
    pridSD = model.predict(np.log10(triwinSize).reshape(-1, 1))
    # get DFA alpha value
    return model.coef_[0],list(np.log10(triwinSize)),list(np.log10(triwinsize_SDmean)),list(pridSD)


def sampBalanceGLM(spikeNumdf_temp_raw,GLM_IV_list,seeds=42,method='upsample',samples=100):
    indexpick = []
    grouped = spikeNumdf_temp_raw.groupby(by=GLM_IV_list)
    group_indices = {key: grouped.groups[key].tolist() for key in grouped.groups}
    # upsample group trials by adding randomsample with replacement to the original trials
    if method=='upsample':
        maxtrials = max(len(lst) for lst in group_indices.values())   
        spikeNumdf_temp = pd.DataFrame()
        for gg, (gkey,glist) in enumerate(group_indices.items()):
            rng = np.random.default_rng(seeds+gg*10)
            addsampleInd = rng.choice(glist,size=maxtrials-len(glist),replace=True)
            # print(addsampleInd)
            spikeNumdf_temp_sub = pd.concat((spikeNumdf_temp_raw.iloc[glist,:],spikeNumdf_temp_raw.iloc[addsampleInd,:]))
            spikeNumdf_temp = pd.concat((spikeNumdf_temp,spikeNumdf_temp_sub))
            indexpick = indexpick+glist+list(addsampleInd)
    elif method=='downsample':
        mintrials = min(len(lst) for lst in group_indices.values())
        spikeNumdf_temp = pd.DataFrame()
        for gg, (gkey,glist) in enumerate(group_indices.items()):
            rng = np.random.default_rng(seeds+gg*10)
            addsampleInd = rng.choice(glist,size=mintrials,replace=False)
            spikeNumdf_temp = pd.concat((spikeNumdf_temp,spikeNumdf_temp_raw.iloc[addsampleInd,:])) 
            indexpick = indexpick+list(addsampleInd)  
    elif method == 'samplewithreplacement':
        trials = samples
        spikeNumdf_temp = pd.DataFrame()
        for gg, (gkey,glist) in enumerate(group_indices.items()):
            rng = np.random.default_rng(seeds+gg*10)
            addsampleInd = rng.choice(glist,size=trials,replace=True)
            spikeNumdf_temp = pd.concat((spikeNumdf_temp,spikeNumdf_temp_raw.iloc[addsampleInd,:])) 
            indexpick = indexpick+list(addsampleInd)                    
    return spikeNumdf_temp.reset_index(drop=True),indexpick

def getROC(x,y):
    critlist = np.arange(0,x.max(),0.5)
    pHit = []
    pFA = []
    for cri_temp in critlist:
        pHit.append(np.sum(x[np.where(y==1)[0]]>cri_temp)/len(np.where(y==1)[0]))
        pFA.append(np.sum(x[np.where(y==0)[0]]>cri_temp)/len(np.where(y==0)[0]))
    roc = np.abs(np.trapz(pHit,x=pFA))
    return roc

def findcriteria(fr_catch_temp_sam,fr_sig_temp_sam):
    # assuming fr_catch_temp_sam,fr_signal_temp_sam have same number of trials
    if np.mean(fr_sig_temp_sam)>np.mean(fr_catch_temp_sam):
        flipflag = 0#sig>catch
        a = fr_sig_temp_sam.copy() 
        b = fr_catch_temp_sam.copy()
    else:
        flipflag = 1#sig<catch
        b = fr_sig_temp_sam.copy()
        a = fr_catch_temp_sam.copy()

    Index = []
    sumhitrate = []
    # Range = np.arange(np.concatenate((fr_catch_temp_sam,fr_sig_temp_sam)).min(),np.concatenate((fr_catch_temp_sam,fr_sig_temp_sam)).max(),1) 
    Range = np.sort(np.unique(np.round(np.concatenate((fr_catch_temp_sam,fr_sig_temp_sam)),decimals=2)),kind='mergesort') 
    for crite in Range:
        fooa=len(np.where(a>=crite)[0])/len(a)
        foob=len(np.where(b<=crite)[0])/len(b)
        # print('crite'+str(np.round(crite,decimals=3))+' a:'+str(np.round(fooa,decimals=3))+' b:'+str(np.round(foob,decimals=3))+' (a-b)/(a+b):'+str(np.abs(np.round((fooa-foob)/(fooa+foob),decimals=2))))
        Index.append(np.round((fooa-foob)/(fooa+foob),decimals=2))
        # Index.append(np.abs(fooa-foob)/fooa+np.abs(fooa-foob)/foob)
        sumhitrate.append(fooa+foob)

    idx = np.where(np.abs(Index)==np.abs(Index).min())[0]#index of closest value
    # print('minbias_index')
    # print(idx)
    if len(idx)==1:
        Criterion = Range[idx][0] #closest value
    if len(idx)>1:
       idxnew = np.where(np.abs(sumhitrate)==np.abs(sumhitrate)[idx].max())[0] 
       Criterion = Range[idxnew][0] #closest value
    return Criterion,flipflag 

def estCorrectRate(criteria,flipflag,fr_sig_temp_sub):     
    if flipflag==1:
        hh=np.where(fr_sig_temp_sub<criteria)[0]
    else:
        hh=np.where(fr_sig_temp_sub>criteria)[0]  
    return len(hh)/len(fr_sig_temp_sub)

def NM4oneboot(spikeNumdf_temp,respLabelNumlist,fr_catch,CRstr,bs,DVstr,seeds):
    np.random.seed(seeds)
    # print('test np.random.choice seeds in NM4oneboot parrallel processing: '+str(np.random.choice(1000,2)))

    df_auc1boot = pd.DataFrame()
    for mod in ['a','av']: 
        # get all signal trial fr
        fr_signal = spikeNumdf_temp[spikeNumdf_temp['trialMod'].isin(['av','a']) & spikeNumdf_temp['respLabel'].isin(respLabelNumlist)]
        # balance trials  in each condition 
        fr_signal = sampBalanceGLM(fr_signal.reset_index(drop=True),['trialMod','snr-shift'],seeds,method='samplewithreplacement',samples=200)[0]
        # using same/diff noise for a/av conditions 
        if CRstr == 'aCRnoise':
            fr_catch_temp = fr_catch['a']  
        if CRstr == 'aCRvCRnoise':
            fr_catch_temp = fr_catch[mod]
        if CRstr =='aMISSnoise':
            fr_catch_temp = fr_catch['miss']
        #bootstrap fr_catch to the same number of trials as fr_signal
        fr_catch_bstrape = np.random.choice(fr_catch_temp,fr_signal.shape[0],replace=True)

        # find criteria: using all sig trials vs all catch trials, 100steps in precision,
        criteria,flipflag = findcriteria(fr_catch_bstrape,fr_signal['spkRate'].values)  

        for snr in sorted(fr_signal['snr'].unique()): #min-max snr
            fr_sig_temp = fr_signal[(fr_signal['snr']==snr) & (fr_signal['trialMod']==mod)]['spkRate'].values
            fr_sigraw_temp = fr_signal[(fr_signal['snr']==snr) & (fr_signal['trialMod']==mod)]['spkRateRaw'].values

            if DVstr == 'hitrate':  
                df_auc_temp = pd.DataFrame({'snr':[snr],
                                            'trialMod':[mod],
                                            'hitrate':[estCorrectRate(criteria,flipflag,fr_sig_temp)],
                                            'FRave':np.mean(fr_sig_temp),
                                            'FRvar':np.var(fr_sig_temp),
                                            'FRRAWave':np.mean(fr_sigraw_temp),
                                            'FRRAWvar':np.var(fr_sigraw_temp),
                                            'bstimes':[bs]})
                df_auc1boot = pd.concat([df_auc1boot,df_auc_temp],axis=0)        
    return df_auc1boot
    
def neurMetric(spikeNumdf_temp,DVstr,respLabelNumlist,noiseCR,bootstraptimes,CRstr='aCRnoise'):
    # respLabel decode:  
    # nosoundplayed: [nan]
    # A:  hit[0],miss/latemiss[1],FAa[2],erlyResp[88]
    # AV: hit[0],miss/latemiss[1],FAv[22],erlyResp[88]
    # V:  hit[0],CRv[10,100],FAv[22]
    df_auc = pd.DataFrame()
    fr_catch = {}
    fr_catch['a'] = np.array(spikeNumdf_temp[(spikeNumdf_temp[noiseCR].notna())][noiseCR]) #  baseline firerate during chorus rightbefore vid/coo in cononset delay longer than 1s
    fr_catch['av'] = np.array(spikeNumdf_temp[(spikeNumdf_temp['trialMod']=='v')&(spikeNumdf_temp['respLabel'].isin(['CR']))]['spkRate']) #  baseline firerate during  vid 
    fr_catch['miss'] = np.array(spikeNumdf_temp[(spikeNumdf_temp['trialMod'].isin(['av','a']))&(spikeNumdf_temp['respLabel'].isin(['miss']))]['spkRate'])
    argItems = [(spikeNumdf_temp,respLabelNumlist,fr_catch,CRstr,bs,DVstr,seeds) for bs,seeds in zip(range(bootstraptimes),np.random.choice(range(10000), size=bootstraptimes, replace=False))]
    df_auc = pd.DataFrame()
    with Pool(processes=cpus) as p:
        # spikeTime is a list: trails X spikeNum
        for df_auc1boot in p.starmap(NM4oneboot,argItems):
            df_auc=pd.concat([df_auc,df_auc1boot])
    p.close()
    p.join()

    df_auc_meanBS = df_auc.groupby(by=['trialMod','snr'])[['hitrate','FRave','FRvar','FRRAWave','FRRAWvar']].mean().reset_index()

    return df_auc_meanBS

def calculate_pearsonr_with_warning(x, y):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # To catch any warning
        p_value = stats.pearsonr(x, y)[1]  # Pearson's correlation calculation
        # Check if any warnings were raised
        if w:
            warning_msg = str(w[-1].message)           
            print('pearsonrWarning'+warning_msg)
            print(str(x)+str(y))
            return np.nan
        return p_value

def neurometricSigTest(df_auc,DVstr):
    testResDict = {}

    testResDict['x_raw_a'] = df_auc[df_auc['trialMod']=='a'].snr.values   
    testResDict['y_raw_a'] = df_auc[df_auc['trialMod']=='a'][DVstr].values
    testResDict['y_rawfr_a'] = df_auc[df_auc['trialMod']=='a']['FRave'].values
    testResDict['y_rawfr2_a'] = df_auc[df_auc['trialMod']=='a']['FRRAWave'].values

    testResDict['x_fit_a'],testResDict['y_fit_a'],popta_temp,y_fit_discrete_a = applyweibullfit(testResDict['x_raw_a'],testResDict['y_raw_a'],threshx=0.5)


    testResDict['x_raw_av'] = df_auc[df_auc['trialMod']=='av'].snr.values
    testResDict['y_raw_av'] = df_auc[df_auc['trialMod']=='av'][DVstr].values
    testResDict['y_rawfr_av'] = df_auc[df_auc['trialMod']=='av']['FRave'].values
    testResDict['y_rawfr2_av'] = df_auc[df_auc['trialMod']=='av']['FRRAWave'].values
    
    testResDict['x_fit_av'],testResDict['y_fit_av'],poptav_temp,y_fit_discrete_av = applyweibullfit(testResDict['x_raw_av'],testResDict['y_raw_av'],threshx=0.5)
    
    # stats-test 
    if np.any(np.isnan(y_fit_discrete_a)):
        print('nan in y_fit_discrete_a')
        print(y_fit_discrete_a)
        testResDict['pvalslopeA'] = np.nan   
    else:
        testResDict['pvalslopeA'] = calculate_pearsonr_with_warning(testResDict['y_raw_a'],y_fit_discrete_a)
        
    
    if np.any(np.isnan(y_fit_discrete_av)):
        print('nan in y_fit_discrete_av')
        print(y_fit_discrete_av)
        testResDict['pvalslopeAV'] = np.nan 
    else:
        testResDict['pvalslopeAV'] = calculate_pearsonr_with_warning(testResDict['y_raw_av'],y_fit_discrete_av)
            
    
    testResDict['slopeA'] = popta_temp[0]
    testResDict['slopeAV'] = poptav_temp[0]
    testResDict['threshA'] = popta_temp[1]
    testResDict['threshAV'] = poptav_temp[1]
    return(testResDict)

def neurMetric2(spikeNumdf_temp,DVstr,resplabelfilter=[0,1]):
    # respLabel decode:  
    # nosoundplayed: [nan]
    # A:  hit[0],miss/latemiss[1],FAa[2],erlyResp[88]
    # AV: hit[0],miss/latemiss[1],FAv[22],erlyResp[88]
    # V:  hit[0],CRv[10,100],FAv[22]
    df_auc = pd.DataFrame()
    for mod in ['a','av']: 
        fr_catch_temp1 = np.array(spikeNumdf_temp[(spikeNumdf_temp['spkRateCRa'].notna())]['spkRateCRa']) #  baseline firerate during chorus rightbefore vid/coo
        fr_catch_temp2 = np.array(spikeNumdf_temp[(spikeNumdf_temp['trialMod']=='v')]['spkRate']) #  baseline firerate during  vid
        # zcatch_a = np.mean(fr_catch_temp1)/np.std(fr_catch_temp1)
        # zcatch_v = np.mean(fr_catch_temp2)/np.std(fr_catch_temp2)
        for snr in sorted(spikeNumdf_temp['snr'].dropna().unique()): #min-max snr
            # print(mod+'--neuromentric measurement SNR '+str(snr))
            df_auc_temp = pd.DataFrame()
            fr_sig_temp = np.array(spikeNumdf_temp[(spikeNumdf_temp['snr']==snr) & (spikeNumdf_temp['trialMod']==mod) & spikeNumdf_temp['respLabel'].isin(resplabelfilter)]['spkRate'])
            # zsig_temp = np.mean(fr_sig_temp)/np.std(fr_sig_temp)
            df_auc_temp['snr'] = [snr]
            df_auc_temp['trialMod'] = [mod]
            df_auc_temp[DVstr]=[(2*(np.mean(fr_sig_temp)-np.mean(fr_catch_temp1)))/(np.std(fr_catch_temp1)+np.std(fr_sig_temp))]
            print(mod+' snr'+str(snr)+' catch trials:'+str(len(fr_catch_temp1)) +' sig trials:'+str(len(fr_sig_temp)))
            # df_auc_temp[DVstr]=zsig_temp-zcatch_a]
            df_auc = pd.concat((df_auc,df_auc_temp))
    return df_auc

def neurometricSigTest2(df_auc_temp,DVstr):
    slopeA = []
    slopeAV = []
    x_raw_a = []
    y_raw_a = []
    x_raw_av = []
    y_raw_av = []    
    testResDict = {}
    # print('curvefitting rpts #'+str(rt))
    df_auc_temp_a = df_auc_temp[df_auc_temp['trialMod']=='a'].copy()
    x_a = df_auc_temp_a['snr'].values
    y_a = df_auc_temp_a[DVstr].values
    testResDict['x_raw_a'] = x_a
    testResDict['y_raw_a'] = y_a
    testResDict['x_fit_a'],testResDict['y_fit_a'],popta_temp,y_fit_discrete_a= applylogisticfit(x_a,y_a)
    slopeA.append(popta_temp[0])

    df_auc_temp_av = df_auc_temp[df_auc_temp['trialMod']=='av'].copy()
    x_av = df_auc_temp_av['snr'].values
    y_av = df_auc_temp_av[DVstr].values
    testResDict['x_raw_av'] = x_av
    testResDict['y_raw_av'] = y_av
    testResDict['x_fit_av'],testResDict['y_fit_av'],poptav_temp,y_fit_discrete_av = applylogisticfit(x_av,y_av)
    slopeAV.append(poptav_temp[0])
    x_raw_a.append(x_a)
    y_raw_a.append(y_a)
    x_raw_av.append(x_av)
    y_raw_av.append(y_av)

    # pearson correlation test
    testResDict['pvalslopeA'] = stats.pearsonr(y_a,y_fit_discrete_a)[1]
    testResDict['pvalslopeAV'] = stats.pearsonr(y_av,y_fit_discrete_av)[1]
    # max dprime track
    testResDict['maxdprime_A'] = y_a.max()
    testResDict['maxdprime_AV'] = y_av.max()

    testResDict['slopeA'] = popta_temp[0]
    testResDict['slopeAV'] = poptav_temp[0]
    testResDict['threshA'] = popta_temp[1]
    testResDict['threshAV'] = poptav_temp[1]
    return(testResDict)


def AVmodulation(spikeNumdf,cond = ['trialMod','snr'],bsSamp=100):
    def bootstrapDF(spikeNumdf,cond = ['trialMod','snr'],shuffleLab=False,sample=100):
        spikeNumdf_BS = pd.DataFrame()
        ## bootstrap samples in each condition and form a dataframe 
        for mod in spikeNumdf['trialMod'].unique():
            spikeNumdf_temp = spikeNumdf[spikeNumdf['trialMod']==mod].copy()
            #mix all snrs within modality condition or if the present is V trialMod
            if len(cond)==1 or all(np.isnan(list(spikeNumdf_temp['snr'].unique()))): 
                spikeNumdf_BS_temp = spikeNumdf_temp.sample(n=sample,replace=True)
                spikeNumdf_BS = pd.concat((spikeNumdf_BS,spikeNumdf_BS_temp))
            else:#separate different snrs in a/av trialMod condition               
                for snr in spikeNumdf_temp['snr'].unique():
                    spikeNumdf_temp_snr = spikeNumdf_temp[spikeNumdf_temp['snr']==snr].copy()
                    spikeNumdf_BS_temp = spikeNumdf_temp_snr.sample(n=sample,replace=True)
                    spikeNumdf_BS = pd.concat((spikeNumdf_BS,spikeNumdf_BS_temp))
        ## shuffle a/av trialMod labels
        if shuffleLab:
            spikeNumdf_BS_v = spikeNumdf_BS[spikeNumdf_BS['trialMod']=='v'].copy()
            spikeNumdf_BS_sig = spikeNumdf_BS[spikeNumdf_BS['trialMod'].isin(['a','av'])].copy()
            spikeNumdf_BS_sig['trialMod'] = random.sample(list(spikeNumdf_BS_sig['trialMod']),len(list(spikeNumdf_BS_sig['trialMod'])))
            spikeNumdf_BS = pd.concat((spikeNumdf_BS_v,spikeNumdf_BS_sig))
        return spikeNumdf_BS
    
    def getAVmod(spikeNumdf_all,cond):
        spikeNumdf = spikeNumdf_all[spikeNumdf_all['trialMod'].isin(['a','av'])].copy()# remove v trials
        AVmod = pd.DataFrame()
        frmean = spikeNumdf.groupby(cond)['spkRate'].mean().reset_index()
        frvar = spikeNumdf.groupby(cond)['spkRate'].var().reset_index() # nan when condition has only one trial 
        if len(cond)==1:
            AVmod_temp = (frmean[frmean['trialMod']=='av']['spkRate'].values-frmean[frmean['trialMod']=='a']['spkRate'].values)/\
                    ((frvar[frvar['trialMod']=='av']['spkRate'].values+frvar[frvar['trialMod']=='a']['spkRate'].values)/2)**0.5
            AVmod = pd.concat((AVmod,pd.DataFrame({'avMod':np.abs(AVmod_temp),'enhance/inhibit':np.sign(AVmod_temp)})))
        
        else:
            for snr in np.sort(spikeNumdf['snr'].unique()):
                AVmod_temp = (frmean[(frmean['trialMod']=='av') & (frmean['snr']==snr)]['spkRate'].values\
                              -frmean[(frmean['trialMod']=='a') & (frmean['snr']==snr)]['spkRate'].values)/\
                            ((frvar[(frvar['trialMod']=='av') & (frvar['snr']==snr)]['spkRate'].values\
                            +frvar[(frvar['trialMod']=='a') & (frvar['snr']==snr)]['spkRate'].values)/2)**0.5               
                AVmod = pd.concat((AVmod,pd.DataFrame({'snr':snr,'avMod':np.abs(AVmod_temp),'enhance/inhibit':np.sign(AVmod_temp)})))
        return AVmod
    
    # get avmod shuffle label 
    df_AVmodshuffle =  pd.DataFrame()
    bsRPT = 500
    for bst in range(bsRPT):
        print('bootstrap+shuffle: '+str(bst))
        spikeNumdf_BS_temp = bootstrapDF(spikeNumdf,cond = cond,shuffleLab=True,sample=bsSamp)
        df_AVmodshuffle = pd.concat((df_AVmodshuffle,getAVmod(spikeNumdf_BS_temp,cond)))
    # get avmod
    df_AVmod =getAVmod(spikeNumdf,cond)
    # compare sig
    df_AVmod_siglist = []
    if 'snr' in cond:
        for snr in df_AVmod.snr.unique():
            Percentile95 = np.percentile(df_AVmodshuffle[df_AVmodshuffle['snr']==snr]['avMod'],95,axis=0) 
            if len(df_AVmod[df_AVmod['snr']==snr]['avMod'].values)>0 and not np.isnan(df_AVmod[df_AVmod['snr']==snr]['avMod'].values):
                if df_AVmod[df_AVmod['snr']==snr]['avMod'].values>Percentile95:
                    df_AVmod_siglist.append(1)
                else:
                    df_AVmod_siglist.append(0)
            else:
                df_AVmod_siglist.append(np.nan)
    else:
        Percentile95 = np.percentile(df_AVmodshuffle['avMod'],95,axis=0) 
        if len(df_AVmod['avMod'].values)>0 and not np.isnan(df_AVmod['avMod'].values):
            if df_AVmod['avMod'].values>Percentile95:
                df_AVmod_siglist.append(1)
            else:
                df_AVmod_siglist.append(0)
        else:
            df_AVmod_siglist.append(np.nan)       
        
    df_AVmod['avModSig'] = df_AVmod_siglist
    return df_AVmod

def estiAVmodindex(snra,fra_raw,fra_zs,fca,snrav,frav_raw,frav_zs,fcav):
    #fra and frav should save at the same snr order
    modind_df = pd.DataFrame()
    for ii,snr in enumerate(snra):
        modind_df = pd.concat((modind_df,pd.DataFrame({'snr':[snr]*2,
                                                       'AVmodInd_fromraw':[(frav_raw[ii]-fra_raw[ii])/(frav_raw[ii]+fra_raw[ii])]*2,
                                                       'AVmodInd_fromzs':[(frav_zs[ii]-fra_zs[ii])/(frav_zs[ii]+fra_zs[ii])]*2,
                                                       'mod':['a','av'],
                                                       'raw_fr':[fra_raw[ii],frav_raw[ii]],
                                                       'zscore_fr':[fra_zs[ii],frav_zs[ii]],
                                                       'FractionCorrect':[fca[ii],fcav[ii]]})))
    # modind_df['AVmodInd_fromraw_mean'] =  (np.mean(frav_raw)-np.mean(fra_raw))/(np.mean(frav_raw)+np.mean(fra_raw))
    return modind_df


def estNoiseCorr(spkdf1,spkdf2,unitpairs,spkCol,cond=['trialMod','snr']):
    NCdf = pd.DataFrame()
    spkdf1_group = spkdf1.groupby(cond)
    spkdf2_group = spkdf2.groupby(cond)
    grou_keys = spkdf1_group.groups.keys()
    for key in grou_keys:
        group1 = spkdf1_group.get_group(key).sort_values(['trialNum'],kind='mergesort')
        group2 = spkdf2_group.get_group(key).sort_values(['trialNum'],kind='mergesort')
        if list(group1['trialNum'].values) == list(group2['trialNum'].values):
            for tim,col in enumerate(spkCol):
                NCdfz_temp = pd.DataFrame()
                spk1_temp_all = group1[col].values
                spk2_temp_all = group2[col].values
                # handle nans in noise fr 
                spk1_temp = np.ma.masked_array(spk1_temp_all, mask=np.isnan(spk1_temp_all))
                spk2_temp = np.ma.masked_array(spk2_temp_all, mask=np.isnan(spk2_temp_all))

                for cc in cond:
                    NCdfz_temp[cc]= [group1[cc].values[0]]
                NC = np.ma.corrcoef(spk1_temp,spk2_temp)[0][1]
                NCdfz_temp['time'] = col
                NCdfz_temp['corrcoef']=NC                
                ## get shuffles nc for 95 percentile
                NC_shuffle = []
                for bts in range(500):
                    NC_shuffle.append(np.ma.corrcoef(random.sample(list(spk1_temp),len(spk1_temp)),\
                                                  list(spk2_temp))[0][1])
                NCdfz_temp['95percentile'] = np.percentile(NC_shuffle,95,axis=0) 
                NCdfz_temp['5percentile'] = np.percentile(NC_shuffle,5,axis=0) 
                if NC>NCdfz_temp['95percentile'].values or NC<NCdfz_temp['5percentile'].values:
                    NCdfz_temp['sig'] = 1
                else:
                    NCdfz_temp['sig'] = -1
                NCdfz_temp['NeuPairs'] = unitpairs[0]+'&'+unitpairs[1]
                NCdf = pd.concat((NCdf,NCdfz_temp))   
    return NCdf

def sampBalanCond(df,cond):
    df_group_counts = df.groupby(cond)[cond].size().reset_index(name='count')
    sampNum = df_group_counts['count'].values.min()
    selected_rows = df.groupby(cond).apply(lambda x: x.sample(sampNum)).reset_index(level=list(range(len(cond))),drop=True).index.tolist()
    return selected_rows

def extract_between_tts(s):
    match = re.search(r'\*(.*?)\*', s) # extract content bewteen * in a string
    return match.group(1) if match else None

def getdPrime(fitacc_nNueron_temp,trialLab,printflag,snrsep):
    def dprimeCal(hitRate, FARate):
        if hitRate==1:
            hitRate = 0.9999
        if hitRate==0:
            hitRate = 0.0001 
        if FARate==1:
            FARate = 0.9999
        if FARate==0:
            FARate = 0.0001  
        dPrime = norm.ppf(hitRate)-norm.ppf(FARate)
        return dPrime 
    
    def ProbCorctCal(hitRate, FARate):
        if hitRate==1:
            hitRate = 0.9999
        if hitRate==0:
            hitRate = 0.0001 
        if FARate==1:
            FARate = 0.9999
        if FARate==0:
            FARate = 0.0001  
        probCorct = norm.cdf((norm.ppf(hitRate)-norm.ppf(FARate))/2)
        return probCorct 
    
    behavdf = pd.DataFrame()
    ylabelStrlist = [extract_between_tts(ii) for ii in trialLab]
    sig_index = [index for index,value in enumerate(ylabelStrlist) if 'sig' in value]
    noise_index = [index for index,value in enumerate(ylabelStrlist) if 'noise' in value]
    # a trials then av trials
    for mod in [['a_'],['av_','v_']]:
        behavdf_temp = pd.DataFrame()
        mod_index = [index for index,value in enumerate(ylabelStrlist) if any(s in value for s in mod)]
        sig_index_mod = list(set(sig_index)&set(mod_index))
        noise_index_mod= list(set(noise_index)&set(mod_index))

        fitacc_nNueron_sig = fitacc_nNueron_temp[sig_index_mod]
        fitacc_nNueron_noise = fitacc_nNueron_temp[noise_index_mod]
        ylabelSig = [ylabelStrlist[kk] for kk in sig_index_mod]

        if printflag==1:
            print('mod:'+mod[0]+' signan: '+str(len(np.where(np.isnan(fitacc_nNueron_sig))[0]))+'/'+str(len(fitacc_nNueron_sig))
                  +' noisenan:'+str(len(np.where(np.isnan(fitacc_nNueron_noise))[0]))+'/'+str(len(fitacc_nNueron_noise)))

        # remove nan from fitacc if there is any
        fitacc_nNueron_sig_nonan = fitacc_nNueron_sig[~np.isnan(fitacc_nNueron_sig)]
        fitacc_nNueron_noise_nonan = fitacc_nNueron_noise[~np.isnan(fitacc_nNueron_noise)]            
        fa_temp = len(np.where(fitacc_nNueron_noise_nonan==0)[0])/len(fitacc_nNueron_noise_nonan)   
            
        if len(snrsep)==0:
            hitrate_temp = len(np.where(fitacc_nNueron_sig_nonan==1)[0])/len(fitacc_nNueron_sig_nonan)
            behavdf_temp['mod'] = [mod[0][0:-1].upper()]
            behavdf_temp['hitrate0'] = [hitrate_temp]
            behavdf_temp['hitrate'] = [ProbCorctCal(hitrate_temp,fa_temp)]
            behavdf_temp['dprime'] = [dprimeCal(hitrate_temp,fa_temp)]
            behavdf_temp['nonantrials_sig'] = [len(fitacc_nNueron_sig_nonan)]
            behavdf_temp['nonantrials_noise'] = [len(fitacc_nNueron_noise_nonan)]
            behavdf = pd.concat((behavdf,behavdf_temp))
        if len(snrsep)>0:
            for cc in list(set(ylabelSig)):
                behavdf_temp_temp = pd.DataFrame()
                ccind = [ind for ind,val in enumerate(ylabelSig) if cc in val]
                fitacc_sig_temp = fitacc_nNueron_sig[ccind]
                hitrate_temp = len(np.where(fitacc_sig_temp[np.where(~np.isnan(fitacc_sig_temp))[0]]==1)[0])/np.sum(~np.isnan(fitacc_sig_temp))
                behavdf_temp_temp['mod'] = [mod[0][0:-1].upper()]
                behavdf_temp_temp['snr'] = [[float(re.search(r'_(\d+\.\d+)_', item).group(1)) for item in [cc]][0]-20]
                behavdf_temp_temp['hitrate0'] = hitrate_temp
                behavdf_temp_temp['hitrate'] = [ProbCorctCal(hitrate_temp,fa_temp)]
                behavdf_temp_temp['dprime'] = [dprimeCal(hitrate_temp,fa_temp)]
                behavdf_temp_temp['nonantrials_sig'] = [len(fitacc_sig_temp[~np.isnan(fitacc_sig_temp)])]
                behavdf_temp_temp['nonantrials_noise'] = [str(len(fitacc_nNueron_noise_nonan))+'/'+str(len(list(set(ylabelSig))))]              
                behavdf = pd.concat((behavdf,behavdf_temp_temp))
    
    return behavdf
