import mat73
import time
from datetime import timedelta
import h5py
import random
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import sem,tstd
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt 
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler,MaxAbsScaler
from sklearn import linear_model
import seaborn as sns
cpus = 10

# from mne.viz import plot_filter
# try:
#     import mne
#     from mne.time_frequency import tfr_array_morlet as tfmorlet
#     from mne.stats import f_threshold_mway_rm, f_mway_rm, fdr_correction,permutation_cluster_test
#     from mne.baseline import rescale
# except:
#     print('fail to load mne package')

# morlet transform parameters
numFreq = 80
morletFreq = np.logspace(np.log10(1),np.log10(150),numFreq)
morletCyc = np.logspace(np.log10(2),np.log10(30),numFreq)

def cutEEG(EEGtimeseriesList,timeRange,fs,IndofInterest,tfAnlyz,seriesFeature,baselineCorrect):
    timLen = max([1,len(np.arange(int(np.fix(timeRange[0]*fs)),int(np.fix(timeRange[1]*fs))))])
    if len(tfAnlyz)>0:
        EEGseg = np.empty((0,numFreq,timLen))
        nanseg = np.empty((1,numFreq,timLen))
        nanseg[:] = np.nan
    if len(seriesFeature)>0:
        EEGseg = np.empty((0,1,timLen))
        nanseg = np.empty((1,1,timLen))
        nanseg[:] = np.nan 

    for ii, (EEGtemp, Indtemp) in enumerate(zip(EEGtimeseriesList,IndofInterest)):                 
        if not np.isnan(Indtemp):
            timpnts = np.arange(0,len(EEGtemp[0]))/fs
            startpnt = int(Indtemp+np.fix(timeRange[0]*fs)) 
            endpnt = startpnt+timLen  

            EEGtemp = np.array(EEGtemp).reshape(1,1,-1) #1 X 1 X time

            timpnts = timpnts-timpnts[int(Indtemp[0])] # align timepnts 0 to IndofInterest
            #  trial by trial baseline correction: zscore by timeseries 0.3s before the cut timewindow 
            if baselineCorrect:
                EEGtemp = rescale(EEGtemp,timpnts,[timeRange[0]-0.3,timeRange[0]-1/fs],mode='zscore',verbose='warning')
            
            # if need to calculate 2D power/ phase 
            if len(tfAnlyz)>0: 
                # array shape: comptfr requires input trialXchanXtim
                EEG_tf_temp = np.squeeze(comptfr(EEGtemp, morletFreq, sfreq=fs.tolist(), method='morlet',\
                                            n_cycles=morletCyc, zero_mean=None, time_bandwidth=None,\
                                            use_fft=True, decim=1, output=tfAnlyz, n_jobs=6,\
                                            verbose=None))
                EEGtemp = np.expand_dims(EEG_tf_temp[:,startpnt:endpnt],axis=0)  #1 X n_freqs X n_times                               
            # if just raw time series, instantaneous phase/power
            if len(seriesFeature)>0:
                if seriesFeature == 'raw':
                    EEGtemp = EEGtemp[0][0][startpnt:endpnt].reshape((1,1,-1)) # 1 X 1 x time array
                    EEGtemp = NormInput(EEGtemp)
                if seriesFeature == 'power':
                    EEG_anly_temp = signal.hilbert(EEGtemp,axis=-1)
                    EEG_powr_temp = np.square(np.abs(EEG_anly_temp))
                    EEGtemp = EEG_powr_temp[0][0][startpnt:endpnt].reshape((1,1,-1)) 
                    EEGtemp = EEGtemp/EEGtemp.max()
                    # EEGtemp = 10*np.log(EEGtemp).reshape((1,1,-1)) # 1 X 1 x time array
                if seriesFeature == 'phase':
                    EEG_anly_temp = signal.hilbert(EEGtemp,axis=-1)
                    EEG_phase_temp = np.angle(EEG_anly_temp)
                    EEGtemp = EEG_phase_temp[0][0][startpnt:endpnt].reshape((1,1,-1)) # 1 X 1 x time array
                if seriesFeature == 'phase_unwrap':
                    EEG_anly_temp = signal.hilbert(EEGtemp,axis=-1)
                    EEG_phase_temp = np.angle(EEG_anly_temp)
                    EEGtemp = np.unwrap(EEG_phase_temp[0][0][startpnt:endpnt]).reshape((1,1,-1)) # 1 X 1 x time array
                                            
                if seriesFeature == 'instFreq':
                    EEG_anly_temp = signal.hilbert(EEGtemp,axis=-1)
                    EEG_phase_temp = np.diff(np.unwrap(np.angle(EEG_anly_temp)))/(2*np.pi)*fs
                    EEGtemp = EEG_phase_temp[0][0][startpnt:endpnt].reshape((1,1,-1)) # 1 X 1 x time array
                    
                if seriesFeature == 'complex':
                    EEG_anly_temp = signal.hilbert(EEGtemp,axis=-1)
                    EEGtemp = EEG_anly_temp[0][0][startpnt:endpnt].reshape((1,1,-1)) # 1 X 1 x time array

            EEGseg = np.concatenate((EEGseg,EEGtemp),axis=0)
        else:
            print('trial '+str(ii)+' has nan indofinterest')
            EEGseg = np.concatenate((EEGseg,nanseg),axis=0)  
    return np.squeeze(EEGseg)

def cutLFP1(LFPtimeseriesArray,timeRange,fs,IndofInterest,seriesFeature,baselineCorrect):
    # cut time series of lfp (measurements) at all channels
    timLen = max([1,len(np.arange(int(np.fix(timeRange[0]*fs)),int(np.fix(timeRange[1]*fs))))])
    LFPseg = []
    nanseg = np.empty((24,timLen))
    nanseg[:] = np.nan 

    for ii, Indtemp in enumerate(IndofInterest):  
        LFPtemp = np.squeeze(LFPtimeseriesArray[:,ii,:].copy()) # chan X time    
        if not np.isnan(Indtemp):
            timpnts = np.arange(0,len(LFPtemp[1]))/fs
            startpnt = int(Indtemp+np.fix(timeRange[0]*fs)) 
            endpnt = startpnt+timLen  
            timpnts = timpnts-timpnts[int(Indtemp)] # align timepnts 0 to IndofInterest
            #  trial by trial baseline correction: zscore by timeseries 0.3s before the cut timewindow 
            if baselineCorrect:
                LFPtemp = rescale(LFPtemp,timpnts,[timeRange[0]-0.3,timeRange[0]-1/fs],mode='zscore',verbose='warning',copy=True)             
            # raw time series, instantaneous phase/power
            if seriesFeature == 'raw':
                LFPtemp = LFPtemp[:,startpnt:endpnt].copy() # chan x time array
                LFPtemp = NormInput(LFPtemp)
            if seriesFeature == 'power':
                LFP_anly_temp = signal.hilbert(LFPtemp,axis=-1)
                LFP_powr_temp = np.square(np.abs(LFP_anly_temp))
                LFPtemp = LFP_powr_temp[:,startpnt:endpnt].copy()# chan x time array
                LFPtemp = NormInput(LFPtemp)
            if seriesFeature == 'phase':
                LFP_anly_temp = signal.hilbert(LFPtemp,axis=-1)
                LFP_phase_temp = np.angle(LFP_anly_temp)
                LFPtemp = LFP_phase_temp[:,startpnt:endpnt].copy()# chan x time array
            if seriesFeature == 'phase_unwrap':
                LFP_anly_temp = signal.hilbert(LFPtemp,axis=-1)
                LFP_phase_temp = np.angle(LFP_anly_temp)
                LFPtemp = np.unwrap(LFP_phase_temp[:,startpnt:endpnt].copy())# chan x time array                                        
            if seriesFeature == 'instFreq':
                LFP_anly_temp = signal.hilbert(LFPtemp,axis=-1)
                LFP_phase_temp = np.diff(np.unwrap(np.angle(LFP_anly_temp),axis=-1),axis=-1)/(2*np.pi)*fs
                LFPtemp = LFP_phase_temp[:,startpnt:endpnt].copy()            
            if seriesFeature == 'complex':
                LFP_anly_temp = signal.hilbert(LFPtemp,axis=-1)
                LFPtemp = LFP_anly_temp[:,startpnt:endpnt].copy()

            LFPseg.append(LFPtemp)  
        else:
            print('trial '+str(ii)+' has nan indofinterest')
            LFPseg.append(nanseg)  
    return LFPseg

def cutLFP2(LFPtimeseriesArray,timeRange,fs,IndofInterest,tfAnlyz,baselineCorrect):
    # cut spectrum of lfp (measurements) at all channels
    timLen = max([1,len(np.arange(int(np.fix(timeRange[0]*fs)),int(np.fix(timeRange[1]*fs))))])
    LFPseg = []
    nanseg = np.empty((24,numFreq,timLen))
    nanseg[:] = np.nan
    for ii, Indtemp in enumerate(IndofInterest): 
        # print('process trial #'+str(ii)) 
        LFPtemp = np.squeeze(LFPtimeseriesArray[:,ii,:].copy()) # chan X time    
        if not np.isnan(Indtemp):
            timpnts = np.arange(0,len(LFPtemp[1]))/fs
            startpnt = int(Indtemp+np.fix(timeRange[0]*fs)) 
            endpnt = startpnt+timLen  
            timpnts = timpnts-timpnts[int(Indtemp)] # align timepnts 0 to IndofInterest
            #  trial by trial baseline correction: zscore by timeseries 0.3s before the cut timewindow 
            if baselineCorrect:
                LFPtemp = rescale(LFPtemp,timpnts,[timeRange[0]-0.3,timeRange[0]-1/fs],mode='zscore',verbose='warning',copy=True) 
            
            # calculate 2D power+ phase complex
            # array shape: comptfr requires input trialXchanXtim
            LFP_tf_temp = np.squeeze(tfmorlet(np.expand_dims(LFPtemp,0),fs.tolist(),morletFreq,n_cycles=morletCyc,\
                                              output=tfAnlyz,n_jobs=cpus,verbose='WARNING'))
            LFPtemp = LFP_tf_temp[:,:,startpnt:endpnt].copy()  #chan X n_freqs X n_times                                                     
            LFPseg.append(LFPtemp)  
        else:
            print('trial '+str(ii)+' has nan indofinterest')
            LFPseg.append(nanseg)  
    return LFPseg

def zerophasefilter(timeserieslist,freqband,fs):
    # order=4
    # sos = signal.butter(order, freqband, 'bandpass', fs=fs, output='sos')
    # timeserieslist_filtered=list(signal.sosfiltfilt(sos,np.array(timeserieslist),axis=2)) 
    ## get filter time frequecy response 
    # filter_params = mne.filter.create_filter(np.random.random((1,1000)), fs,
    #                                         l_freq=freqband[0], h_freq=freqband[1])
    # mne.viz.plot_filter(filter_params, fs)
    timeserieslist_filtered = list(mne.filter.filter_data(np.array(timeserieslist),fs,l_freq=freqband[0], h_freq=freqband[1]))              
    return timeserieslist_filtered

def grouptfarray2dict_deprecated(lfpMeasdict,lfpSeg_temp,labelFilterDFchunk,input):
    # sumup phase/power among trials within the same category
    def updatelfpdict(lfpSegdict,lfpSegdict_temp):
        if len(lfpSegdict)>0:
            lfpSegdict_updated = {}
            keyscomb = list(set(lfpSegdict.keys()).union(set(lfpSegdict_temp.keys())))
            for key in keyscomb:
                if key in list(lfpSegdict.keys()) and key in list(lfpSegdict_temp.keys()):
                    lfpSegdict_updated[key] = lfpSegdict[key]+lfpSegdict_temp[key]
                else:
                    try:
                        lfpSegdict_updated[key] = lfpSegdict[key]
                    except KeyError:
                        lfpSegdict_updated[key] = lfpSegdict_temp[key]
        else:
            lfpSegdict_updated = lfpSegdict_temp.copy()
        return lfpSegdict_updated
    
    labgroups = labelFilterDFchunk.groupby(['trialMod','snr-shift','respLabel'])
    dict_temp = {}
    for gpname,gpind in labgroups.groups.items():       
        if input=='power':
            powerarray = np.power(np.abs(lfpSeg_temp[gpind,:,:,:]),2)
            dict_temp[str(gpname)]=np.sum(Norm4DInput(powerarray),axis=0) # normalize power within each trial
            dict_temp[str(gpname)+'trials'] = len(gpind)
        if input=='phase':
            dict_temp[str(gpname)]=np.sum(np.exp(np.angle(lfpSeg_temp[gpind,:,:,:])*1j),axis=0)# normalize amp to 1 before sumup angle
            dict_temp[str(gpname)+'trials'] = len(gpind)

    return updatelfpdict(lfpMeasdict,dict_temp)

def loadTimWindowedEEG(eegfilePathway,eegalignStr,chanStr,timeRange,freqband,tfAnlyz,seriesFeature,norm,baselineCorrect):
    
## load eeg series
    data = mat73.loadmat(eegfilePathway)
    fieldnames = list(data.keys())
    EEGtimeseriesList = data[fieldnames[0]][chanStr]

    ## load synapse time events and convert to eeg sampling frequency 
    synapseEvents = mat73.loadmat(eegfilePathway[0:-24]+'labviewNsynapseBehavStruct.mat')['eventsIndStruct']

    try:
        if all(EEGtimeseriesList==0):
            print('warning: the channel selected is empty')
    except TypeError:
        pass
    
    fs = data[fieldnames[0]]['fs']
    # baseline correction for each trial here?

    # normalize before filtering
    if len(norm)>0:
       EEGtimeseriesList=eval('NormMethod(EEGtimeseriesList).'+norm+'()' )

    # add zero phase  band pass filter
    if len(freqband)==2:
        EEGtimeseriesList = zerophasefilter(EEGtimeseriesList,freqband,fs)

    # cut eegsegments (timeseries/phase/power/spectrogram)
    if eegalignStr=='align2chorus':
            eegSeg = cutEEG(EEGtimeseriesList,timeRange,fs,synapseEvents['chorusOnsetInd'],tfAnlyz,seriesFeature,baselineCorrect)
    if eegalignStr=='align2coo':
            eegSeg = cutEEG(EEGtimeseriesList,timeRange,fs,synapseEvents['cooOnsetInd'],tfAnlyz,seriesFeature,baselineCorrect)
    if eegalignStr=='align2js':
            eegSeg = cutEEG(EEGtimeseriesList,timeRange,fs,synapseEvents['joystickOnsetInd'],tfAnlyz,seriesFeature,baselineCorrect)        
    
    return eegSeg

def loadTimWindowedLFP(LFPfilePathway,labelFilterDF,behavefs,alignStr,timeRange,freqband,tfAnlyz,seriesFeature,filename,baselineCorrect):
    fs = behavefs.copy()# LFP at the same sampling frequency as synapse behave data
    synapseEvents = labelFilterDF[alignStr].values
    lfpSeg = []
    labelFilterDF_new = pd.DataFrame()
    ## load lfp series
    with h5py.File(LFPfilePathway, 'r') as file:
        dataset = file['LFPpreprocTimeseries'] 
        # Determine the size of the dataset
        total_trials = dataset.shape[2] # total trials: chan X Time X trials
        chunk_size = 100  # Adjust this value based on your memory capacity and dataset size        
        # Read the dataset in chunks
        # for i in range(0, 100, chunk_size):
        for i in range(0, total_trials, chunk_size):
            LFPchunk = dataset[:,:,i:i + chunk_size] # chan X Time X trials
            LFPchunk2 = np.swapaxes(LFPchunk,1,2)# chan X trials X Time 
            synapseEventschunk = synapseEvents[i:i + chunk_size] 
            labelFilterDFchunk = labelFilterDF.iloc[i:i + chunk_size,:].reset_index(drop=True) 
            # delete trails with nan respLabel, (sound not played at all)
            nantrials = labelFilterDFchunk.loc[labelFilterDFchunk['respLabel']=='NaN'].index
            LFPchunk2 = np.delete(LFPchunk2,list(nantrials),axis=1)
            synapseEventschunk = np.delete(synapseEventschunk,list(nantrials),axis=0)
            labelFilterDFchunk = labelFilterDFchunk.drop(nantrials).reset_index(drop=True)
            
            # add zero phase band pass filter
            if len(freqband)==2:
                LFPchunk2 = np.array(zerophasefilter(LFPchunk2,freqband,fs)).copy() #chan X trials X Time 
                # cut eegsegments (timeseries)  
                lfpSeg_temp = cutLFP1(LFPchunk2,timeRange,fs,synapseEventschunk,seriesFeature,baselineCorrect)#[trials][chan x time]
                lfpSeg =  lfpSeg+ lfpSeg_temp #[trials][chan x time]

            elif len(tfAnlyz)>0:
                # cut eegsegments (complex spectrogram) 
                lfpSeg_temp = cutLFP2(LFPchunk2,timeRange,fs,synapseEventschunk,tfAnlyz,baselineCorrect)#[trials][chan x freq x time] 
                # # groupsum to save memory and reduce file size
                # Catsumdict = grouptfarray2dict(Catsumdict,np.stack(lfpSeg_temp),labelFilterDFchunk,input) #[CATkey] chan X freq X time [CATkeytrials] #
                # save trial by trial power and phase(angle) for stats_test               
                # normalize power between 0-1 within a trial eliminate the absolute power diff between conditions
                # lfpSegPower = Norm4DInput(np.power(np.abs(lfpSeg_temp),2)).copy() 
                lfpSegPower = np.power(np.abs(lfpSeg_temp),2)#trial X chan X freq X time
                lfpSegPhase = np.angle(lfpSeg_temp)
                # print(lfpSegPower[:,4,4,10:13])
                if i==0:
                    start_time = time.monotonic()
                    # create .h5 file to save phase and power
                    with h5py.File(filename+'_power.h5','w') as filepower:
                        datasetpower = filepower.create_dataset('LFPpowerSeg',lfpSegPower.shape,maxshape=(None,)+lfpSegPower.shape[1:],chunks=True,compression='gzip')
                        datasetpower[:] = lfpSegPower
                    filepower.close()
                    with h5py.File(filename+'_phase.h5','w') as filephase:
                        datasetphase = filephase.create_dataset('LFPphaseSeg',lfpSegPhase.shape,maxshape=(None,)+lfpSegPhase.shape[1:],chunks=True,compression='gzip')
                        datasetphase[:] = lfpSegPhase
                    filephase.close()
                    end_time = time.monotonic()
                    print('time expend to create the two original .h5 files')
                    print(timedelta(seconds=end_time - start_time))

                else:
                    start_time = time.monotonic()
                    with h5py.File(filename+'_power.h5', 'a') as filepower:
                        existing_dataset = filepower['LFPpowerSeg']  
                        new_shape = (existing_dataset.shape[0]+ lfpSegPower.shape[0],)+existing_dataset.shape[1:]
                        existing_dataset.resize(new_shape)
                        existing_dataset[-lfpSegPower.shape[0]:,:,:,:]=lfpSegPower 
                        filepower.close()
                    with h5py.File(filename+'_phase.h5', 'a') as filephase:
                        existing_dataset = filephase['LFPphaseSeg']                    
                        new_shape = (existing_dataset.shape[0]+ lfpSegPhase.shape[0],)+existing_dataset.shape[1:]
                        existing_dataset.resize(new_shape)
                        existing_dataset[-lfpSegPhase.shape[0]:,:,:,:]=lfpSegPhase
                        filephase.close()
                    end_time = time.monotonic()
                    print('time expend to update the two .h5 files')
                    print(timedelta(seconds=end_time - start_time))
                labelFilterDF_new = pd.concat((labelFilterDF_new,labelFilterDFchunk))
    return labelFilterDF_new.reset_index(drop=True)

def genLabels(behavfilePathway,filterCond,catCond):
    data = mat73.loadmat(behavfilePathway)['LabviewBehavStruct']
    fieldnames = list(data.keys())
    # filter out trials index
    filtind = list(range(len(data[fieldnames[0]])))
    for i,(k,v) in enumerate(filterCond.items()):
        if len(filterCond[k])==1:
            ind_temp = [ii for ii, element in enumerate(data[k]) if element==filterCond[k][0]]
            filtind = [x for x in ind_temp if x in filtind] 
        else:
            print('filtercondition has more than one value in a key!')
    # categorize index in each conditions, and generate label vector 
    label = np.empty((len(data[fieldnames[0]]),1))
    label[:] = np.nan
    print('number of trials in each condition of this session:')
    for i,v in enumerate(list(catCond.values())[0]):
        print(i,v)
        ind_temp = [index for index, element in enumerate(data[list(catCond.keys())[0]]) if element==v]
        ind = [x for x in ind_temp if x in filtind] 
        print(str(len(ind)) + ' trials')
        label[ind] = int(i)
    return label  

def genLabelsComb(behavfilePathway,filterCond,catCond_comb):
    data = mat73.loadmat(behavfilePathway)['LabviewBehavStruct']
    fieldnames = list(data.keys())

    # generate all possible catcondition combinations
    filterCond_new = []
    if len(list(catCond_comb.keys())) ==2:
        for v1 in catCond_comb[list(catCond_comb.keys())[0]]:
            for v2 in catCond_comb[list(catCond_comb.keys())[1]]:
                filterCond_new_temp = {}
                filterCond_new_temp[list(catCond_comb.keys())[0]]=[v1]
                filterCond_new_temp[list(catCond_comb.keys())[1]]=[v2]
                filterCond_new_temp.update(filterCond)
                filterCond_new.append(filterCond_new_temp.copy())
                del filterCond_new_temp
        
    if len(list(catCond_comb.keys())) ==1:
        for v1 in catCond_comb[list(catCond_comb.keys())[0]]:
            filterCond_new_temp = {}
            filterCond_new_temp[list(catCond_comb.keys())[0]]=[v1]
            filterCond_new_temp.update(filterCond)
            filterCond_new.append(filterCond_new_temp.copy())
            del filterCond_new_temp
    # filter out trials index in each cat
    label = []
    columstr = []
    for cc, filterCond_new_temp in enumerate(filterCond_new):
        filtind = list(range(len(data[fieldnames[0]])))
        label_temp = np.empty((len(data[fieldnames[0]]),))
        label_temp[:] = np.nan
        for i,(k,v) in enumerate(filterCond_new_temp.items()):           
            ind_temp = [ii for ii, element in enumerate(data[k]) if element==filterCond_new_temp[k][0]]
            filtind = [x for x in ind_temp if x in filtind] 
        label_temp[filtind] = int(cc)
        print(filterCond_new_temp)
        print(str(len(filtind)) + ' trials')
        label.append(label_temp)
        columstr.append(str(filterCond_new_temp))
    label_df = pd.DataFrame(np.array(label).transpose(), columns=columstr) # each column indicates valid trial index in this condition
    return label_df

def genLabelsDF(behavfilePathway):
#     behavfilePathway = './EEGanalyzeElay_221109_labviewNsynapseBehavStruct.mat'
#     filterCond = {'respLabel':[0], 'snr':[10]} # have multipe keys, each key can only has one value
#     catCond = {'trialMod':['a','v','av']} # only one cat condition key, can have multiple values

    data_all = mat73.loadmat(behavfilePathway)
    data = data_all['LabviewBehavStruct'].copy()  
    # allkeys: 'RT2chorus', 'RT2coo', 'RT2vid','chorus', 'chorusOnsetInd',
    #            'coo', 'cooOnsetInd', 'joystick', 'joystickOnsetInd','respLabel', 
    #               'reward', 'rewardOnsetInd', 'vidOnsetInd', 'video'
    data2_temp = data_all['eventsIndStruct'].copy()  
    # data2_temp.pop('fs') 
    # data2_temp.pop('note')
    data2 = pd.DataFrame(data2_temp['RT2coo'])   
  
    fieldnames = list(data.keys())
    labelDF = pd.DataFrame(data)
    # add response string column
    respMap = {}
    for respNum in list(labelDF.respLabel.unique()):
        if respNum==0:
            respMap[respNum] = 'hit'
        elif respNum==1:
                respMap[respNum] = 'miss'
        elif respNum==2 or respNum==88:
            respMap[respNum] = 'FA'          
        else:
            respMap[respNum] = 'none'  
    labelDF['respLabelStr'] =  labelDF['respLabel'].map(respMap)  
    labelDF['rt'] = 1000*data2.iloc[:,0]   # using response time calculated by synapse                     
    return labelDF 



## normalize input time series array between [0,1], X is a 3dim array
def NormInput(X):
    X_norm = np.empty((0,X.shape[1],X.shape[2]))
    for a1 in range(X.shape[0]):
        X_norm_temp = np.empty((1,0,X.shape[2]))
        for a2 in range(X.shape[1]):
            X_temp = X[a1,a2,:].copy()
            X_min=X_temp.min()
            X_max=X_temp.max()
            X_temp=(X_temp-X_min)/(X_max-X_min)
            X_norm_temp = np.concatenate((X_norm_temp,X_temp.reshape([1,1,-1])),axis=1)
        X_norm = np.concatenate((X_norm,X_norm_temp),axis=0)
    return X_norm

## normalize time frequency power of each chan separately, X is a 4dim array trial X chan X freq X time
def Norm4DInput(X):
    X_norm = np.empty((0,X.shape[1],X.shape[2],X.shape[3]))
    for tt in range(X.shape[0]):
        X_norm_temp = np.empty((0,X.shape[2],X.shape[3]))
        for ch in range(X.shape[1]):
            X_temp = X[tt,ch,:,:].copy()
            X_temp=(X_temp-X_temp.min())/(X_temp.max()-X_temp.min())
            X_norm_temp = np.concatenate((X_norm_temp,np.expand_dims(X_temp,axis=0)),axis=0)
        X_norm = np.concatenate((X_norm,np.expand_dims(X_norm_temp,axis=0)),axis=0)    
    return X_norm #trialXchanXfreqXtime array

class NormMethod:
    def __init__(self,xxList):
        self.xxarray = np.squeeze(np.array(xxList)) # list [trial][1][time] array: trial X time
    def applyStandardScaler(self): #NaNs are treated as missing values: disregarded in fit, and maintained in transform
        scaler = StandardScaler()
        scaler.fit(self.xxarray)
        xxarray_norm=scaler.transform(self.xxarray)
        xxList_norm = list(np.expand_dims(xxarray_norm,axis=1))
        return xxList_norm
    def applyMinMaxScaler(self):#NaNs are treated as missing values: disregarded in fit, and maintained in transform
        scaler = MinMaxScaler()
        scaler.fit(self.xxarray)
        xxarray_norm=scaler.transform(self.xxarray)
        xxList_norm = list(np.expand_dims(xxarray_norm,axis=1))

        
## equalize trials in each category, based on the category with the minimum num of trials
def BalanceSamples(y):
    ycat,ycounts=np.unique(y,return_counts=True)
    minnum=np.amin(ycounts)
    ind_choice=[]
    for i in range(len(ycat)):      
        ind_temp=np.where(y==ycat[i])[0]
        ind_choice.extend(np.random.choice(ind_temp,minnum,replace=False))          
    return ind_choice

## equalize trials in each category, based on the category with the maximum num of trials
def BalanceSamples2(y):
    ycat,ycounts=np.unique(y,return_counts=True)
    maxnum=np.amax(ycounts)
    ind_choice=[]
    for i in range(len(ycat)):      
        ind_temp=np.where(y==ycat[i])[0]
        ind_choice.extend(list(ind_temp)+list(np.random.choice(ind_temp,maxnum-len(ind_temp),replace=True)))        
    return ind_choice

# calculate mean of time series responses at each condition separately
def catmean(xx_partial,yy_partial,catNames,seriesFeature):
    raw_mean = dict()
    uniCat = np.unique(yy_partial)
    for cc in range(len(uniCat)):
        ind_temp = np.where(yy_partial==uniCat[cc])[0]
        if seriesFeature == 'phase':
            raw_mean[catNames[cc]] = [np.abs(np.mean(np.exp(xx_partial[ind_temp,:]*1j),axis=0)).tolist(),\
                                     np.zeros((1,xx_partial.shape[1])).tolist()]
        else:
            raw_mean[catNames[cc]] = [np.mean(xx_partial[ind_temp,:],axis=0).tolist(),\
                                        tstd(xx_partial[ind_temp,:],axis=0).tolist()]
    raw_mean['CatnumTrials'] = len(ind_temp)
    return raw_mean

def ITCest(xx_ori,yy_ori,catNames,outputPathway,timeRange,jsonnameStr):
    ITC_tf_cats = {}
    # balance num of samples in each cat
    ind_choice = BalanceSamples(yy_ori)
    if len(ind_choice)>0:
        xx = xx_ori[ind_choice,:,:]
        yy = yy_ori[ind_choice]

        uniCat = np.unique(yy)
        for cc in range(len(uniCat)):
            ind_temp = np.where(yy==uniCat[cc])[0]
            ITC_tf_cats[catNames[cc]] = np.abs(np.mean(np.exp(xx[ind_temp,:,:]*1j),axis=0)).tolist()
        with open(outputPathway+jsonnameStr+'.json', 'w') as fp:
            json.dump({'ITC_tf_cats':ITC_tf_cats,\
                'morletFreq':morletFreq.tolist(),\
                    'timeRange':timeRange,\
                        'trials':len(ind_temp)}, fp)
            print("Done writing JSON data into .json file")  

def lfpITCest(xx_ori,yy_ori,catNames):
    ITC_tf_cats = {}
    # balance num of samples in each cat
    ind_choice = BalanceSamples(yy_ori)
    if len(ind_choice)>0:
        xx = xx_ori[ind_choice,:,:]
        yy = yy_ori[ind_choice]

        uniCat = np.unique(yy)
        for cc in range(len(uniCat)):
            ind_temp = np.where(yy==uniCat[cc])[0]
            ITC_tf_cats[catNames[cc]] = np.abs(np.mean(np.exp(xx[ind_temp,:,:]*1j),axis=0)).tolist()
    return ITC_tf_cats

def ITCest_resample(xx_ori,yy_ori,tfStr, subsamSize,subsamNum):
    totalTrials = len(yy_ori)
    validTrials = totalTrials-yy_ori.isnull().sum() 
    condNum = len(yy_ori.columns)
    if tfStr == 'phase':
        xx_bootstr = np.empty((subsamNum,condNum,xx_ori.shape[1],xx_ori.shape[2]))
    if tfStr == 'power':
        xx_bootstr = np.empty((subsamSize,condNum,xx_ori.shape[1],xx_ori.shape[2])) 

    if min(validTrials)<=subsamSize:
        print('the minimum num of trials among all conditions is <= subsample Size ')
    
    for cc in range(condNum):
        yy_temp = yy_ori[list(yy_ori)[cc]].values
        validTrial_temp = [i for i,v in enumerate(yy_temp) if v==v]
        if tfStr == 'phase' and min(validTrials)>subsamSize:
            for ss in range(subsamNum):                        
                ind_choice_temp = random.sample(validTrial_temp,k=subsamSize)               
                xx_bootstr[ss,cc,:,:] = np.abs(np.mean(np.exp(xx_ori[ind_choice_temp,:,:]*1j),axis=0))
        if tfStr == 'power' and len(subsamNum)==0:
                ind_choice_temp = random.sample(validTrial_temp,k=subsamSize)               
                xx_bootstr[:,cc,:,:] = xx_ori[ind_choice_temp,:,:]

    return xx_bootstr

def ITCstats(xx_bootstr, timeRange, outputPathway,jsonnameStr,factor_levels, effects,effects_all,effect_labels,n_jobs,n_permutations):  
    # anova
    fvals, pvals = f_mway_rm(xx_bootstr, factor_levels, effects=effects)
    fig1, axes = plt.subplots(3, 1, figsize=(6, 6))
    # let's visualize our effects by computing f-images
    for effect, sig, effect_label, ax in zip(fvals, pvals, effect_labels, axes):
        # show naive F-values in gray
        ax.imshow(effect, cmap='gray', aspect='auto', origin='lower',
                extent=[timeRange[0], timeRange[-1], morletFreq[0], morletFreq[-1]])
        # create mask for significant time-frequency locations
        effect[sig >= 0.05] = np.nan
        c = ax.imshow(effect, cmap='autumn', aspect='auto', origin='lower',
                    extent=[timeRange[0], timeRange[-1], morletFreq[0], morletFreq[-1]])
        fig1.colorbar(c, ax=ax)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'Time-locked response for "{effect_label}"')
    fig1.tight_layout()
    fig1.savefig(outputPathway+jsonnameStr+'_anovaSig.png')

    # permutation clustering test and FDR test
    n_replications = xx_bootstr.shape[0]    
    pthresh = 0.05  # set threshold rather high to save some time
    tail = 1  # f-test, so tail > 0

    fig2, axes = plt.subplots(3, 1, figsize=(6, 6))
    fig3, axes3 = plt.subplots(3, 1, figsize=(6, 6))

    for effect_temp, effect_label, ax , ax3, pval_temp in zip(effects_all, effect_labels, axes,axes3,pvals):
        def stat_fun(*args):
            return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                            effects=effect_temp, return_pvals=False)[0]

        # The ANOVA returns a tuple f-values and p-values, we will pick the former.
        f_thresh = f_threshold_mway_rm(n_replications, factor_levels, effect_temp,
                                   pthresh)
        F_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_test(
            list(np.swapaxes(xx_bootstr, 1, 0)), stat_fun=stat_fun, threshold=f_thresh, tail=tail,
            n_jobs=n_jobs, seed=52 ,n_permutations=n_permutations, buffer_size=None,
            out_type='mask')

        good_clusters = np.where(cluster_p_values < .05)[0]
        F_obs_plot = F_obs.copy()
        try:
            F_obs_plot[~clusters[np.squeeze(good_clusters)]] = np.nan
            for f_image, cmap in zip([F_obs, F_obs_plot], ['gray', 'autumn']):
                c = ax.imshow(f_image, cmap=cmap, aspect='auto', origin='lower',
                            extent=[timeRange[0], timeRange[-1], morletFreq[0], morletFreq[-1]])
        except TypeError:
            for f_image, cmap in zip([F_obs], ['gray']):
                c = ax.imshow(f_image, cmap=cmap, aspect='auto', origin='lower',
                            extent=[timeRange[0], timeRange[-1], morletFreq[0], morletFreq[-1]])

        #fdr 
        mask, _ = fdr_correction(pval_temp)
        F_obs_plot2 = F_obs.copy()
        try:
            F_obs_plot2[~mask.reshape(F_obs_plot.shape)] = np.nan
            for f_image, cmap in zip([F_obs, F_obs_plot2], ['gray', 'autumn']):
                c = ax3.imshow(f_image, cmap=cmap, aspect='auto', origin='lower',
                            extent=[timeRange[0], timeRange[-1], morletFreq[0], morletFreq[-1]])
        except TypeError:
            for f_image, cmap in zip([F_obs], ['gray']):
                c = ax3.imshow(f_image, cmap=cmap, aspect='auto', origin='lower',
                            extent=[timeRange[0], timeRange[-1], morletFreq[0], morletFreq[-1]])

        fig3.colorbar(c, ax=ax3)
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Frequency (Hz)')
        ax3.set_title(f'"{effect_label}" cluster-level corrected (p <= 0.05)')

        fig2.colorbar(c, ax=ax)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'"{effect_label}" cluster-level corrected (p <= 0.05)')
    fig2.tight_layout()
    fig2.savefig(outputPathway+jsonnameStr+'_clusterPermutateSig.png')
    fig3.tight_layout()
    fig3.savefig(outputPathway+jsonnameStr+'_clusterFDRSig.png')

def phasDist(xx,yy,timeRange,outputPathway,jsonnameStr,catCond_comb,bins=15,densityLogi=False):
    if len(list(catCond_comb.keys()))==2:
        fig, axes = plt.subplots(len(catCond_comb[list(catCond_comb.keys())[0]]), len(catCond_comb[list(catCond_comb.keys())[1]]), figsize=(6, 12),sharex=True,sharey=True)
        axes = np.ravel(axes)
    if len(list(catCond_comb.keys()))==1:
        fig, axes = plt.subplots(1,1, figsize=(2,2),sharex=True,sharey=True)
        axes = np.ravel(axes)

    for cc, cat_temp in enumerate(yy.columns.values.tolist()):
        y_temp = yy.loc[:,cat_temp]
        validtrialInd = list(y_temp[y_temp.notnull()].index)
        x_temp = xx[validtrialInd,:]
        timepntsNum = xx.shape[1]
        hist_cat = np.empty((bins,timepntsNum))
        for tt in range(timepntsNum):
            [hist_cat[:,tt],binedges_temp]= np.histogram(x_temp[:,tt],bins=bins,range=(-np.pi,np.pi),density=densityLogi)

        c=axes[cc].imshow(hist_cat, cmap='autumn', aspect='auto', origin='lower',
                                extent=[timeRange[0], timeRange[-1], -np.pi, np.pi],vmin=0,vmax=0.4)
        
        fig.colorbar(c, ax=axes[cc])
        # axes[cc].set_xlabel('Time (ms)')
        # axes[cc].set_ylabel('phase')
        title_temp = cat_temp.replace(': ','').replace('\'','').replace(', ','').replace('{','').replace('}','')
        axes[cc].set_title(f'{title_temp}',fontsize=8)
    fig.tight_layout()
    fig.savefig(outputPathway+jsonnameStr+'phaseDistCond.png')

def phasPointDist(data,outputPathway,jsonnameStr,filterCond,snr,bins=16,densityLogi=False):
    fig, axes = plt.subplots(int(len(snr))+1,1, figsize=(4,12),sharex=True,sharey=True)
    # fiter out trials stimulus played
    data = data[data['Stim']==1]
    data_filtered = data[data[list(filterCond.keys())[0]]==list(filterCond.values())[0]]

    Param2comp = 'trialMod'
    compCond = ['a','av']
    catStr = 'snr'
    palette =sns.color_palette("Set2")[0:2]

    data_filtered_coo = data_filtered[data_filtered[Param2comp].isin(compCond)]  
    for cc,(snr_temp,ax_temp)in enumerate(zip(snr,axes)):
        data_filtered_coo_snr = data_filtered_coo[data_filtered_coo[catStr]==snr_temp]
        gg = sns.histplot(data_filtered_coo_snr,x='phase',hue=Param2comp,stat='count',common_norm=False,bins=bins,fill=True,ax=ax_temp,alpha=0.3,hue_order=compCond,palette=palette)
        gg.set_title(catStr+str(snr_temp))
        # ax_temp.legend(compCond,fontsize=5)

    data_filtered_v = data_filtered[data_filtered['trialMod']=='v']  
    gg=sns.histplot(data=data_filtered_v,x='phase',stat='count',bins=bins,fill=True,ax=axes[-1],alpha=0.3) 
    gg.set_title('CR')
    plt.tight_layout()
    plt.savefig(outputPathway+jsonnameStr+'.png')
    plt.close()

def phasPointDist2(data,outputPathway,jsonnameStr,filterCond,snr,bins=16,densityLogi=False):
    fig, axes = plt.subplots(int(len(snr))+1,1, figsize=(4,12),sharex=True,sharey=True)
    # fiter out trials stimulus played
    data = data[data['Stim']==1]    
    data_filtered = data[data[list(filterCond.keys())[0]]==list(filterCond.values())[0]]

    Param2comp = 'respLabelStr'
    compCond = ['hit','FA'] # hits vs fa
    catStr = 'trialMod'
    palette =sns.color_palette("Set2")[0:2]

    data_filtered_coo = data_filtered[data_filtered[Param2comp].isin(compCond)]  
    for cc,(snr_temp,ax_temp)in enumerate(zip(snr,axes)):
        data_filtered_coo_snr = data_filtered_coo[data_filtered_coo[catStr]==snr_temp]
        gg = sns.histplot(data_filtered_coo_snr,x='phase',hue=Param2comp,stat='count',common_norm=False,bins=bins,fill=True,ax=ax_temp,alpha=0.3,hue_order=compCond,palette=palette)
        gg.set_title(catStr+'_'+str(snr_temp))
        # ax_temp.legend(legends,fontsize=5)

    data_filtered_v = data_filtered[data_filtered['trialMod']=='v'] 
    data_filtered_v = data_filtered_v[data_filtered_v['respLabelStr'].isin(compCond)]
    gg=sns.histplot(data=data_filtered_v,x='phase',hue=Param2comp,stat='count',bins=bins,fill=True,ax=axes[-1],alpha=0.3,hue_order=compCond,palette=palette) 
    gg.set_title('V')
    plt.tight_layout()
    plt.savefig(outputPathway+jsonnameStr+'.png')
    plt.close()

def TimeseriesAlignbyRT(xx,yy,outputPathway,jsonnameStr,timeRange,seriesFeature):
    def calcMean(data_sort_temp,avetrials,timeRange,yycolumns):
        ave_data = []
        RTvsITCpl = []
        timpnts = np.linspace(timeRange[0],timeRange[1],data_sort_temp.iloc[0,yycolumns:].shape[0])
        Endind = np.where(timpnts<0)[0][-1]
        fs = data_sort_temp.iloc[0,yycolumns:].shape[0]/(timeRange[1]-timeRange[0])
        for ii in np.arange(0,data_sort_temp.shape[0],avetrials):
            if ii+avetrials<=data_sort_temp.shape[0]:
                ITC_temp = list(np.abs(np.mean(np.exp(data_sort_temp.iloc[ii:ii+avetrials,yycolumns:]*1j),axis=0)))
                ave_data.append(ITC_temp)
                sigma = np.std(ITC_temp[0:int(fs*0.3)])
                mean_temp = np.mean(ITC_temp[0:int(fs*0.3)])
                try:
                    ITC_temp_windowsum = np.convolve(ITC_temp,np.ones(int(fs*10/1000)),mode='same')/int(fs*10/1000)
                    RTvsITCpl.append([np.mean(data_sort_temp.rt[ii:ii+avetrials],axis=0),1000*(np.where(ITC_temp_windowsum>=mean_temp+3*sigma)[0][0])/fs])        
                except IndexError:
                    # pass
                    RTvsITCpl.append([np.mean(data_sort_temp.rt[ii:ii+avetrials],axis=0),np.nan])
        return np.array(ave_data),np.array(RTvsITCpl)

    def gaussianFit(x,y):
        # Define the Gaussian function
        def gaussian(x, amplitude, mean, stddev):
            return amplitude * np.exp(-((x - mean) / stddev) ** 2)  
        # Fit the data to the Gaussian function
        popt, pcov = curve_fit(gaussian, x, y)
        return popt

    def Linregfit(x,y):
        regr = linear_model.LinearRegression()
        regr.fit(x.reshape(-1,1), y)
        y_pred = regr.predict(x.reshape(-1,1))
        slope= regr.coef_[0]
        return y_pred,slope

    catCondAll = [{'trialMod':['a','av','v']},{'trialMod':['v']}]
    filterStr = ['hit','FA']
    data = pd.concat([yy,pd.DataFrame(xx)],axis=1)

    fig, axes = plt.subplots(4,3, figsize=(12,10),sharey = 'row',sharex='col', gridspec_kw={'width_ratios': [3, 1.5, 1.5]})
    if seriesFeature == 'phase':
        fig2, axes2 = plt.subplots(4,2, figsize=(6,11),sharey = 'col')
 
    # filter out nan phase,  respLabelStr!=hit  trials
    validTrials_hit = np.intersect1d(np.array(yy[yy.respLabelStr.isin(['hit'])].index),np.where(~np.isnan(xx).all(axis=1)==True)[0])
    data_filtered_hit = data.loc[validTrials_hit,:]

    # filter out nan phase, stim not played, respLabel!=88  trials
    validTrials_FA = np.intersect1d(np.array(yy[yy.respLabelStr.isin(['FA'])].index),np.where(~np.isnan(xx).all(axis=1)==True)[0])
    data_filtered_FA = data.loc[validTrials_FA,:]
   
    ind_temp = 0
    for catCond,data_filtered,filterStr_temp in zip(catCondAll,[data_filtered_hit,data_filtered_FA],filterStr):
        for ind, cc in enumerate(list(catCond.values())[0]):
            data_filtered_temp = data_filtered[data_filtered[list(catCond.keys())[0]]==cc]
            data_sort_temp = data_filtered_temp.sort_values(by=['rt'],axis = 0,ignore_index=True,kind='mergesort')

            c=axes[ind_temp,0].imshow(np.array(data_sort_temp.iloc[:,len(yy.columns):]), cmap='viridis',extent=[timeRange[0], timeRange[-1], len(data_sort_temp.index),1], aspect='auto', origin='upper')
            axes[ind_temp,0].set_title(cc+'-'+filterStr_temp,fontsize=8)
            fig.colorbar(c, ax=axes[ind_temp,0])
            # plot snr sorted by RT
            sns.swarmplot(x=data_sort_temp.snr,y=np.arange(1,len(data_sort_temp.index)+1,1), hue=data_sort_temp.snr,ax=axes[ind_temp,1],legend=False,size=0.5,dodge=True)
            #plot rt distribution
            sns.scatterplot(data=data_sort_temp,x='rt',y=np.arange(1,len(data_sort_temp.index)+1,1),size=0.5,legend=False,ax=axes[ind_temp,2]) 
            if seriesFeature == 'phase':
                # plot first 300 vs last 300 trials
                plottrials = 50
                ave_data,RTvsITCpl = calcMean(data_sort_temp,plottrials,timeRange,len(yy.columns))                            
                color_map = plt.get_cmap('viridis')
                for i in range(ave_data.shape[0]):
                    axes2[ind_temp,0].plot(np.linspace(timeRange[0], timeRange[-1],data_sort_temp.iloc[:,len(yy.columns):].shape[1]),ave_data[i,:],\
                        color=color_map(i/ave_data.shape[0]),alpha=0.3,label=str(i))    
                if i<10:
                    axes2[ind_temp,0].legend(fontsize=2)
                axes2[ind_temp,0].set_title(cc+'-'+filterStr_temp,fontsize=8)

                if len(RTvsITCpl.shape)==2:
                    axes2[ind_temp,1].scatter(RTvsITCpl[:,0],RTvsITCpl[:,1])
                else:
                    print('RTvsITCpl')
                    print(RTvsITCpl) 
                try:
                    RTvsITCpl_pred,rgSlope = Linregfit(RTvsITCpl[:,0],RTvsITCpl[:,1])
                    axes2[ind_temp,1].plot(RTvsITCpl[:,0],RTvsITCpl_pred)
                    axes2[ind_temp,1].set_title(cc+'-'+filterStr_temp+' '+str(np.round(rgSlope,decimals=4)),fontsize=8)
                except:
                    pass
            ind_temp = ind_temp+1

    fig.tight_layout()
    fig.savefig(outputPathway+jsonnameStr+'seriesAlignbyRT.png')
    plt.close()
    
    if seriesFeature == 'phase':
        fig2.tight_layout()
        fig2.savefig(outputPathway+jsonnameStr+'seriesAlignbyRT_ave.png')
        plt.close()
 
def TimeseriesAlignbyPhase(xx,yy,outputPathway,jsonnameStr,timeRange):

    catCondAll = [{'trialMod':['a','av','v']},{'trialMod':['v']}]
    filterStr = ['hit','FA']
    data = pd.concat([yy,pd.DataFrame(xx)],axis=1)

    fig, axes = plt.subplots(4,3, figsize=(12,10),sharey = 'row',sharex='col', gridspec_kw={'width_ratios': [3, 1.5, 1.5]})

    # filter out nan phase, stim not played, respLabel!=0  trials
    validTrials_hit = np.intersect1d(np.array(yy[yy.respLabel==0].index),np.where(~np.isnan(xx).all(axis=1)==True)[0])
    data_filtered_hit = data.loc[validTrials_hit,:]

    # filter out nan phase, stim not played, respLabel!=88  trials
    validTrials_FA = np.intersect1d(np.array(yy[yy.respLabel==88].index),np.where(~np.isnan(xx).all(axis=1)==True)[0])
    data_filtered_FA = data.loc[validTrials_FA,:]
   
    ind_temp = 0
    for catCond,data_filtered,filterStr_temp in zip(catCondAll,[data_filtered_hit,data_filtered_FA],filterStr):
        for ind, cc in enumerate(list(catCond.values())[0]):
            data_filtered_temp = data_filtered[data_filtered[list(catCond.keys())[0]]==cc]
            data_sort_temp = data_filtered_temp.sort_values(by=1098,axis = 0,ignore_index=True)

            c=axes[ind_temp,0].imshow(np.array(data_sort_temp.iloc[:,8:-1]), cmap='viridis',extent=[timeRange[0], timeRange[-1], len(data_sort_temp.index),1], aspect='auto', origin='upper')
            axes[ind_temp,0].set_title(cc+'-'+filterStr_temp,fontsize=8)
            fig.colorbar(c, ax=axes[ind_temp,0])
            # plot snr sorted by RT
            sns.swarmplot(x=data_sort_temp.snr,y=np.arange(1,len(data_sort_temp.index)+1,1), hue=data_sort_temp.snr,ax=axes[ind_temp,1],legend=False,size=0.5,dodge=True)
            #plot rt distribution
            sns.scatterplot(data=data_sort_temp,x='rt',y=np.arange(1,len(data_sort_temp.index)+1,1),size=0.5,legend=False,ax=axes[ind_temp,2]) 
    
            ind_temp = ind_temp+1

    fig.tight_layout()
    fig.savefig(outputPathway+jsonnameStr+'seriesAlignbyPhase.png')
    
    