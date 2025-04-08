import time
from datetime import timedelta
import numpy as np
import seaborn as sns
import pickle
import pandas as pd
from matplotlib import pyplot as plt 
from scipy import stats
import re
from scipy.stats import norm
from spikeUtilities import getdPrime, SortfilterDF
from decoders import decoder_detection,decoder_detection_v2,decoder_decision
from sharedparam import getMonkeyDate_all,neuronfilterDF
from scipy.stats import zscore
# concatenate all clusters in all sessions together, and then do the population decoding across sessions
# classify hit/miss or SNR along slide windows 
#
#obtain all neurons recorded from one monkey
def pickneuronpopulations(Monkey,STRFstr):
    clsInfo,_ = pickle.load(open(wavformPathway+'AllUnits_spkwavform_Wuless.pkl','rb')) 
    clsInfo_temp = clsInfo[clsInfo['Monkey']==Monkey]                   
    # filter neurons based on different rules
    df_avMod_all_sig_ori,_ = neuronfilterDF(AVmodPathway,STRFexcelPath,'ttest',Monkey,STRF=STRFstr)
    df_avMod_all_sig_ori = df_avMod_all_sig_ori[df_avMod_all_sig_ori['session_cls'].isin(clsInfo_temp.session_cls.values)]
    # randomly pick clusters  
    df_avMod_all_sig_unique = df_avMod_all_sig_ori.groupby(['Monkey','session','cls','session_cls','STRFsig']).size().reset_index()                    
    # df_avMod_all_sig = df_avMod_all_sig_unique.groupby('STRFsig').sample(30,replace=False).sort_values(by=['Monkey','session'])
    # print(Monkey+' '+STRFstr)
    # print(df_avMod_all_sig_unique.groupby('STRFsig').size().to_string())
    return df_avMod_all_sig_unique

# concatenate all trials for all neurons
def catTrials(Monkey,Date,alignKeys,df_avMod_all_sig,filterdict):
    # initialize dataframe to save psth time series data of all neurons
    AllSUspk_df_4decoding = pd.DataFrame()

    for Date_temp in Date:  
        print('session date:'+Date_temp)
        AllSUspk_df=pickle.load(open(dataPathway+Monkey+'_'+Date_temp+'_allSU+MUA_alltri_'+alignKeys+'_'+binmethod+'_'+str(bin)+'msbinRawPSTH_df.pkl','rb')).reset_index(drop=True) 
        frcolall = [s for s in list(AllSUspk_df.columns) if 'fr_' in s]        
        # reduce temporal resolution
        frcolfull = [frcolall[i] for i in np.arange(0,len(frcolall),resolutionred,dtype=int)]
        frcol = [element for element in frcolfull if float(element.split('_')[1]) >= timwinStart[0] and float(element.split('_')[1]) <= timwinStart[1]]
        xlimrange = [float(frcol[0].split('_')[1]),float(frcol[-1].split('_')[1])]
        AllSUspk_df_sub = AllSUspk_df[list(AllSUspk_df.columns[:15])+frcol]

        # preprocess each cls separately
        df_avMod_sess_sig = df_avMod_all_sig[df_avMod_all_sig['session']==Date_temp]
        for cc,cls in enumerate(list(df_avMod_sess_sig['cls'].unique())):              
            AllSUspk_df_temp_cls = AllSUspk_df_sub[AllSUspk_df_sub['cls']==cls].reset_index(drop=True).copy()
            df_avMod_sess_temp = df_avMod_all_sig[df_avMod_all_sig['session_cls']==Date_temp+'_'+cls] 
            # if need to baseline correct firing rate of each trial of this cls
            if baselinecorrectFlag: 
                PSTH_arr = AllSUspk_df_temp_cls[frcol+noiseCR].values - AllSUspk_df_temp_cls['baselineFR'].values.reshape(-1,1) 
            else:
                PSTH_arr = AllSUspk_df_temp_cls[frcol+noiseCR].values
            AllSUspk_df_temp_cls[frcol+noiseCR] = PSTH_arr            
            # zscore spkrate time series
            AllSUspk_df_temp_cls[frcol+noiseCR] = (AllSUspk_df_temp_cls[frcol+noiseCR] - AllSUspk_df_temp_cls[frcol+noiseCR].mean(skipna=True))\
                                                    /AllSUspk_df_temp_cls[frcol+noiseCR].std(skipna=True)
            # concatenate trials of all neurons
            trials = AllSUspk_df_temp_cls.shape[0]
            info_temp_df = pd.DataFrame({'sess_cls':[Date_temp+'_'+cls]*trials})  
            AllSUspk_df_temp = pd.concat([info_temp_df,AllSUspk_df_temp_cls],axis=1) 
            AllSUspk_df_4decoding = pd.concat([AllSUspk_df_4decoding,AllSUspk_df_temp],axis=0) 
    AllSUspk_df_4decoding_filter= SortfilterDF(AllSUspk_df_4decoding,filterlable =filterdict)[0].reset_index(drop=True)   # filter out trials       
    #remove sessions with less than 2 trials in any subgroup, or missing conditions
    group_sizes = AllSUspk_df_4decoding_filter.groupby(by=['sess_cls','trialMod','snr','respLabel','AVoffset']).size()                    
    # find cls missing conditions
    CondNumAllCls = group_sizes.groupby('sess_cls').size().reset_index(name='size')
    maxCondNum=CondNumAllCls.value_counts('size').reset_index(name='clsnum')['size'].max()
    missCondCls = list(CondNumAllCls[CondNumAllCls['size']<maxCondNum].sess_cls.unique())
    #find cls with less than 2 trials in any subgroup
    fewtrialsCls = list(group_sizes[group_sizes < 2].index.get_level_values('sess_cls').unique())
    sess_cls_to_remove = missCondCls+fewtrialsCls
    # Filter out cls
    AllSUspk_df_4decoding_ready = AllSUspk_df_4decoding_filter[~AllSUspk_df_4decoding_filter['sess_cls'].isin(sess_cls_to_remove)]                
    print('removed '+str(len(sess_cls_to_remove))+' clusters from '+Monkey+':')
    print('missing condition cls: '+ str(missCondCls))
    print('missing trials cls: '+str(fewtrialsCls))
    return AllSUspk_df_4decoding_ready,frcol,xlimrange

def decoderPlot(Monkey,namstrallSU,figtitle,xlimrange):
    # ## plot fit results 
    print('Start plotting results................')
    fitacc_nNueron_dict,frcol,xlimrange = pickle.load(open(ResPathway+Monkey+'_Xsession'+namstrallSU+figtitle+'.pkl','rb'))  # Nneurons X clsSamp X trials X bootstrapTimes
    
    behavdf = pd.DataFrame() 
    matshape = fitacc_nNueron_dict['fitacc_nNeuron'].shape      

    for Nn in range(matshape[0]):
        for tim in range(matshape[1]):  
            for bs in range(matshape[3]):
                fitacc_nNueron_temp = fitacc_nNueron_dict['fitacc_nNeuron'][Nn,tim,:,bs].T.reshape((-1,))
                # trialLab = fitacc_nNueron_dict['trialLabel']
                fitacc_nNueron_temp_bs = fitacc_nNueron_dict['fitacc_nNeuron_bsline'][Nn,tim,:,bs].T.reshape((-1,))
                # trialLab_bs = [ls+'_baseline' for ls in fitacc_nNueron_dict['trialLabel']]                             
                behavdf_temp = pd.DataFrame({'mod':['condition','baseline'],'acc':list(fitacc_nNueron_temp)+list(fitacc_nNueron_temp_bs)}) 
                behavdf_temp['time'] = np.float64(frcol[tim].split('_',1)[-1])
                behavdf_temp['BSrpt'] = bs
                behavdf = pd.concat((behavdf,behavdf_temp),axis=0)

    fig, axess = plt.subplots(1,1,figsize=(8,4)) #  
    print(behavdf.sort_values(by=['time','BSrpt','mod'],kind='mergesort').to_string())

    colorpalette = ['black','dimgray']
    sns.lineplot(behavdf,x = 'time',y = 'acc',hue='mod',palette=colorpalette,hue_order=['condition','baseline'],ax=axess,errorbar = 'ci')                  
    axess.set_xlabel('Time',fontsize=fontsizeNo)
    axess.set_ylabel('Accuracy',fontsize=fontsizeNo)
    axess.set_ylim([0,1])
    axess.xaxis.set_ticks([xlimrange[0],0,xlimrange[1]])
    axess.legend(frameon=False, framealpha=0,fontsize=fontsizeNo-4,loc='upper left') 
    axess.tick_params(axis='both', which='major', labelsize=fontsizeNo-2)
    axess.set_title(figtitle,fontsize=fontsizeNo)
    fig.tight_layout()
    fig.savefig(figSavePath+Monkey+'_svmdecoderACCbyTime'+namstrallSU+figtitle+'.png')
    plt.close(fig)


MonkeyDate_all = {'Elay':['230420','230616']}#,,'230620'
dataPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/PSTHdataframe/'
AVmodPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/AVmodIndex/'
ResPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/decoder/'
figSavePath = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Figures/decoder/'
STRFexcelPath = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/STRF/'
wavformPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/wavformStruct/'

# MonkeyDate_all = getMonkeyDate_all()
# dataPathway = '/home/huaizhen/Documents/MonkeyAVproj/data/PSTHdataframe/'
# AVmodPathway = '/home/huaizhen/Documents/MonkeyAVproj/data/Fitresults/AVmodIndex/'
# ResPathway = '/home/huaizhen/Documents/MonkeyAVproj/data/Fitresults/decoder/'
# figSavePath = '/home/huaizhen/Documents/MonkeyAVproj/data/Figures/decoder/'
# STRFexcelPath = '/home/huaizhen/Documents/MonkeyAVproj/data/Fitresults/STRF/'
# wavformPathway = '/home/huaizhen/Documents/MonkeyAVproj/data/Fitresults/wavformStruct/'

baselinecorrectFlag = False
noiseCR = ['spkRateCRa2']#['OnsetCRa','spkRateCRa','spkRateCRa1','spkRateCRa2','spkRateCRa3']
binmethod = 'overlaptimwin' 
bin = 200 #ms 
bstimes = 2 #50
resolutionred = 40 #10 # reduce temporal resolotion of the decoding by this scale, original temporal resolution:0.01
units = 3 # number of neurons used for decoding 
# ##### decode data input filter and decode conditions           
# inputPara ={'Elay':[
#                 {'decodeCat':{'snr':['difficult','medium','easy']},'figtitle':'decode-SNR-AVoffset90-120',
#                 'filterdict':{'trialMod':['a','av'],'respLabel':['hit'],'AVoffset':[90,120],'snr':['difficult','medium','easy']}}],
#             'Wu':[
#                 {'decodeCat':{'snr':['difficult','medium','easy']},'figtitle':'decode-SNR-AVoffset90-120',
#                 'filterdict':{'trialMod':['a','av'],'respLabel':['hit'],'AVoffset':[90,120],'snr':['difficult','medium','easy']}}]}                  

inputPara ={
            'Elay':{
                'decodeCat':{'trialMod':['a','av']},'figtitle':'decodeMod-AVoffset90-SNRdiff',
                'filterdict':{'trialMod':['a','av'],'respLabel':['hit'],'AVoffset':[90],'snr':[-15,-10]}},
            'Wu':{
                'decodeCat':{'trialMod':['a','av']},'figtitle':'decodeMod-AVoffset90-SNRmedium',
                'filterdict':{'trialMod':['a','av'],'respLabel':['hit'],'AVoffset':[90],'snr':[-5,0]}}
            } 

fontsizeNo = 20
figformat = 'png'
fontnameStr = 'Arial'#'DejaVu Sans'

if __name__ == '__main__':
    for alignKeys,timwinStart in zip(['align2coo'],[[-0.9,1]]):  #zip(['align2coo','align2js','align2DT'],[[-0.5,1],[-1,0],[-1,1]]):  
        for STRFstr in ['notsig','sig']:#['all','notsig','sig']:                      
                namstrallSU = '_'+alignKeys+'_Hit=A+AV_ttestfilteredUnits_STRF'+STRFstr+'_spkwavShape-all'+str(bin)+'msslidewin'+'_AVoffset90+120'+'_sample30noreplacement_lessWu-sess'
                start_time = time.monotonic()
                for Monkey,Date in MonkeyDate_all.items():
                    # obtain all usable strf/nstrf nuerons from the current monkey
                    df_avMod_all_sig = pickneuronpopulations(Monkey,STRFstr)
                    AllSUspk_df_4decoding_ready,frcol,xlimrange = catTrials(Monkey,Date,alignKeys,df_avMod_all_sig,inputPara[Monkey]['filterdict'])                                        
                    # AllSUspk_df_4decoding_ready['snr']=AllSUspk_df_4decoding_ready['snr'].replace({-15:'difficult',-10:'difficult',-5:'medium',0:'medium',5:'easy',10:'easy'})
                    
                    decodeCat = inputPara[Monkey]['decodeCat'] 
                    figtitle = inputPara[Monkey]['figtitle'] 
                                   
                    print('start the decoding.....')
                    # decoding start here
                    for tim,col in enumerate(frcol):
                        print('decoding time points '+col)
                        time1=time.monotonic()
                        AllSUspk_df_4decoding_temp = AllSUspk_df_4decoding_ready[['Monkey','sess','sess_cls','respLabel',
                                                        'AVoffset', 'snr', 'trialMod', 'snr-shift', 'trialNum']
                                                        +noiseCR+[col]]
                        AllSUspk_df_4decoding_temp = AllSUspk_df_4decoding_temp.rename(columns={noiseCR[0]:'spkRateNoise',col:'spkRateSig'}).reset_index(drop=True)
                        fitacc_nNueron_dict_temp = decoder_decision(AllSUspk_df_4decoding_temp,decodeCat,units=units,trainSample_eachCat = 50,testSample_eachCat = 10,bstimes=bstimes) # # 'fitacc_nNeuron': clsSamp X 1 X A/AV X bootstrapTimes 
                        # fitacc_nNueron_dict_temp = decoder_decision(AllSUspk_df_4decoding_temp,decodeCat,trainSample_eachCat = 20,testSample_eachCat = 5,bstimes=bstimes) # # 'fitacc_nNeuron': clsSamp X 1 X A/AV X bootstrapTimes                         
                        if tim==0:       
                            fitacc_nNueron_dict = fitacc_nNueron_dict_temp.copy()
                        else:
                            fitacc_nNueron_dict['fitacc_nNeuron'] = np.concatenate((fitacc_nNueron_dict['fitacc_nNeuron'],fitacc_nNueron_dict_temp['fitacc_nNeuron']),axis=1) # 'fitacc_nNeuron': clsSamp X time X A/AV X bootstrapTimes 
                            fitacc_nNueron_dict['fitacc_nNeuron_bsline'] = np.concatenate((fitacc_nNueron_dict['fitacc_nNeuron_bsline'],fitacc_nNueron_dict_temp['fitacc_nNeuron_bsline']),axis=1)
                        print('done with decoding time points '+col +': '+str(timedelta(seconds= time.monotonic()- time1)))

                    pickle.dump([fitacc_nNueron_dict,frcol,xlimrange],open(ResPathway+Monkey+'_Xsession'+namstrallSU+figtitle+'.pkl','wb'))            
                    #######plot decoder results
                    decoderPlot(Monkey,namstrallSU,figtitle,xlimrange)

            


