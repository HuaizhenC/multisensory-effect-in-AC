import time
from datetime import timedelta
import numpy as np
import random
from scipy import stats
import seaborn as sns
import os
import pickle
import pandas as pd
from matplotlib import pyplot as plt 
from spikeUtilities import SortfilterDF,estNoiseCorr,sampBalanceGLM,sampBalanCond
from sharedparam import getMonkeyDate_all,neuronfilterDF



def plotfitcorr(slopeThres_df1,axess4,titlestr, catstr, varStr, compCol, xvar, yvar, snrorder,colorlist, markerSymb, sortlist):
    # for modtest,colr in zip(slopeThres_df1[catstr].unique(),colorlist):
    for ii, (modtest,colr) in enumerate(zip(snrorder,colorlist)):        
        slopeThres_temp = slopeThres_df1[slopeThres_df1[catstr]==modtest]
        # print(slopeThres_temp.to_string())
        axess4.scatter(slopeThres_temp[slopeThres_temp[compCol]==xvar].sort_values(sortlist,kind='mergesort')[varStr].values,
                                    slopeThres_temp[slopeThres_temp[compCol]==yvar].sort_values(sortlist,kind='mergesort')[varStr].values,
                                    alpha=0.5,c=colr,label=modtest,marker=markerSymb,edgecolors='none') 
        statsRes = stats.wilcoxon(slopeThres_temp[slopeThres_temp[compCol]==xvar].sort_values(sortlist,kind='mergesort')[varStr].values,
            slopeThres_temp[slopeThres_temp[compCol]==yvar].sort_values(sortlist,kind='mergesort')[varStr].values,nan_policy='omit')

        if np.mean(slopeThres_temp[slopeThres_temp[compCol]==xvar].sort_values(sortlist,kind='mergesort')[varStr].values)>\
        np.mean(slopeThres_temp[slopeThres_temp[compCol]==yvar].sort_values(sortlist,kind='mergesort')[varStr].values):
            axess4.text(0.8,-0.1-ii*0.05,xvar+'>'+yvar+' '+str(modtest)+' p='+str(np.round(statsRes[1],decimals=3)),horizontalalignment='right',verticalalignment='center',fontsize=10,fontname=fontnameStr)
        else:
            axess4.text(0.8,-0.1-ii*0.05,xvar+'<'+yvar+' '+str(modtest)+' p='+str(np.round(statsRes[1],decimals=3)),horizontalalignment='right',verticalalignment='center',fontsize=10,fontname=fontnameStr)
        print(xvar+' median:'+str(np.median(slopeThres_temp[slopeThres_temp[compCol]==xvar].sort_values(sortlist,kind='mergesort')[varStr].values)))
        print(yvar+' median:'+str(np.median(slopeThres_temp[slopeThres_temp[compCol]==yvar].sort_values(sortlist,kind='mergesort')[varStr].values)))

    # axisLoBound = min([axess4.get_xlim()[0],axess4.get_ylim()[0]])
    # axisHiBound = max([axess4.get_xlim()[1],axess4.get_ylim()[1]])
    # axess4.set_xlim([axisLoBound,axisHiBound])
    # axess4.set_ylim([axisLoBound,axisHiBound])
    # axess4.plot(np.linspace(axisLoBound,axisHiBound,100),np.linspace(axisLoBound,axisHiBound,100),'--',color='gray',linewidth=1.5)

    axess4.set_xlim([-0.8,0.9])
    axess4.set_ylim([-0.8,0.9])
    axess4.set_xticks(list(np.arange(-0.8,0.9,0.5)))
    axess4.set_yticks(list(np.arange(-0.8,0.9,0.5)))
    axess4.plot(np.linspace(-0.8,0.9,100),np.linspace(-0.8,0.9,100),'--',color='gray',linewidth=2)

    axess4.set_title(titlestr,fontsize=fontsizeNo,fontname=fontnameStr)
    if xvar=='a':
        axess4.set_xlabel('Static Visual',fontsize=fontsizeNo,fontname=fontnameStr)
        axess4.set_ylabel('Congruent Visual',fontsize=fontsizeNo,fontname=fontnameStr)
    else:
        axess4.set_xlabel(xvar,fontsize=fontsizeNo,fontname=fontnameStr)
        axess4.set_ylabel(yvar,fontsize=fontsizeNo,fontname=fontnameStr)
    axess4.legend(frameon=False,loc='upper left',fontsize=fontsizeNo,prop={'family': fontnameStr})
    axess4.tick_params(axis='both', which='major', labelsize=fontsizeNo-2)

# debug in local pc
# MonkeyDate_all = {'Elay':['230420','230620']}#,,'230718'
# ResPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/noisecorrelation/'
# figSavePath = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Figures/noisecorrelation/'
# DataPathway= '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/PSTHdataframe/'
# glmfitPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/glmfit/'
# AVmodPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/AVmodIndex/'
# wavformPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/wavformStruct/'
# STRFexcelPath = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/STRF/'

# run in monty
MonkeyDate_all = getMonkeyDate_all() 
figSavePath = '/home/huaizhen/Documents/MonkeyAVproj/data/Figures/noisecorrelation/'
DataPathway= '/home/huaizhen/Documents/MonkeyAVproj/data/PSTHdataframe/'
ResPathway = '/home/huaizhen/Documents/MonkeyAVproj/data/Fitresults/noisecorrelation/'
AVmodPathway = '/home/huaizhen/Documents/MonkeyAVproj/data/Fitresults/AVmodIndex/'
glmfitPathway = '/home/huaizhen/Documents/MonkeyAVproj/data/Fitresults/glmfit/'
wavformPathway = '/home/huaizhen/Documents/MonkeyAVproj/data/Fitresults/wavformStruct/'
STRFexcelPath = '/home/huaizhen/Documents/MonkeyAVproj/data/Fitresults/STRF/'

fontsizeNo = 15
baselinecorrectFlag = False
noiseCR = ['spkRateCRa2']#['OnsetCRa','spkRateCRa','spkRateCRa1','spkRateCRa2','spkRateCRa3']
bin_all = [300]#[200,300] #ms 

fontsizeNo = 20
figformat = 'png'
fontnameStr = 'Arial'#'DejaVu Sans'

# filter out these trials in each session 
filterdict = {'trialMod':['a','av'],'respLabel':['hit'],'AVoffset':[90,120]}
# neurons used in the final analysis
clsInfo,_ = pickle.load(open(wavformPathway+'AllUnits_spkwavform_Wuless.pkl','rb')) 
if __name__ == '__main__':
    for bin in bin_all:
        for extrastr in ['align2coo']:#['align2coo','align2DT']:  #zip(['align2coo','align2js','align2DT'],[[-0.5,1],[-1,0],[-1,1]]):  
            
            fig, axess = plt.subplots(1,2,figsize=(10,5))

            for ss,STRFstr in enumerate(['notsig','sig']):#['all','notsig','sig']:
                for spkLabel in ['all']: #['all','RS/FS','positive','triphasic']:   
                    
                    namstrallFig = '_'+extrastr+'_Hit=A+AV_ttestfilteredUnits_spkwavShape-'+spkLabel.replace('/','or')+str(bin)+'msslidewin'+'_AVoffset90+120'+'_sample30noreplacement_lessWu-sess'                          
                    namstrallSU = '_'+extrastr+'_Hit=A+AV_ttestfilteredUnits_STRF'+STRFstr+'_spkwavShape-'+spkLabel.replace('/','or')+str(bin)+'msslidewin'+'_AVoffset90+120'+'_sample30noreplacement_lessWu-sess'
                                       
                    NCdf_filtered_2monks = pd.DataFrame()
                    for mm,(Monkey,Date) in enumerate(MonkeyDate_all.items()):
                        AllSUspk_df_4decoding = pd.DataFrame()
                        clsNum = 0
                        # filter neurons based on different rules
                        # clsInfo_temp = clsInfo[(clsInfo['spkLabel']==spkLabel)&(clsInfo['Monkey']==Monkey)&(clsInfo['clsNo']==clsNo)]
                        if spkLabel=='all':
                            clsInfo_temp = clsInfo[clsInfo['Monkey']==Monkey]
                        else:
                            clsInfo_temp = clsInfo[(clsInfo['spkLabel']==spkLabel)&(clsInfo['Monkey']==Monkey)]
                        df_avMod_all_sig,_ = neuronfilterDF(AVmodPathway,STRFexcelPath,'ttest',Monkey,STRF=STRFstr)
                        df_avMod_all_sig = df_avMod_all_sig[df_avMod_all_sig['session_cls'].isin(clsInfo_temp.session_cls.values)]
                    
                        for Date_temp in Date:  
                            print('...............'+'process session '+Monkey+'-'+Date_temp+'...............')
                            # using firing rate 300ms after coo onset
                            AllSUspk_df_win=pickle.load(open(DataPathway+Monkey+'_'+Date_temp+'_allSU+MUA_alltri_'+extrastr+'_overlaptimwin_'+str(bin)+'msbinRawPSTH_df.pkl','rb')).reset_index(drop=True)             

                            allcols = list(AllSUspk_df_win.columns)
                            frcols = [ele for ele in allcols if 'fr_' in ele]
                            if extrastr == 'align2DT' or extrastr == 'align2js':
                                split_elements = [np.float64(element.split('_')[1])+bin/1000 for element in frcols]
                            else:                        
                                split_elements = [np.float64(element.split('_')[1]) for element in frcols]
                            split_elements = [np.float64(element.split('_')[1]) for element in frcols]
                            index_of_closest = split_elements.index(min(split_elements, key=lambda x: abs(x)))
                            usefulcols = list(set(allcols) ^ set(frcols))+[frcols[index_of_closest]]
                            AllSUspk_df = AllSUspk_df_win[usefulcols]
                            AllSUspk_df = AllSUspk_df.rename(columns={'fr_'+str(split_elements[index_of_closest]):'spkRate'}) 
                            # print(AllSUspk_df.to_string())

                            # filter trial conditions 
                            AllSUspk_df_filter= SortfilterDF(AllSUspk_df,filterlable =filterdict)[0].reset_index(drop=True)   # filter out trials  
                            # print(AllSUspk_df_filter.to_string())
                            
                            # preprocess each cls separately
                            for cc,cls in enumerate(list(AllSUspk_df_filter['cls'].unique())): 
                                if (df_avMod_all_sig['session_cls']==Date_temp+'_'+cls).any():               
                                    AllSUspk_df_temp_cls = AllSUspk_df_filter[AllSUspk_df_filter['cls']==cls].reset_index(drop=True).copy()
                                    df_avMod_sess_temp = df_avMod_all_sig[df_avMod_all_sig['session_cls']==Date_temp+'_'+cls] 
                                    mod = "+".join(sorted(df_avMod_sess_temp['iv'].values))                                                      
                                    # # get su discrimination rate in A and AV conditions from NMfit
                                    # hitrateA = np.mean(data[cls]['y_raw_a'])
                                    # hitrateAV = np.mean(data[cls]['y_raw_av'])
                                    hitrateA = 0
                                    hitrateAV = 0

                                    # if need to baseline correct firing rate of each trial of this cls
                                    if baselinecorrectFlag: 
                                        PSTH_arr = AllSUspk_df_temp_cls[['spkRate']+noiseCR].values - AllSUspk_df_temp_cls['baselineFR'].values.reshape(-1,1) 
                                    else:
                                        PSTH_arr = AllSUspk_df_temp_cls[['spkRate']+noiseCR].values
                                    AllSUspk_df_temp_cls[['spkRate']+noiseCR] = PSTH_arr
                                    
                                    # zscore spkrate within a cluster
                                    meanfrcls = np.nanmean(PSTH_arr)
                                    SDfrcls = np.nanstd(PSTH_arr)
                                    for col in ['spkRate']+noiseCR:
                                        AllSUspk_df_temp_cls[col] = AllSUspk_df_temp_cls[col].apply(lambda x: (x-meanfrcls)/SDfrcls)
                                    # rename sig and noise columns
                                    AllSUspk_df_temp_cls = AllSUspk_df_temp_cls.rename(columns={'spkRate':'spkRateSig',noiseCR[0]:'spkRateNoise'}).reset_index(drop=True)    
            
                                    # concatenate ordered trials in each cluster
                                    trials = AllSUspk_df_temp_cls.shape[0]
                                    spknumcoloumns = ['respLabel','AVoffset', 'snr', 'trialMod', 'snr-shift', 'trialNum', 'spkRateSig','spkRateNoise']
                                    info_temp_df = pd.DataFrame({'Monkey':[Monkey]*trials,'sess':[Date_temp]*trials,'sess_cls':[Date_temp+'_'+cls]*trials,'clsNum':[int(clsNum)]*trials,
                                                                'neurMod':[mod]*trials,'NM_A_hit':[hitrateA]*trials, 'NM_AV_hit':[hitrateAV]*trials,'NM_hit_ave':[np.mean([hitrateA,hitrateAV])]*trials})  
                                    AllSUspk_df_temp = pd.concat([info_temp_df,AllSUspk_df_temp_cls[spknumcoloumns]],axis=1) 
                                    AllSUspk_df_4decoding = pd.concat([AllSUspk_df_4decoding,AllSUspk_df_temp],axis=0) 
                                    clsNum = clsNum+1
                         
                        # group snrs into 3 cat
                        AllSUspk_df_4decoding['snr']=AllSUspk_df_4decoding['snr'].replace({-15:'difficult',-10:'difficult',-5:'medium',0:'medium',5:'easy',10:'easy'})
                                
                        spkCol = ['spkRateSig','spkRateNoise']
                        NCdf = pd.DataFrame()
                        # calculate NC for one monkey
                        for sess_temp in AllSUspk_df_4decoding.sess.unique(): 
                            AllSUspk_df_4decoding_sess = AllSUspk_df_4decoding[AllSUspk_df_4decoding['sess']==sess_temp]
                            cls_temp1 = list(AllSUspk_df_4decoding_sess.sess_cls.unique())
                            # cls_temp = [cc for cc in cls_temp1 if 'good' in cc]
                            cls_temp = cls_temp1
                            if len(cls_temp)>1: # only calculate NC in sessions with more than 2 single units
                                print('...............'+'estimating noisecorr measurement in '+Monkey+'-'+sess_temp+'...............')
                                # select trials with balanced number across categories for this session
                                AllSUspk_df_4decoding_sess_temp = AllSUspk_df_4decoding_sess[AllSUspk_df_4decoding_sess['sess_cls']==cls_temp[0]].reset_index(drop=True)
                                # selected_rows = sampBalanCond(AllSUspk_df_4decoding_sess_temp,['trialMod','snr']) # balance number of trials in each condition                              
                                _,selected_rows = sampBalanceGLM(AllSUspk_df_4decoding_sess_temp,['trialMod','snr'],seeds=42,method='upsample')
                                for cc1, cls1 in enumerate(cls_temp[:-1]):
                                    AllSUspk_df_4decoding_sess_cls1 = AllSUspk_df_4decoding_sess[AllSUspk_df_4decoding_sess['sess_cls']==cls1].reset_index(drop=True)
                                    SPKdf_temp_part = AllSUspk_df_4decoding_sess_cls1.loc[selected_rows].sort_values(['trialNum'],kind='mergesort')                    
                                    SPK = SPKdf_temp_part[['trialNum','trialMod','snr',]+spkCol].sort_values(['trialMod','snr','trialNum'],kind='mergesort')
                                    for cc2, cls2 in enumerate(cls_temp[cc1+1:]):
                                        AllSUspk_df_4decoding_sess_cls2 = AllSUspk_df_4decoding_sess[AllSUspk_df_4decoding_sess['sess_cls']==cls2].reset_index(drop=True)
                                        SPKdf_temp2_part = AllSUspk_df_4decoding_sess_cls2.loc[selected_rows].sort_values(['trialNum'],kind='mergesort')   
                                        SPK2 = SPKdf_temp2_part[['trialNum','trialMod','snr',]+spkCol].sort_values(['trialMod','snr','trialNum'],kind='mergesort')
                                        #estimate noise correlation
                                        # NCdf_temp = estNoiseCorr(SPK,SPK2,[cls1[7:],cls2[7:]],spkCol,cond=['trialMod','snr'])
                                        NCdf_temp = estNoiseCorr(SPK,SPK2,[cls1[7:],cls2[7:]],spkCol,cond=['trialMod']) 
                                        NCdf_temp['snr']=0

                                        NCdf_temp['sess'] = sess_temp
                                        NCdf_temp['Monkey'] = Monkey
                                        NCdf = pd.concat((NCdf,NCdf_temp))                        
                        ## plot NC  
                        # NCdf_filtered = NCdf.dropna(subset=['sig'],inplace=False)
                        print(NCdf.iloc[:10].to_string())

                        NCdf_filtered_2monks = pd.concat([NCdf_filtered_2monks,NCdf],axis=0)
                    # A vs AV at a snr
                    snr = 'difficult'
                    if len(NCdf_filtered_2monks.snr.unique())>1:
                        # a vs av condition
                        NCtemp = NCdf_filtered_2monks[(NCdf_filtered_2monks['time']=='spkRateSig')&(NCdf_filtered_2monks['snr']==snr)]
                        plotfitcorr(NCtemp,axess[ss],STRFstr,'snr','corrcoef', 'trialMod', 'a', 'av', [snr],['gray'], 'o', ['sess','Monkey','NeuPairs'])
                        axess[ss].legend([],frameon=False,loc='upper left',fontsize=fontsizeNo)
                        # plotfitcorr(NCdf_filtered_2monks[(NCdf_filtered_2monks['time']=='spkRateNoise')&(NCdf_filtered_2monks['snr']==snr)],axess[1],STRFstr+'_NoiseNC','snr','corrcoef', 'trialMod', 'a', 'av', [snr],['gray'], 'o', ['sess','Monkey','NeuPairs'])
                        # axess[1].legend([],frameon=False,loc='upper left',fontsize=fontsizeNo)
                fig.tight_layout()
                fig.savefig(figSavePath+'NC_byMod-'+snr+'-scatter-2monks'+namstrallFig+'.'+figformat)
                plt.close(fig)  


print('done')

