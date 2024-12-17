import time
from datetime import timedelta
import numpy as np
import seaborn as sns
import os
import pickle
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt 
from spikeUtilities import getClsRasterMultiprocess,countTrialSPKs,SortfilterDF,loadPreprocessMat,glmfit,sampBalanceGLM
from sharedparam import getMonkeyDate_all
import random

############# 
# estimate whether the neuron modified by A, AV, or V using wilcoxon test, or modified by video.snr using glmfit
# only save clusters with fr>1
#############

start_time = time.monotonic()

#debug in local PC
# MonkeyDate_all = {'Elay':['230620']} #
# Pathway='/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/data'
# ResPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/AVmodIndex/'
# figSavePath = os.getcwd()+'/'

# run in monty
MonkeyDate_all = getMonkeyDate_all()
Pathway='/home/huaizhen/Documents/MonkeyAVproj/data/preprocNeuralMatfiles/Vprobe+EEG/'
ResPathway = '/home/huaizhen/Documents/MonkeyAVproj/data/Fitresults/AVmodIndex/'
figSavePath = '/home/huaizhen/Documents/MonkeyAVproj/data/Figures/AVmodIndex/'

df_avMod_all = pd.DataFrame()
# df_avMod_all_ori = pickle.load(open(ResPathway+'AVmodTTestDF.pkl','rb'))
# ## remove about2estimated sessions in the dataframe in case this session has been estimated/saved before
# df_avMod_all = df_avMod_all_ori[~df_avMod_all_ori['session_cls'].str.contains('|'.join(MonkeyDate_all[list(MonkeyDate_all.keys())[0]]))]
# # print(df_avMod_all.to_string())

if __name__ == '__main__':
  for Monkey,Date in MonkeyDate_all.items():
    for Date_temp in Date:  
      print('...............'+'estimating A/V modulation test in '+Monkey+'-'+Date_temp+'...............')                
      spikeTimeDict,labelDictFilter, \
          timeSamp2Chorus,spikefs,behavefs= loadPreprocessMat(Monkey,Date_temp,Pathway)
    
      for cls in list(spikeTimeDict.keys()):
        if any(substr in cls for substr in ['mua','good']):
        # if any(substr in cls for substr in ['cls140_ch13_mua']):
        # if 'cls126_ch10' in cls:
          print('cluster '+cls+' in progress............')
          # # get trial by trial raster in each cluster
          # labelDict_sub = {}
          # ntrials = 100
          # for key,value in labelDictFilter.items():
          #     labelDict_sub[key] = value[-ntrials:]
          # labelDictFilter = labelDict_sub.copy()
          # spikeTime_temp = spikeTimeDict[cls][-ntrials:]

          spikeTime_temp = spikeTimeDict[cls]

          # respLabel decode:  
          # nosoundplayed: [nan]
          # A:  hit[0],miss/latemiss[1],FAa[2],erlyResp[88]
          # AV: hit[0],miss/latemiss[1],FAa[2],erlyResp[88],FAv[22]
          # V:  hit[0],CRv[10,100miss],FAa[2],FAv[22]

          ################## check trial ave fr 
          extrastr = 'ChorusOn'
          timwin = [-1,4]
          spikeTimedf_temp_ss = getClsRasterMultiprocess(timeSamp2Chorus,spikeTime_temp,spikefs,behavefs,\
                                              labelDictFilter['chorusOnsetInd'],\
                                                labelDictFilter['chorusOnsetInd'] ,\
                                                labelDictFilter['JSOnsetIndwithVirtual'],\
                                              timwin,labelDictFilter)  
          spikeNumdf_temp=countTrialSPKs(spikeTimedf_temp_ss,estwin='off') #fr estimate before&after chorus onset    
          # only save clusters with  mean fr>1 during [-1, 1] relative to coo onset
          print('spkNum: '+str(spikeNumdf_temp['spknum'].values)+' mean:'+str(np.nanmean(spikeNumdf_temp['spknum'].values)/5))

          if np.nanmean(spikeNumdf_temp['spknum'].values)>=5:  
            ################## modification by sound onset, align2 ChorusOn
            # print('flag chorusOn')
            timwin = [-1,1]
            spikeTimedf_temp_A = getClsRasterMultiprocess(timeSamp2Chorus,spikeTime_temp,spikefs,behavefs,\
                                                labelDictFilter['chorusOnsetInd'],\
                                                  labelDictFilter['chorusOnsetInd'] ,\
                                                  labelDictFilter['JSOnsetIndwithVirtual'],\
                                                timwin,labelDictFilter)            
            spikeNumdf_temp_A=countTrialSPKs(spikeTimedf_temp_A,estwin='BlNSig',fs = behavefs,winTim=[-0.5,0.5]) #fr estimate before&after chorus onset
            spikeNumdf_temp_A_baseline = countTrialSPKs(spikeTimedf_temp_A,estwin='subwinoff',winTim=[-0.5,0]) # spknum before chorus onset
            spikeNumdf_temp_filtered_A,_ = SortfilterDF(spikeNumdf_temp_A,filterlable = {'respLabel':[0,1,88,10,100,22]})
            A_stats,A_pval = stats.wilcoxon(spikeNumdf_temp_filtered_A['spkRate_sig'].values,
                                            spikeNumdf_temp_filtered_A['spkRate_baseline'].values,alternative='two-sided')
            ################## modification by video onset, align2 VidOn
            # print('flag vidOn')
            timwin = [-0.5,0.5]
            spikeTimedf_temp_V = getClsRasterMultiprocess(timeSamp2Chorus,spikeTime_temp,spikefs,behavefs,\
                                                labelDictFilter['chorusOnsetInd'],\
                                                  labelDictFilter['VidOnsetIndwithVirtual'] ,\
                                                  labelDictFilter['JSOnsetIndwithVirtual'],\
                                                timwin,labelDictFilter)                            
            spikeNumdf_temp_raw_V=countTrialSPKs(spikeTimedf_temp_V,estwin='BlNSig',fs = behavefs,winTim=[-0.3,0.3])
            spikeTimedf_temp_filtered_V,_ = SortfilterDF(spikeNumdf_temp_raw_V,filterlable = {'respLabel':[0,10,100],
                                                                                            'trialMod':['v'],'AVoffset':[60,90,120,150]})
            V_stats,V_pval = stats.wilcoxon(spikeTimedf_temp_filtered_V['spkRate_sig'].values,
                                            spikeTimedf_temp_filtered_V['spkRate_baseline'].values,alternative='two-sided')            
            ################## modification on auditory responses by video, align2 CooOn                           
            # print('flag cooOn')
            timwin = [0,0.3]
            spikeTimedf_temp_AV = getClsRasterMultiprocess(timeSamp2Chorus,spikeTime_temp,spikefs,behavefs,\
                                                labelDictFilter['chorusOnsetInd'],\
                                                  labelDictFilter['cooOnsetIndwithVirtual'] ,\
                                                  labelDictFilter['JSOnsetIndwithVirtual'],\
                                                timwin,labelDictFilter)                            
            spikeTimedf_temp_filtered_AV,_ = SortfilterDF(spikeTimedf_temp_AV,filterlable = {'respLabel':[0],'trialMod':['av','a']})
            spikeNumdf_temp_raw_AV=countTrialSPKs(spikeTimedf_temp_filtered_AV,estwin='off')
            spikeNumdf_temp_A_baseline_filtered_V = SortfilterDF(spikeNumdf_temp_A_baseline,filterlable = {'respLabel':[0],'trialMod':['av','a']})[0].reset_index(drop=True)
            if (spikeNumdf_temp_raw_AV['trialNum'].values ==spikeNumdf_temp_A_baseline_filtered_V['trialNum'].values).all():
              # spikeNumdf_temp_raw_AV['spkRate'] = (spikeNumdf_temp_raw_AV['spknum'].values-
              #                                       spikeNumdf_temp_A_baseline_filtered_V['spknum'].values)/0.3
              pass
            else:
              print('trial mismatch between AV and baseline!!!!')
            # balance a/av trials
            spikeNumdf_temp_AV = spikeNumdf_temp_raw_AV.groupby(by=['trialMod','snr','respLabel','AVoffset']).sample(100,random_state=random.randint(1, 10000),replace=True) 
            spikeNumdf_temp_AV.sort_values(by=['trialMod','snr','respLabel','AVoffset','trialNum'],kind='mergesort',inplace=True)
            AV_stats,AV_pval = stats.wilcoxon(spikeNumdf_temp_AV[spikeNumdf_temp_AV['trialMod']=='a']['spknum'].values,spikeNumdf_temp_AV[spikeNumdf_temp_AV['trialMod']=='av']['spknum'].values,alternative='two-sided')          
            ################## modification on auditory responses by snr,trialMod using glmfit, align2 CooOn                           
            # print('flag cooOn')
            try :
                coeff_temp,pval_temp,evalparam = glmfit(spikeNumdf_temp_AV,['spknum'],['trialMod','snr'],'gaussian') #familystr=='poisson' or 'gaussian'
            except RuntimeWarning:
                print('fail to fit glm model in'+Monkey+Date_temp+' unit:'+cls)
                coeff_temp = pd.DataFrame([[np.nan]*2],columns=['coef_'+iv for iv in ['trialMod','snr']])
                pval_temp = pd.DataFrame([[np.nan]*2],columns=['pval_'+iv for iv in ['trialMod','snr']])
                evalparam = pd.DataFrame([[np.nan]*2],columns=['aic','bic'])           

            df_avMod_all = pd.concat([df_avMod_all,pd.DataFrame({'Monkey':[Monkey]*5,'session':[Date_temp]*5,'cls':[cls]*5,'session_cls':[Date_temp+'_'+cls]*5,\
                                                                  'iv':['A','V','AV','trialMod','snr'],
                                                                  'stats':[A_stats,V_stats,AV_stats,coeff_temp['coef_trialMod'].values[0],coeff_temp['coef_snr'].values[0]],
                                                                  'pval':[A_pval,V_pval,AV_pval,pval_temp['pval_trialMod'].values[0],pval_temp['pval_snr'].values[0]],
                                                                  'fr>1':[True]*5})])                              
      pickle.dump(df_avMod_all,open(ResPathway+'AVmodTTestDF.pkl','wb'))            

  ## plot 
  datasig = df_avMod_all[(df_avMod_all['pval']<0.05)&(df_avMod_all['fr>1']==True)] 
  # print(df_avMod_all.to_string())
  print(datasig.to_string())
  print('Elay: SigUnits #'+str(len(datasig[datasig['Monkey']=='Elay'].session_cls.unique())))
  print('Wu: SigUnits #'+str(len(datasig[datasig['Monkey']=='Wu'].session_cls.unique())))
  fig, axess = plt.subplots(1,2,figsize=(12,6),gridspec_kw={'width_ratios': [1,1]})
  sns.stripplot(datasig,x='iv',y='stats',hue='session_cls',ax=axess[0],legend=False)
  sns.countplot(datasig,x='iv',hue='Monkey',ax=axess[1])
  plt.tight_layout()
  plt.savefig(figSavePath+'AVmodIndexTTest_byCluster.png')
  plt.close()

  end_time = time.monotonic()
  print(timedelta(seconds=end_time - start_time))


