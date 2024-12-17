import time
from datetime import timedelta
import numpy as np
import seaborn as sns
import mat73
import os
import pickle
import pandas as pd
from matplotlib import pyplot as plt 
# from spikeUtilities import getClsRasterMultiprocess,countTrialSPKs,loadPreprocessMat,SortfilterDF,decodrespLabel2str
# from decoders import svmdecoder

start_time = time.monotonic()

# MonkeyDate_all = {'Elay':['230420']} #

Pathway='/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/data'
ResPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/decoder/'
AVmodPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/AVmodIndex/'
figSavePath = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Figures/decoder/'

# MonkeyDate_all = {'Elay':['230420','230503','230509','230525','230531',
#                         '230602','230606','230608','230613','230616','230620','230627',
#                         '230705','230711','230717','230718','230719','230726','230728',
#                         '230802','230808','230810','230814','230818','230822','230829'], 
#                   'Wu':['230809','230815','230821','230830']}

# # MonkeyDate_all = {'Elay':['230420'],'Wu':['230508']} #

# Pathway='/data/by-user/Huaizhen/preprocNeuralMatfiles/Vprobe+EEG/'
# ResPathway = '/data/by-user/Huaizhen/Fitresults/decoder/'
# AVmodPathway = '/data/by-user/Huaizhen/Fitresults/AVmodIndex/'
# figSavePath = '/data/by-user/Huaizhen/Figures/decoder/'

# filter neurons based on different rules
df_avMod_all = pickle.load(open(AVmodPathway+'AVmodTTestDF.pkl','rb'))
df_avMod_all_sig = df_avMod_all[df_avMod_all['pval']<0.05] 


timwinStart = np.arange(-1,1,0.2) # 
winlen = 0.3
extrastr_all = ['align2js']
x0str_all = ['JSon']
alignkeys_all = ['joystickOnsetInd']
NaneventsIndtarg_all = ['labelDictFilter["cooOnsetIndwithVirtual"]']

filterdict_list_all = [{'trialMod':['a','av'],'snr':[-15,-10],'respLabel':['hit','miss']}]
ylabel_all = ['respLabel']


for extrastr,alignkeys,NaneventsIndtarg,filterdict_list,ylabel,x0str in zip(extrastr_all,alignkeys_all,NaneventsIndtarg_all,filterdict_list_all,ylabel_all,x0str_all):
    fitresult_df = pd.DataFrame()
    # for Monkey,Date in MonkeyDate_all.items():
    #     for Date_temp in Date:  
    #         AllSinUnits_dict = {}
    #         print('...............'+'process session '+Monkey+'-'+Date_temp+'...............')

    #         spikeTimeDict,labelDictFilter, \
    #             timeSamp2Chorus,spikefs,behavefs= loadPreprocessMat(Monkey,Date_temp,Pathway)
    #         # get cluster has significant AV modulation
    #         df_avMod_sess = df_avMod_all_sig[(df_avMod_all_sig['Monkey']==Monkey) & (df_avMod_all_sig['sess']==Date_temp)]   

    #         for cc,(cls,mod) in enumerate(zip(df_avMod_sess['cluster'].values.tolist(),df_avMod_sess['mod'].values.tolist())):                
    #         # for cls in list(spikeTimeDict.keys()):
    #         #     mod = 'na'
    #             if any(substr in cls for substr in ['good','mua']):                    

    #                 print('cluster '+cls+' in progress............')
    #                 # # get trial by trial raster in each cluster
    #                 # labelDict_sub = {}
    #                 # ntrials = 50
    #                 # for key,value in labelDictFilter.items():
    #                 #     labelDict_sub[key] = value[-ntrials:]
    #                 # labelDictFilter = labelDict_sub.copy()
    #                 # spikeTime_temp = spikeTimeDict[cls][-ntrials:]   
                                        
    #                 spikeTime_temp = spikeTimeDict[cls]
    #                 start_time2 = time.monotonic()

    #                 spikeTimedf_temp1 = getClsRasterMultiprocess(timeSamp2Chorus,spikeTime_temp,spikefs,behavefs,\
    #                                                     labelDictFilter['chorusOnsetInd'],\
    #                                                     labelDictFilter[alignkeys] ,\
    #                                                     labelDictFilter['joystickOnsetInd'],\
    #                                                     [timwinStart[0],timwinStart[-1]+winlen],\
    #                                                     labelDictFilter,eval(NaneventsIndtarg))
    #                 end_time2 = time.monotonic()
    #                 print('time spend to getraster' )
    #                 print(timedelta(seconds=end_time2 - start_time2))
    #                 # decodrespLabel2str
    #                 spikeTimedf_temp = decodrespLabel2str(spikeTimedf_temp1)
    #                 # filter out trials
    #                 spikeTimedf_temp_filter = SortfilterDF(spikeTimedf_temp,filterlable =filterdict_list)

    #                 for tt, tStart in enumerate(timwinStart):
    #                     spikeNumdf_temp=countTrialSPKs(spikeTimedf_temp_filter,estwin='subwinoff',fs = behavefs,winTim=[tStart,tStart+winlen])
    #                     # apply svm on dataset
    #                     print('START SVM TRAINING................')
    #                     xx = spikeNumdf_temp['spkRate'].values.reshape([-1,1])
    #                     yy = pd.factorize(spikeNumdf_temp[ylabel])[0]
    #                     fit_result_slide_temp_df= pd.DataFrame(svmdecoder(xx,yy,[]),index=[0])
    #                     info_temp_df = pd.DataFrame({'Monkey':Monkey,'sess':Date_temp,'cluster':cls,'neurMod':mod,'time':np.round(tStart,decimals=1)},index=[0])
    #                     fit_result_slide_temp = pd.concat([info_temp_df,fit_result_slide_temp_df],axis=1)
    #                     fitresult_df = pd.concat([fitresult_df,fit_result_slide_temp])
    #                     del xx, yy
    #                     pickle.dump(fitresult_df,open(ResPathway+'svm_fit_'+extrastr+'_Cat-'+ylabel+'2.pkl','wb'))            

    #                 print('SVM TRAINING DONE FOR '+cls+'.................')
    #                 print('time spend for SVM decoding of cls '+cls )
    #                 end_time3 = time.monotonic()
    #                 print(timedelta(seconds=end_time3 - end_time2))            

    ## plot fit results 
    fitresult_df = pickle.load(open(ResPathway+'svm_fit_'+extrastr+'_Cat-'+ylabel+'2.pkl','rb'))

    def add0Str2xtick(xt,x0str):            
        xt = np.delete(xt,np.where(xt==0)[0])
        xt = np.append(xt,0)
        xtl = xt.tolist()
        xtl = [np.round(xtl[i],1) for i in range(len(xtl))]
        xtl[-1] = x0str
        return xt,xtl
    ## plot accuracy over time
    fig, axess = plt.subplots(2,2,figsize=(14,8),sharex='col') # 
    sns.swarmplot(fitresult_df[fitresult_df['Monkey']=='Elay'].reset_index(),x='time',y='accuracy_score',ax=axess[0,0],size=2)
    sns.lineplot(fitresult_df[fitresult_df['Monkey']=='Elay'].reset_index(),x='time',y='accuracy_score',ax=axess[0,1],estimator='mean',errorbar=('ci',95))
    chanceacc = fitresult_df[fitresult_df['Monkey']=='Elay']['accuracy_chance'].unique()[0]
    axess[0,1].plot([timwinStart[0],timwinStart[-1]],[chanceacc,chanceacc],label='chance',color='black')
    axess[0,0].set_title('Elay')
    axess[0,1].legend(frameon=False, framealpha=0,fontsize=5) 

    sns.swarmplot(fitresult_df[fitresult_df['Monkey']=='Wu'].reset_index(),x='time',y='accuracy_score',ax=axess[1,0],size=2)
    sns.lineplot(fitresult_df[fitresult_df['Monkey']=='Wu'].reset_index(),x='time',y='accuracy_score',ax=axess[1,1],estimator='mean',errorbar=('ci',95))
    chanceacc = fitresult_df[fitresult_df['Monkey']=='Wu']['accuracy_chance'].unique()[0]
    axess[1,1].plot([timwinStart[0],timwinStart[-1]],[chanceacc,chanceacc],label='chance',color='black')
    axess[1,0].set_title('Wu')
    xt = axess[1,1].get_xticks() 
    xt,xtl = add0Str2xtick(xt,x0str)
    axess[1,1].set_xticks(xt)
    axess[1,1].set_xticklabels(xtl)
    axess[1,1].legend(frameon=False, framealpha=0,fontsize=5) 

    fig.tight_layout()
    fig.savefig(figSavePath+'SVMdecoderACCovertime_'+extrastr+'_Cat-'+ylabel+'2.png')
    plt.close(fig)
    
    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))

print('done')
