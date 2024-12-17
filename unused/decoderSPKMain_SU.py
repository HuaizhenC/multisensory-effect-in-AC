import time
from datetime import timedelta
import numpy as np
import seaborn as sns
import mat73
import os
import pickle
import pandas as pd
from matplotlib import pyplot as plt 
from spikeUtilities import getClsRasterMultiprocess,countTrialSPKs,loadPreprocessMat,SortfilterDF,decodrespLabel2str
from decoders import svmdecoder

def add0Str2xtick(xt,x0str):            
    xt = np.delete(xt,np.where(xt==0)[0])
    xt = np.append(xt,0)
    xtl = xt.tolist()
    xtl = [np.round(xtl[i],1) for i in range(len(xtl))]
    xtl[-1] = x0str
    return xt,xtl


# MonkeyDate_all = {'Elay':['230420'],'Wu':['230809']} #

Pathway='/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/data'
ResPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/decoder/'
AVmodPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/AVmodIndex/'
figSavePath = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Figures/decoder/'

MonkeyDate_all = {'Elay':['230420','230503','230509','230525','230531',
                        '230602','230606','230608','230613','230616','230620','230627',
                        '230705','230711','230717','230718','230719','230726','230728',
                        '230802','230808','230810','230814','230818','230822','230829'], 
                  'Wu':['230809','230815','230821','230830']}

# Pathway='/data/by-user/Huaizhen/preprocNeuralMatfiles/Vprobe+EEG/'
# ResPathway = '/data/by-user/Huaizhen/Fitresults/decoder/'
# AVmodPathway = '/data/by-user/Huaizhen/Fitresults/AVmodIndex/'
# figSavePath = '/data/by-user/Huaizhen/Figures/decoder/'

timwinStart = np.arange(-1,1,0.3) # 
winlen = 0.3
extrastr = 'align2coo'
x0str = 'cooOn'
alignkeys = 'cooOnsetIndwithVirtual'
filterdict = {'trialMod':['a','av'],'respLabel':['hit','miss']}
ylabel_all = ['trialMod','snr']

# timwinStart = np.arange(-1,1,0.3) # 
# winlen = 0.3
# extrastr_all = 'align2js'
# x0str_all = 'JSon'
# alignkeys_all = 'JSOnsetIndwithVirtual'
# filterdict = {'trialMod':['a','av'],'respLabel':['hit']}
# ylabel_all = ['trialMod','snr' ]

# filter neurons based on different rules
df_avMod_all = pickle.load(open(AVmodPathway+'AVmodTTestDF.pkl','rb'))
df_avMod_all_sig = df_avMod_all[df_avMod_all['pval']<0.05] 
# print(df_avMod_all_sig.to_string())

for ylabel in ylabel_all:
    fitresult_df = pd.DataFrame()
    for Monkey,Date in MonkeyDate_all.items():
        for Date_temp in Date:  
            AllSinUnits_dict = {}
            print('...............'+'process session '+Monkey+'-'+Date_temp+'...............')
            start_time = time.monotonic()

            spikeTimeDict,labelDictFilter, \
                timeSamp2Chorus,spikefs,behavefs= loadPreprocessMat(Monkey,Date_temp,Pathway)
            # get cluster has significant AV modulation
            df_avMod_sess = df_avMod_all_sig[(df_avMod_all_sig['Monkey']==Monkey) & (df_avMod_all_sig['session_cls'].str.contains(Date_temp))]   

            for cc,(sesscls,mod) in enumerate(zip(df_avMod_sess['session_cls'].values.tolist(),df_avMod_sess['mod'].values.tolist())):                
                cls = sesscls[7:]
            # for cls,mod in zip(list(spikeTimeDict.keys()),['a']*len(list(spikeTimeDict.keys()))):                               
                # # get trial by trial raster in each cluster
                # labelDict_sub = {}
                # ntrials = 100
                # for key,value in labelDictFilter.items():
                #     labelDict_sub[key] = value[-ntrials:]
                # labelDictFilter = labelDict_sub.copy()
                # spikeTime_temp = spikeTimeDict[cls][-ntrials:]   
                                    
                spikeTime_temp = spikeTimeDict[cls]

                spikeTimedf_temp1 = getClsRasterMultiprocess(timeSamp2Chorus,spikeTime_temp,spikefs,behavefs,\
                                                    labelDictFilter['chorusOnsetInd'],\
                                                    labelDictFilter[alignkeys] ,\
                                                    labelDictFilter['JSOnsetIndwithVirtual'],\
                                                    [timwinStart[0],timwinStart[-1]+winlen],\
                                                    labelDictFilter)                                     
                # decodrespLabel2str
                spikeTimedf_temp = decodrespLabel2str(spikeTimedf_temp1)
                # filter out trials
                spikeTimedf_temp_filter,_ = SortfilterDF(spikeTimedf_temp,filterlable =filterdict)
                
                end_time2 = time.monotonic()
                for tt, tStart in enumerate(timwinStart):
                    spikeNumdf_temp=countTrialSPKs(spikeTimedf_temp_filter,estwin='subwinoff',fs = behavefs,winTim=[tStart,tStart+winlen])
                    # apply svm on dataset
                    # print('START SVM TRAINING AT Tim'+str(tt)+'................')
                    xx = spikeNumdf_temp['spkRate'].values.reshape([-1,1])
                    yy = pd.factorize(spikeNumdf_temp[ylabel])[0]
                    fit_result_temp_df = pd.DataFrame()
                    for svmRP in range(20):
                        fit_result_temp_df = pd.concat([fit_result_temp_df,pd.DataFrame(svmdecoder(xx,yy,[]),index=[0])],axis=0)
                    fit_result_slide_temp_df= pd.DataFrame(fit_result_temp_df.mean()).transpose()
                    info_temp_df = pd.DataFrame({'Monkey':Monkey,'sess_cls':Date_temp+'_'+cls,'neurMod':mod,'time':np.round(tStart,decimals=1)},index=[0])
                    fit_result_slide_temp = pd.concat([info_temp_df,fit_result_slide_temp_df],axis=1)
                    fitresult_df = pd.concat([fitresult_df,fit_result_slide_temp])
                    del xx, yy
                pickle.dump(fitresult_df,open(ResPathway+'svm_fit_'+extrastr+'_Cat-'+ylabel+'.pkl','wb'))            
                
                end_time3 = time.monotonic()
                print('SVM TRAINING DONE FOR CLS'+Monkey+'-'+Date_temp+'-'+cls+'.................')
                print('time spend ' )               
                print(timedelta(seconds=end_time3 - end_time2))            
            print('time spend for SVM decoding of session '+Date_temp)
            print(timedelta(seconds=end_time3 - start_time)) 
            
    # ## plot fit results 
    # fitresult_df = pickle.load(open(ResPathway+'svm_fit_'+extrastr+'_Cat-'+ylabel+'.pkl','rb'))

    ## plot accuracy over time
    fig, axess = plt.subplots(2,2,figsize=(14,8),sharex='col') # 
    sns.swarmplot(fitresult_df[fitresult_df['Monkey']=='Elay'].reset_index(),x='time',y='accuracy_score',hue='sess_cls',ax=axess[0,0],size=3,legend=False)
    sns.lineplot(fitresult_df[fitresult_df['Monkey']=='Elay'].reset_index(),x='time',y='accuracy_score',ax=axess[0,1],estimator='mean',errorbar=('ci',95))
    chanceacc = fitresult_df[fitresult_df['Monkey']=='Elay']['accuracy_chance'].mean()
    print()
    axess[0,1].plot([timwinStart[0],timwinStart[-1]],[chanceacc,chanceacc],label='chance',color='black')
    axess[0,0].set_title('Elay')
    axess[0,1].legend(frameon=False, framealpha=0,fontsize=5) 

    sns.stripplot(fitresult_df[fitresult_df['Monkey']=='Wu'].reset_index(),x='time',y='accuracy_score',hue='sess_cls',ax=axess[1,0],size=3,legend=False)
    sns.lineplot(fitresult_df[fitresult_df['Monkey']=='Wu'].reset_index(),x='time',y='accuracy_score',ax=axess[1,1],estimator='mean',errorbar=('ci',95))
    chanceacc = fitresult_df[fitresult_df['Monkey']=='Wu']['accuracy_chance'].mean()
    axess[1,1].plot([timwinStart[0],timwinStart[-1]],[chanceacc,chanceacc],label='chance',color='black')
    axess[1,0].set_title('Wu')
    xt = axess[1,1].get_xticks() 
    xt,xtl = add0Str2xtick(xt,x0str)
    axess[1,1].set_xticks(xt)
    axess[1,1].set_xticklabels(xtl)
    axess[1,1].legend(frameon=False, framealpha=0,fontsize=5) 

    fig.tight_layout()
    fig.savefig(figSavePath+'SVMdecoderACCovertime_'+extrastr+'_Cat-'+ylabel+'.png')
    plt.close(fig)

print('done')
