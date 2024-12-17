import time
from datetime import timedelta
import numpy as np
import seaborn as sns
import os
import pickle
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt 
from matplotlib.ticker import FuncFormatter
from spikeUtilities import getClsRasterMultiprocess,countTrialSPKs,loadPreprocessMat,glmfit,SortfilterDF,addIVcol,addIVcol2

# MonkeyDate_all = {'Elay':['230420'],'Wu':['230508']} #

# Pathway='/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/data'
# ResPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/glmfit/'
# figSavePath = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Figures/glmfit/'

MonkeyDate_all = {'Elay':['230420','230503','230509','230525','230531','230602','230606','230608'],\
                  'Wu':['230508','230512','230515','230517','230522','230530','230601','230605','230607']} #

Pathway='/data/by-user/Huaizhen/preprocNeuralMatfiles/EEG+Vprobe/'
ResPathway = '/data/by-user/Huaizhen/Fitresults/glmfit/'
figSavePath = '/data/by-user/Huaizhen/Figures/glmfit/'

# extrastr_all = ['ali2co','ali2js']
# alignkeys_all = ['cooOnsetIndwithVirtual','joystickOnsetInd']
# NaneventsIndtarg_all = ['[]','labelDictFilter["cooOnsetIndwithVirtual"]']
# x0str_all = ['cooOn','JSon']
# timwinStart_all = [np.arange(-0.8,0.9,0.2),np.arange(-0.8,0.5,0.2)] # 
# winlen = 0.2
# filter_dict_all = [{'trialMod':['a','av','v'],'respLabel':[0,10,100]},\
#                    {'trialMod':['a','av'],'snr':[-15,-10],'respLabel':[0,1]}]
# GLM_IV_all = [['A','V','AV'],['A','V','jsOnset']]

extrastr_all = ['ali2co3mods_300ms_addIVcol','ali2co2modssnr_300ms_addIVcol','ali2co2mods_300ms_addIVcol',]
alignkeys_all = ['cooOnsetIndwithVirtual','cooOnsetIndwithVirtual','cooOnsetIndwithVirtual']
NaneventsIndtarg_all = ['[]','[]','[]']
x0str_all = ['cooOn','cooOn','cooOn']
winlen = 0.3
timwinStart_all = [np.arange(-0.8,0.9,winlen),np.arange(-0.8,0.9,winlen),np.arange(-0.8,0.9,winlen)] # 
filter_dict_all = [{'trialMod':['a','av','v'],'respLabel':[0,10,100]},\
                   {'trialMod':['a','av'],'respLabel':[0]},\
                    {'trialMod':['a','av'],'respLabel':[0]}]
GLM_IV_all = [['A','V','AV','time'],['snr','A','AV','time'],['A','AV','time']]

# extrastr_all = ['ali2co3mods_300ms_addIVcol_test']
# alignkeys_all = ['cooOnsetIndwithVirtual']
# NaneventsIndtarg_all = ['[]']
# x0str_all = ['cooOn']
# winlen = 0.3
# timwinStart_all = [np.arange(-0.8,0.9,winlen)] # 
# filter_dict_all = [{'trialMod':['a','av','v'],'respLabel':[0,10,100]}]
# GLM_IV_all = [['A','V','AV','time']]

model_comp_param = pd.DataFrame()
for extrastr,alignkeys,NaneventsIndtarg,timwinStart,filter_dict,GLM_IV_list,x0str in zip(extrastr_all,alignkeys_all,NaneventsIndtarg_all,timwinStart_all,filter_dict_all,GLM_IV_all,x0str_all):
    glmfitres = pd.DataFrame()
    start_time = time.monotonic()
    spikeNumdf_all = pd.DataFrame()                                                                                                               

    # for Monkey,Date in MonkeyDate_all.items():
    #     for Date_temp in Date: 
    #         print('...............'+'glm fit for '+Monkey+'-'+Date_temp+'...............')

    #         spikeTimeDict,labelDictFilter, \
    #             timeSamp2Chorus,spikefs,behavefs= loadPreprocessMat(Monkey,Date_temp,Pathway)

    #         # save sessions have clusters
    #         for cls in list(spikeTimeDict.keys()):
    #             if any(substr in cls for substr in ['good','mua']):   
    #                 print('cluster '+cls+' in progress............')
    #                 start_time_temp = time.monotonic()
    #                 # # get trial by trial raster in each cluster
    #                 # labelDict_sub = {}
    #                 # ntrials = 50
    #                 # for key,value in labelDictFilter.items():
    #                 #     labelDict_sub[key] = value[-ntrials:]
    #                 # labelDictFilter = labelDict_sub.copy()
    #                 # spikeTime_temp = spikeTimeDict[cls][-ntrials:]   
                                        
    #                 spikeTime_temp = spikeTimeDict[cls]

    #                 spikeTimedf_temp = getClsRasterMultiprocess(timeSamp2Chorus,spikeTime_temp,spikefs,behavefs,\
    #                                                     labelDictFilter['chorusOnsetInd'],\
    #                                                     labelDictFilter[alignkeys] ,\
    #                                                     labelDictFilter['joystickOnsetInd'],\
    #                                                     [timwinStart[0],timwinStart[-1]+winlen],\
    #                                                     labelDictFilter,eval(NaneventsIndtarg))
    #                 # filter trials
    #                 spikeTimedf_temp_filtered,_ = SortfilterDF(spikeTimedf_temp,filterlable = filter_dict)
    #                 # add IV columns 
    #                 spikeTimedf_temp_filtered =  addIVcol(spikeTimedf_temp_filtered)
    #                 #spikeTimedf_temp_filtered =  addIVcol2(spikeTimedf_temp_filtered)

    #                 spikeNumdf_temp_alltim = pd.DataFrame()
    #                 for tt, tStart in enumerate(timwinStart):
    #                     spikeNumdf_temp=countTrialSPKs(spikeTimedf_temp_filtered,estwin='subwinoff',fs = behavefs,winTim=[tStart,tStart+winlen])
    #                     spikeNumdf_temp['time'] = np.around(tStart,decimals=2)
    #                     spikeNumdf_temp['Monkey'] = Monkey
    #                     spikeNumdf_temp['sess_cls'] = Date_temp+'_'+cls
    #                     spikeNumdf_temp_alltim = pd.concat((spikeNumdf_temp_alltim,spikeNumdf_temp))
                    
    #                 spikeNumdf_all = pd.concat((spikeNumdf_all,spikeNumdf_temp_alltim))    
    #                 IVnum = len(GLM_IV_list)
    #                 try :
    #                         coeff_temp,pval_temp,evalparam = glmfit(spikeNumdf_temp_alltim,['spkRate'],GLM_IV_list)
    #                 except ValueError:
    #                     print('fail to fit glm model')
    #                     coeff_temp = pd.DataFrame([[np.nan]*IVnum],columns=['coef_'+iv for iv in GLM_IV_list])
    #                     pval_temp = pd.DataFrame([[np.nan]*IVnum],columns=['pval_'+iv for iv in GLM_IV_list])
    #                     evalparam = pd.DataFrame([[np.nan]*2],columns=['aic','bic'])
    #                 glmfitres_temp = pd.DataFrame.from_dict({'Monkey':[Monkey]*IVnum,'session_cls':[Date_temp+'_'+cls]*IVnum,\
    #                                                         'iv':GLM_IV_list,\
    #                                                         'slope':[coeff_temp['coef_'+iv].values[0] for iv in GLM_IV_list],\
    #                                                         'pval':[pval_temp['pval_'+iv].values[0] for iv in GLM_IV_list],\
    #                                                         'aic':[evalparam.aic.values[0]]*IVnum,'bic':[evalparam.bic.values[0]]*IVnum})                        
    #                 glmfitres = pd.concat([glmfitres,glmfitres_temp])
    #                 pickle.dump(glmfitres,open(ResPathway+'glmfitCoefDF_'+extrastr+'.pkl','wb'))            
    #                 pickle.dump(spikeNumdf_all,open(ResPathway+'glmfitSPKoverTimDF_'+extrastr+'.pkl','wb'))            
                
    #                 print('done with cluster '+cls)
    #                 end_time_temp = time.monotonic()
    #                 # print(timedelta(seconds=end_time_temp - start_time_temp))
    
    glmfitres = pickle.load(open(ResPathway+'glmfitCoefDF_'+extrastr+'.pkl','rb'))
    model_comp_param_temp = glmfitres.groupby(['Monkey','session_cls'])['aic','bic'].mean().reset_index()
    model_comp_param_temp['model'] = extrastr #str(GLM_IV_list)
    model_comp_param = pd.concat((model_comp_param,model_comp_param_temp))
    
    glmfitres_sig = glmfitres[glmfitres['pval']<0.05].copy()
    # plot sig slopes
    def plotsigSlp(axrow,monkey,x0str,axess):
        # Define custom formatter function
        def scale_formatter(x, pos):
            return f'{x/totalNumCls:.2f}'
        def add0Str2xtick(xt,x0str):            
            xt = np.delete(xt,np.where(xt==0)[0])
            xt = np.append(xt,0)
            xtl = xt.tolist()
            xtl = [np.round(xtl[i],1) for i in range(len(xtl))]
            xtl[-1] = x0str
            return xt,xtl
        totalNumCls = glmfitres[glmfitres['Monkey']==monkey].groupby(['session_cls'])['slope'].mean().reset_index().shape[0]
        sns.lineplot(glmfitres_sig[glmfitres_sig['Monkey']==monkey].reset_index(),x='iv',y='slope',hue='session_cls',style='session_cls',ax=axess[axrow,0],dashes=False,legend= False,size=2,lw=0.4)
        sns.stripplot(glmfitres_sig[glmfitres_sig['Monkey']==monkey].reset_index(),x='iv',y='slope',hue='session_cls',ax=axess[axrow,0],dodge=False,legend= False,size=2)

        axess[axrow,0].set_title(monkey)

        sns.countplot(glmfitres_sig[glmfitres_sig['Monkey']==monkey].reset_index(),x='iv',ax=axess[axrow,1])
        axess[axrow,1].yaxis.set_major_formatter(FuncFormatter(scale_formatter))
        axess[axrow,1].set_ylabel('units%')
        axess[axrow,1].legend(frameon=False, framealpha=0,fontsize=5)

    fig, axess = plt.subplots(2,2,figsize=(8,5),sharex='col') # 

    plotsigSlp(0,'Elay',x0str,axess)
    plotsigSlp(1,'Wu',x0str,axess)

    fig.tight_layout()
    fig.savefig(figSavePath+'glmfitSlope_'+extrastr+''+'.png')
    plt.close(fig)
    end_time = time.monotonic()
    print('total time spend for one condition')
    print(timedelta(seconds=end_time - start_time))

# model comparison measurements
fig, axess = plt.subplots(2,2,figsize=(10,8),sharex='col') # 
sns.stripplot(model_comp_param[model_comp_param['Monkey']=='Elay'].reset_index(),x='model',y='aic',hue='session_cls',ax=axess[0,0],dodge=False,legend= False,size=2.5)
sns.lineplot(model_comp_param[model_comp_param['Monkey']=='Elay'].reset_index(),x='model',y='aic',hue='session_cls',style='session_cls',ax=axess[0,0],dashes=False,legend= False,size=2.5,lw=0.4)
sns.stripplot(model_comp_param[model_comp_param['Monkey']=='Elay'].reset_index(),x='model',y='bic',hue='session_cls',ax=axess[0,1],dodge=False,legend= False,size=2.5)
sns.lineplot(model_comp_param[model_comp_param['Monkey']=='Elay'].reset_index(),x='model',y='bic',hue='session_cls',style='session_cls',ax=axess[0,1],dashes=False,legend= False,size=2.5,lw=0.4)
xticklabels = axess[1,0].get_xticklabels()
axess[1,0].set_xticklabels(xticklabels,rotation=30)

sns.stripplot(model_comp_param[model_comp_param['Monkey']=='Wu'].reset_index(),x='model',y='aic',hue='session_cls',ax=axess[1,0],dodge=False,legend= False,size=2.5)
sns.lineplot(model_comp_param[model_comp_param['Monkey']=='Wu'].reset_index(),x='model',y='aic',hue='session_cls',style='session_cls',ax=axess[1,0],dashes=False,legend= False,size=2.5,lw=0.4)
sns.stripplot(model_comp_param[model_comp_param['Monkey']=='Wu'].reset_index(),x='model',y='bic',hue='session_cls',ax=axess[1,1],dodge=False,legend= False,size=2.5)
sns.lineplot(model_comp_param[model_comp_param['Monkey']=='Wu'].reset_index(),x='model',y='bic',hue='session_cls',style='session_cls',ax=axess[1,1],dashes=False,legend= False,size=2.5,lw=0.4)
xticklabels = axess[1,1].get_xticklabels()
axess[1,1].set_xticklabels(xticklabels,rotation=30)

fig.tight_layout()
fig.savefig(figSavePath+'glmfitModComp.png')
plt.close(fig)

print('done')


