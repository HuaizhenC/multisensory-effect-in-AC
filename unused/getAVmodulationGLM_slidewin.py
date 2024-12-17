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

MonkeyDate_all = {'Elay':['230420','230503','230509','230525','230531',
                        '230602','230606','230608','230613','230616','230620','230627',
                        '230705','230711','230717','230718','230719','230726','230728',
                        '230802','230808','230810']}

Pathway='/data/by-user/Huaizhen/preprocNeuralMatfiles/EEG+Vprobe/'
ResPathway = '/data/by-user/Huaizhen/Fitresults/glmfit/'
figSavePath = '/data/by-user/Huaizhen/Figures/glmfit/'

# alignkeys = 'cooOnsetIndwithVirtual'
# NaneventsIndtarg = '[]'
# x0str = 'cooOn'
# winlen = 0.3
# timwinStart = np.arange(-0.8,0.9,winlen) # 
# filter_dict = {'trialMod':['a','av'],'respLabel':[0,1]}
# extrastr_all = ['ali2co2IVs','ali2co3mods3IVs']
# GLM_IV_all = [['snr','AV'],['snr','AV','respLabel']]

alignkeys = 'JSOnsetIndwithVirtual'
NaneventsIndtarg = '[]'
x0str = 'JSOn'
winlen = 0.3
timwinStart = np.arange(-1.5,0.2,winlen) # 
filter_dict = {'trialMod':['a','av'],'respLabel':[0,1]}
extrastr_all = ['ali2js2IVs','ali2js3mods3IVs']
GLM_IV_all = [['snr','AV'],['snr','AV','respLabel']]

glmfitres_all = pd.DataFrame()                                                                                                             
for Monkey,Date in MonkeyDate_all.items():    
    for Date_temp in Date: 
        start_time = time.monotonic()
        print('...............'+'glm fit for '+Monkey+'-'+Date_temp+'...............')
        spikeTimeDict,labelDictFilter, \
            timeSamp2Chorus,spikefs,behavefs= loadPreprocessMat(Monkey,Date_temp,Pathway)

        # save sessions have clusters
        for cls in list(spikeTimeDict.keys()):
            if any(substr in cls for substr in ['good','mua']):   
                print('cluster '+cls+' in progress............')

                # # get trial by trial raster in each cluster
                # labelDict_sub = {}
                # ntrials = 100
                # for key,value in labelDictFilter.items():
                #     labelDict_sub[key] = value[-ntrials:]
                # labelDictFilter = labelDict_sub.copy()
                # spikeTime_temp = spikeTimeDict[cls][-ntrials:]   
                                    
                spikeTime_temp = spikeTimeDict[cls]

                spikeTimedf_temp = getClsRasterMultiprocess(timeSamp2Chorus,spikeTime_temp,spikefs,behavefs,\
                                                    labelDictFilter['chorusOnsetInd'],\
                                                    labelDictFilter[alignkeys] ,\
                                                    labelDictFilter['joystickOnsetInd'],\
                                                    [timwinStart[0],timwinStart[-1]+winlen],\
                                                    labelDictFilter,eval(NaneventsIndtarg))
                # filter trials
                spikeTimedf_temp_filtered,_ = SortfilterDF(spikeTimedf_temp,filterlable = filter_dict)
                # add IV columns 
                # spikeTimedf_temp_filtered =  addIVcol(spikeTimedf_temp_filtered)# modalityIV merged with snrIV
                spikeTimedf_temp_filtered =  addIVcol2(spikeTimedf_temp_filtered) # modalityIV separated from snrIV

                for mm,(extrastr,GLM_IV_list) in enumerate(zip(extrastr_all,GLM_IV_all)):                    
                    IVnum = len(GLM_IV_list)
                    for tt, tStart in enumerate(timwinStart):
                        spikeNumdf_temp=countTrialSPKs(spikeTimedf_temp_filtered,estwin='subwinoff',fs = behavefs,winTim=[tStart,tStart+winlen])                                                
                        print('timepoint '+str(tt))
                        try :
                            coeff_temp,pval_temp,evalparam = glmfit(spikeNumdf_temp,['spkRate'],GLM_IV_list)
                        except ValueError:
                            print('fail to fit glm model in'+Monkey+Date_temp+' unit:'+cls)
                            coeff_temp = pd.DataFrame([[np.nan]*IVnum],columns=['coef_'+iv for iv in GLM_IV_list])
                            pval_temp = pd.DataFrame([[np.nan]*IVnum],columns=['pval_'+iv for iv in GLM_IV_list])
                            evalparam = pd.DataFrame([[np.nan]*2],columns=['aic','bic'])
                        glmfitres_temp = pd.DataFrame.from_dict({'Monkey':[Monkey]*IVnum,'session_cls':[Date_temp+'_'+cls]*IVnum,\
                                                                'time':[np.around(tStart,decimals=2)]*IVnum,'iv':GLM_IV_list,\
                                                                'slope':[coeff_temp['coef_'+iv].values[0] for iv in GLM_IV_list],\
                                                                'pval':[pval_temp['pval_'+iv].values[0] for iv in GLM_IV_list],\
                                                                'aic':[evalparam.aic.values[0]]*IVnum,'bic':[evalparam.bic.values[0]]*IVnum,\
                                                                'GLMmod':[str(GLM_IV_list)]*IVnum})                        
                        glmfitres_all = pd.concat([glmfitres_all,glmfitres_temp])
                print('done with cluster '+cls)
        end_time = time.monotonic()
        print('fit time for session '+Date_temp)
        print(timedelta(seconds=end_time - start_time))
        pickle.dump(glmfitres_all,open(ResPathway+'glmfitCoefovertimeDF_'+alignkeys+'.pkl','wb'))                       
                

model_comp_param = pd.DataFrame()  
glmfitres_all = pickle.load(open(ResPathway+'glmfitCoefovertimeDF_'+alignkeys+'.pkl','rb'))
for mm,(extrastr,GLM_IV_list) in enumerate(zip(extrastr_all,GLM_IV_all)): 
    glmfitres = glmfitres_all[glmfitres_all['GLMmod']==str(GLM_IV_list)]
    model_comp_param_temp = glmfitres.groupby(['Monkey','session_cls','time'])[['aic','bic']].mean().reset_index()
    model_comp_param_temp['model'] = str(GLM_IV_list)+str(mm)
    model_comp_param = pd.concat((model_comp_param,model_comp_param_temp))
    
    glmfitres_sig = glmfitres[glmfitres['pval']<0.05].copy()
    ## plot sig slopes
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
        sns.stripplot(glmfitres_sig[glmfitres_sig['Monkey']==monkey].reset_index(),x='time',y='slope',hue='iv',hue_order=GLM_IV_list,ax=axess[axrow,0],dodge=True,size=1.5)
        xt = axess[1,0].get_xticks() 
        axess[1,0].set_xticks(xt)
        xticklabels = axess[1,0].get_xticklabels()
        formatted_xticklabels = [f'{float(tick_val._text):.1f}' for tick_val in xticklabels]
        axess[1,0].set_xticklabels(formatted_xticklabels)
        axess[axrow,0].set_title(monkey)
        axess[axrow,0].legend(frameon=False, framealpha=0,fontsize=5)

        sns.lineplot(glmfitres_sig[glmfitres_sig['Monkey']==monkey].reset_index(),x='time',y='slope',hue='iv',hue_order=GLM_IV_list,style='iv',ax=axess[axrow,1],estimator='mean',markers=True,dashes=False,errorbar=('ci',95))
        xt = axess[axrow,1].get_xticks()  
        xt,xtl = add0Str2xtick(xt,x0str)
        axess[axrow,1].set_xticks(xt)
        axess[axrow,1].set_xticklabels(xtl)
        axess[axrow,1].legend(frameon=False, framealpha=0,fontsize=5)

        sns.countplot(glmfitres_sig[glmfitres_sig['Monkey']==monkey].reset_index(),x='time',hue='iv',hue_order=GLM_IV_list,ax=axess[axrow,2])
        axess[axrow,2].yaxis.set_major_formatter(FuncFormatter(scale_formatter))
        axess[axrow,2].set_ylabel('units%')
        xt = axess[1,2].get_xticks() 
        axess[1,2].set_xticks(xt)
        xticklabels = axess[1,2].get_xticklabels()
        formatted_xticklabels = [f'{float(tick_val._text):.1f}' for tick_val in xticklabels]
        axess[1,2].set_xticklabels(formatted_xticklabels)
        axess[axrow,2].legend(frameon=False, framealpha=0,fontsize=5)

    fig, axess = plt.subplots(2,3,figsize=(16,5),sharex='col') # 
    plotsigSlp(0,'Elay',x0str,axess)
    plotsigSlp(1,'Wu',x0str,axess)

    fig.tight_layout()
    fig.savefig(figSavePath+'glmfitSlopeovertime_'+extrastr+'_'+alignkeys+''+'.png')
    plt.close(fig)

# model comparison measurements
fig, axess = plt.subplots(2,2,figsize=(10,5),sharex='col') # 
sns.stripplot(model_comp_param[model_comp_param['Monkey']=='Elay'].reset_index(),x='model',y='aic',hue='time',ax=axess[0,0],dodge=True,size=1.5)
sns.stripplot(model_comp_param[model_comp_param['Monkey']=='Elay'].reset_index(),x='model',y='bic',hue='time',ax=axess[0,1],dodge=True,size=1.5)
axess[0,0].legend(frameon=False, framealpha=0,fontsize=5)
axess[0,1].legend(frameon=False, framealpha=0,fontsize=5)

sns.stripplot(model_comp_param[model_comp_param['Monkey']=='Wu'].reset_index(),x='model',y='aic',hue='time',ax=axess[1,0],dodge=True,size=1.5)
sns.stripplot(model_comp_param[model_comp_param['Monkey']=='Wu'].reset_index(),x='model',y='bic',hue='time',ax=axess[1,1],dodge=True,size=1.5)
axess[1,0].legend(frameon=False, framealpha=0,fontsize=5)
axess[1,0].tick_params(axis='x',rotation=45)
axess[1,1].legend(frameon=False, framealpha=0,fontsize=5)
axess[1,1].tick_params(axis='x',rotation=45)

fig.tight_layout()
fig.savefig(figSavePath+'glmfitModComp_'+alignkeys+'.png')
plt.close(fig)

print('done')


