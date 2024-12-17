import time
start_time = time.monotonic()
from datetime import timedelta
import numpy as np
import pandas as pd
import os 
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.special import erf
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
from spikeUtilities import applyweibullfit,loadPreprocessMat
from sharedparam import getMonkeyDate_all,neuronfilterDF
from scipy import stats
import mat73
import scipy.io

def DprimRTCal(labelDF_mod,snr,FA_temp,dd,trialMod_temp):
    def dprimeCal(hitRate, FARate):
        if hitRate==1:
            hitRate = 0.9999
        if FARate==0:
            FARate = 0.0001  
        dPrime = norm.ppf(hitRate)-norm.ppf(FARate)
        return dPrime 
    
    def ProbCorctCal(hitRate, FARate):
        if hitRate==1:
            hitRate = 0.9999
        if FARate==0:
            FARate = 0.0001  
        probCorct = norm.cdf((norm.ppf(hitRate)-norm.ppf(FARate))/2)
        return probCorct  
    
    results_temp = pd.DataFrame()
    for snr_temp in snr:
        labelDF_mod_snr = labelDF_mod[labelDF_mod['snr']==snr_temp]
        hit_subtemp = labelDF_mod_snr[labelDF_mod_snr['respLabel']==0].shape[0]
        miss_subtemp = labelDF_mod_snr[labelDF_mod_snr['respLabel']==1].shape[0]
        try:
            hitrate_temp = hit_subtemp /(hit_subtemp+miss_subtemp)
        except ZeroDivisionError:   
            hitrate_temp = np.nan 
            print('no hits no misses in session '+dd+ ' condition_'+trialMod_temp+'_snr'+str(snr_temp))
        dprime_temp=dprimeCal(hitrate_temp,FA_temp) 
        dprime_temp2=ProbCorctCal(hitrate_temp,FA_temp) 
        
        rt_temp_list=labelDF_mod_snr[labelDF_mod_snr['respLabel']==0]['RT2coo'].values.tolist()
        if len(rt_temp_list)==0:
            rt_temp=np.nan
        else:
            rt_temp=np.nanmean(rt_temp_list)
        results_temp = pd.concat((results_temp,\
                                  pd.DataFrame({'session':[dd],'mod':[trialMod_temp],'snr':[snr_temp],'hitrate':[hitrate_temp],'dprime':[dprime_temp],'probcorct':[dprime_temp2],'RTmean':[rt_temp]}))) 
    return results_temp

def fit2mod(ProbCorrect_df_temp,xcol,ycol,threshx=0.7,lapseflag=False):
    # Input data and apply weibullfit
    # print('fit A condition....')
    x_a = ProbCorrect_df_temp[ProbCorrect_df_temp['mod']=='a'][xcol].values
    y_a = ProbCorrect_df_temp[ProbCorrect_df_temp['mod']=='a'][ycol].values
    # print('fit AV condition....')
    x_av = ProbCorrect_df_temp[ProbCorrect_df_temp['mod']=='av'][xcol].values
    y_av = ProbCorrect_df_temp[ProbCorrect_df_temp['mod']=='av'][ycol].values    
    if lapseflag:
        x_fit_a,y_fit_a, popt_a,_= applyweibullfit(x_a,y_a,threshx,1-y_a[np.argmax(x_a)]) 
        x_fit_av,y_fit_av,popt_av,_ = applyweibullfit(x_av,y_av,threshx,1-y_av[np.argmax(x_av)])
    else:
        x_fit_a,y_fit_a, popt_a,_= applyweibullfit(x_a,y_a,threshx) 
        x_fit_av,y_fit_av,popt_av,_ = applyweibullfit(x_av,y_av,threshx)         
    return x_fit_a,y_fit_a, popt_a, x_fit_av,y_fit_av,popt_av,[x_a,y_a,x_av,y_av]

# run in monty
MonkeyDate_all = getMonkeyDate_all()
Pathway='/home/huaizhen/Documents/MonkeyAVproj/data/preprocNeuralMatfiles/Vprobe+EEG2/'
outputPathway = '/home/huaizhen/Documents/MonkeyAVproj/data/Figures/BehavPerform/'
ResPathway = '/home/huaizhen/Documents/MonkeyAVproj/data/Fitresults/BehavPerform/'
ddmfitDTpathway = '/home/huaizhen/Documents/MonkeyAVproj/data/Fitresults/BehavPerform/one-choice-DDMfit_struct.mat'

# debug in local
# Pathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/data/'
# outputPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Figures/BehavPerform/'
# ResPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/BehavPerform/'
# ddmfitDTpathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/BehavPerform/one-choice-DDMfit_struct.mat'

fontsizeNo = 18
fontnameStr = 'DejaVu Sans'#'DejaVu Sans'
figformat = 'svg'

fig, axess = plt.subplots(2,3, figsize=(15,7),sharex='col', gridspec_kw={'width_ratios': [1,1,0.5]})

##### load ddm fit
ddmfitres = mat73.loadmat(ddmfitDTpathway)['fitparamStruct']
ddmfitres_df = pd.DataFrame({'Monkey':np.array(ddmfitres['Monkey']).flatten().tolist(),
                             'trialMod':np.array(ddmfitres['trialMod']).flatten().tolist(),
                             'session':np.array(ddmfitres['session']).flatten().tolist(),
                            'RT':np.array(ddmfitres['RT']).flatten().tolist(),
                             'DT':np.array(ddmfitres['DT']).flatten().tolist(),
                             'Ter':np.array(ddmfitres['Ter']).flatten().tolist(),
                             'driftrate':np.array(ddmfitres['driftrate']).flatten().tolist(),
                             'driftrateSD':np.array(ddmfitres['driftrateSD']).flatten().tolist()})
ddmfitres_df['trialMod'] = ddmfitres_df['trialMod'].replace({'a':'A','av':'AV'})

results_alldays_2monk = {}
# fitinfo = {'Var':'dprime','threshx':0,'ylabel':'dprime','ylim':[-1,7],'ytickstep':1,'lapseflag':False}
# namestr = 'dprime_' #'hitrate_' 'dprime_'
fitinfo = {'Var':'hitrate','threshx':0.65,'ylabel':'Hit Rate(%)','ylim':[0,101],'ytickstep':20,'lapseflag':True}
namestr = 'hitrate_' #'hitrate_' 'dprime_'

# results_alldays_2monk = pickle.load(open(ResPathway+'BehavPerformDF_2Monk.pkl','rb'))  

fitParam_2monkeys = pd.DataFrame()
for mm,(Monkey,Date) in enumerate(MonkeyDate_all.items()):   
    results_alldays = pd.DataFrame()
    visFArate = []
    audFArate = []

    for Date_temp in Date:  
        print('...............'+'estimating behavioral performance in '+Monkey+'-'+Date_temp+'...............')
        _,labelDictFilter, _,_,_= loadPreprocessMat(Monkey,Date_temp,Pathway,loadSPKflag='off')
        # respLabel decode:  
        # nosoundplayed: [nan]
        # A:  hit[0],miss/latemiss[1],FAa[2],erlyResp[88]
        # AV: hit[0],miss/latemiss[1],FAa[2],erlyResp[88],FAv[22]
        # V:  hit[0],CRv[10,100miss],FAa[2],FAv[22]
        labelDF = pd.DataFrame.from_dict(labelDictFilter)
        Vismodtials = labelDF[(labelDF['trialMod']=='v') & (labelDF['respLabel'].isin([0,10,100,22]))].shape[0] 
        Amodtials = labelDF[(labelDF['trialMod'].isin(['av','a']))
                             & (labelDF['respLabel'].isin([0,1,2,88]))].shape[0] 
        VisFA = labelDF[(labelDF['trialMod']=='v') & (labelDF['respLabel'].isin([22]))].shape[0]
        audFA = labelDF[(labelDF['trialMod'].isin(['av','a']))
                             & (labelDF['respLabel'].isin([2,88]))].shape[0]
        visFArate.append([VisFA/Vismodtials])
        audFArate.append([audFA/Amodtials])
        
        for trialMod_temp,FaLab in zip(['a','av'],[[2,88],[2,22,88]]):
            labelDF_mod = labelDF[labelDF['trialMod']==trialMod_temp]
            snr = np.sort(labelDF_mod['snr'].unique())
            #calculate dprime and rt for A condition, regardless of onset delay    
            FAs_temp = labelDF_mod[labelDF_mod['respLabel'].isin(FaLab)].shape[0]                          
            Hits_temp = labelDF_mod[labelDF_mod['respLabel']==0].shape[0]
            #calculate FA for A condition: (earlyHit/Hit+earlyHit)        
            try:
                if trialMod_temp == 'a':
                    FAnumerator = FAs_temp
                    FAdenominator = Hits_temp+FAs_temp                
                if trialMod_temp == 'av':
                    FAnumerator = VisFA+FAs_temp
                    FAdenominator = Hits_temp+Vismodtials+FAs_temp
                FA_rate = FAnumerator/FAdenominator
            except ZeroDivisionError:
                FA_rate = np.nan
            results_alldays= pd.concat((results_alldays,DprimRTCal(labelDF_mod,snr,FA_rate,Date_temp,trialMod_temp)))                                    
    print(Monkey+' visCR mean rate:'+str(1-np.round(np.mean(visFArate),decimals=2))
          +' visCR median rate:'+str(1-np.round(np.median(visFArate),decimals=2))
          +' visCR rate SD:'+str(np.std(visFArate))
          +' audFA mean rate:'+str(np.round(np.mean(audFArate),decimals=2))
          +' audFA median rate:'+str(np.round(np.median(audFArate),decimals=2))
          +' audFA rate SD:'+str(np.std(audFArate)))
    results_alldays_2monk[Monkey] = results_alldays

    # results_alldays = results_alldays_2monk[0][Monkey]

    ddmfitres_monk = ddmfitres_df[ddmfitres_df['Monkey']==Monkey].reset_index(drop=True).sort_values(by=['Monkey','session','trialMod'])
    
    fitParam = pd.DataFrame()
    fityval = pd.DataFrame()
    RTinfo = pd.DataFrame()
    # fit individual sessions for one monkey
    for sess in MonkeyDate_all[Monkey]:
        print('fitting session '+sess)
        #plot ddm RT fit
        ddmfitres_monk_sess = ddmfitres_monk[ddmfitres_monk['session']==sess]
        axess[mm,2].plot(ddmfitres_monk_sess['trialMod'].values,ddmfitres_monk_sess['DT'].values,marker='o',linewidth=2,color='lightgray')
        # fit and plot performance: probcorrect
        ProbCorrect_df_temp = results_alldays[results_alldays['session']==sess]
        x_fit_a,y_fit_a, popt_a, x_fit_av,y_fit_av,popt_av,rawData = fit2mod(ProbCorrect_df_temp,'snr',fitinfo['Var'],threshx=fitinfo['threshx'],lapseflag=fitinfo['lapseflag'])
        axess[mm,0].plot(rawData[0], rawData[1],color='lightsteelblue')#rawData: x_a,y_a,x_av,y_av
        axess[mm,0].plot(rawData[2], rawData[3],color='mistyrose')        
        axess[mm,1].plot(x_fit_a, y_fit_a*100,color='lightsteelblue',linewidth=2,alpha=0.5,zorder=0)
        axess[mm,1].plot(x_fit_av, y_fit_av*100,color='mistyrose',linewidth=2,alpha=0.5,zorder=0)
        # save performance fit results for A/AV condition
        fitParam = pd.concat((fitParam,pd.DataFrame({'Monkey':[Monkey]*len(rawData[0]),
                                                     'sess':[sess]*len(rawData[0]),
                                                     'mod':['a']*len(rawData[0]),
                                                     'slope':[popt_a[0]]*len(rawData[0]),
                                                     'thresh':[popt_a[1]]*len(rawData[0]),
                                                     'snr':rawData[0],fitinfo['Var']:rawData[1]}))) # save A condition fit res
        fityval = pd.concat((fityval,pd.DataFrame({'Monkey':[Monkey]*len(y_fit_a),
                                                     'sess':[sess]*len(y_fit_a),
                                                     'mod':['A']*len(y_fit_a),
                                                     'slope':[popt_a[0]]*len(y_fit_a),
                                                     'thresh':[popt_a[1]]*len(y_fit_a),
                                                     'snr':x_fit_a,fitinfo['Var']+'_fit':y_fit_a*100}))) # save A condition fit res
        fitParam = pd.concat((fitParam,pd.DataFrame({'Monkey':[Monkey]*len(rawData[2]),
                                                     'sess':[sess]*len(rawData[2]),
                                                     'mod':['av']*len(rawData[2]),
                                                     'slope':[popt_av[0]]*len(rawData[2]),
                                                     'thresh':[popt_av[1]]*len(rawData[2]),
                                                     'snr':rawData[2],fitinfo['Var']:rawData[3]}))) # save AV condition fit res
        fityval = pd.concat((fityval,pd.DataFrame({'Monkey':[Monkey]*len(y_fit_av),
                                                     'sess':[sess]*len(y_fit_av),
                                                     'mod':['AV']*len(y_fit_av),
                                                     'slope':[popt_av[0]]*len(y_fit_av),
                                                     'thresh':[popt_av[1]]*len(y_fit_av),
                                                     'snr':x_fit_av,fitinfo['Var']+'_fit':y_fit_av*100}))) # save A condition fit res
       
        # fit and plot response time
        x_fit_a,y_fit_a, popt_a, x_fit_av,y_fit_av,popt_av,rawRT = fit2mod(ProbCorrect_df_temp,'snr','RTmean',threshx=0.9,lapseflag=False)
        # save RT fit results for A/AV condition
        RTinfo = pd.concat((RTinfo,pd.DataFrame({'Monkey':[Monkey]*len(rawRT[0]),
                                                'sess':[sess]*len(rawRT[0]),
                                                 'mod':['a']*len(rawRT[0]),
                                                 'slope':[popt_a[0]]*len(rawRT[0]),
                                                 'thresh':[popt_a[1]]*len(rawRT[0]),
                                                 'snr':rawRT[0],'RTmean':rawRT[1]})))
        RTinfo = pd.concat((RTinfo,pd.DataFrame({'Monkey':[Monkey]*len(rawRT[0]),
                                                'sess':[sess]*len(rawRT[2]),
                                                 'mod':['av']*len(rawRT[2]),
                                                 'slope':[popt_av[0]]*len(rawRT[2]),
                                                 'thresh':[popt_av[1]]*len(rawRT[2]),
                                                 'snr':rawRT[2],'RTmean':rawRT[3]})))
          
    # mean DT
    dt_stats,dt_pval = stats.wilcoxon(ddmfitres_monk[ddmfitres_monk['trialMod']=='A']['DT'].values,
                                      ddmfitres_monk[ddmfitres_monk['trialMod']=='AV']['DT'].values
                                      ,alternative='greater') 
             
    print(Monkey+' pval:'+str(dt_pval)+'\n      DT median A:'+str(np.median(ddmfitres_monk[ddmfitres_monk['trialMod']=='A']['DT'].values))
                +'\n DT [25 75]percentile A: ['+str(np.percentile(ddmfitres_monk[ddmfitres_monk['trialMod']=='A']['DT'].values,25))+'   '
                                                +str(np.percentile(ddmfitres_monk[ddmfitres_monk['trialMod']=='A']['DT'].values,75))+']'
                +'\n RT median A:'+str(np.median(RTinfo[RTinfo['mod']=='a']['RTmean'].values))
                +'\n RT [25 75]percentile A: ['+str(np.percentile(RTinfo[RTinfo['mod']=='a']['RTmean'].values,25))+'   '
                                                +str(np.percentile(RTinfo[RTinfo['mod']=='a']['RTmean'].values,75))+']'
                +'\n      DT median AV:'+str(np.median(ddmfitres_monk[ddmfitres_monk['trialMod']=='AV']['DT'].values))
                +'\n DT [25 75]percentile AV: ['+str(np.percentile(ddmfitres_monk[ddmfitres_monk['trialMod']=='AV']['DT'].values,25))+'   '
                                                +str(np.percentile(ddmfitres_monk[ddmfitres_monk['trialMod']=='AV']['DT'].values,75))+']'
                +'\n RT median AV:'+str(np.median(RTinfo[RTinfo['mod']=='av']['RTmean'].values))
                +'\n RT [25 75]percentile AV: ['+str(np.percentile(RTinfo[RTinfo['mod']=='av']['RTmean'].values,25))+'   '
                                                +str(np.percentile(RTinfo[RTinfo['mod']=='av']['RTmean'].values,75))+']')

    ddmfitres_monk_mean = ddmfitres_monk.groupby('trialMod')['DT'].mean().reset_index()
    axess[mm,2].plot(ddmfitres_monk_mean['trialMod'].values,ddmfitres_monk_mean['DT'].values,marker='o',color='black',linewidth=3,zorder=100)
    axess[mm,2].set_ylabel('Decision Time(s)',fontsize=fontsizeNo,fontname=fontnameStr)
    if Monkey=='Elay':
        axess[mm,2].set_yticks(list(np.arange(0.1,0.41,0.1)))
    else:
        axess[mm,2].set_yticks(list(np.arange(0.2,0.51,0.1)))
    legend = axess[mm,2].legend(title='monkey '+Monkey[0],frameon=False, framealpha=0,loc='upper right', prop={'size':fontsizeNo,'family':fontnameStr},title_fontsize=fontsizeNo+2)    
    if dt_pval<0.05:
        axess[mm,2].text(0.5, 0.5, '*',#'p = '+str(np.round(dt_pval,decimals=4)), 
                    transform=axess[mm,2].transAxes,  # Specify the position in axis coordinates
                    fontsize=fontsizeNo-4,             # Font size
                    fontname=fontnameStr,
                    verticalalignment='top', # Align text at the top
                    horizontalalignment='right') # Align text to the right  
    axess[mm,2].tick_params(axis='both', which='major', labelsize=fontsizeNo-2)
    for tick in axess[mm, 2].get_xticklabels() + axess[mm, 2].get_yticklabels():
        tick.set_fontname(fontnameStr)                 
    #  mean of each fitting curve
    sns.lineplot(fityval.reset_index(),x='snr',y=fitinfo['Var']+'_fit',hue='mod',hue_order=['A','AV'],palette=['blue','red'],ax=axess[mm,1],estimator='mean', errorbar=('ci', 95),linewidth=3,zorder=300)
    axess[mm,1].set_ylabel(fitinfo['ylabel'],fontsize=fontsizeNo,fontname=fontnameStr)
    # axess[mm,1].set_xlabel('SNR(dB)',fontsize=fontsizeNo,fontname=fontnameStr)
    axess[mm,1].set_yticks(list(np.arange(fitinfo['ylim'][0],fitinfo['ylim'][1],fitinfo['ytickstep'])))
    axess[mm,1].set_xticks(list(np.arange(-15,15,5)))
    legend = axess[mm,1].legend(title='monkey '+Monkey[0],frameon=False, framealpha=0,loc='lower right', prop={'size':fontsizeNo,'family':fontnameStr},title_fontsize=fontsizeNo+2)    
    axess[mm,1].tick_params(axis='both', which='major', labelsize=fontsizeNo-2)
    for tick in axess[mm, 1].get_xticklabels() + axess[mm, 1].get_yticklabels():
        tick.set_fontname(fontnameStr)    

axess[1,0].set_xlabel('SNR (dB)',fontsize=fontsizeNo,fontname=fontnameStr)
axess[1,1].set_xlabel('SNR (dB)',fontsize=fontsizeNo,fontname=fontnameStr)
axess[1,2].set_xlabel('Condition',fontsize=fontsizeNo,fontname=fontnameStr)
fig.tight_layout()
fig.savefig(outputPathway+namestr+'BehavPerformweibullfit.'+figformat,dpi=300,format = figformat)
plt.close(fig)


end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))






