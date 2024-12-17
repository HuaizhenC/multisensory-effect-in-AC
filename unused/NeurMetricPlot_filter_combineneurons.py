import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from itertools import chain
from statsmodels.stats.anova import AnovaRM


# from spikeUtilities import applylogisticfit,applyweibullfit
# use logistic cdf to fit psychometric fun
def applylogisticfit(x,y): 
    if y[0]>=y[-1]: #slope is negative
        bounds_low = (-np.inf,0,x.min(),y.min())
        bounds_hi = (0,np.inf,x.max(),y.max())
    else:#slope is positive
        bounds_low = (0,0,x.min(),y.min())
        bounds_hi = (np.inf,np.inf,x.max(),y.max())        

    def logisticfun(x, a, b, x0, y0):
        return y0 + b/(1+np.exp(-(x-x0)/a)) #a-slope b-range
    # Use curve fitting to find the best values for the parameters
    try:
        popt, _ = curve_fit(logisticfun, x, y,bounds=(bounds_low,bounds_hi),method='trf',maxfev=15000)
    except Exception as e:
        print('neurMetric curve_fit error:')
        print(e)
        popt = [np.nan,np.nan,np.nan,np.nan]
    # Generate the fitted curve using the optimized parameters
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = logisticfun(x_fit, popt[0], popt[1], popt[2], popt[3]) # apply the optimized parameters : a_opt b_opt
    y_fit_discrete = logisticfun(x, popt[0], popt[1],popt[2], popt[3])
    # get threshold at dprime=1
    if y.min()>1:
        thresh = y.min()
    elif y.max()<1:
        thresh = np.nan
    else:
        thresh = x_fit[np.argmin(np.abs(y_fit-1))]
    popt[1] = thresh
    return x_fit,y_fit,popt,y_fit_discrete

# use weibull cdf to fit psychometric fun
def applyweibullfit(x,y,threshx = 0.7):
    if y[0]>=y[-1]: #slope is negative
        bounds_low = (-np.inf,0,0,0)
        bounds_hi = (0,np.inf,np.inf,np.inf)
    else:#slope is positive
        bounds_low = (0,0,0,0)
        bounds_hi = (np.inf,np.inf,np.inf,np.inf)       
    def weibull_cdf(x, a, b,c,d):
        return c - d*np.exp(-(x/b)**a) #a-slope b-threshold
    # Use curve fitting to find the best values for the parameters
    try:
        # popt, _ = curve_fit(weibull_cdf, x+np.abs(x.min())+2, y,bounds=((-np.inf,-np.inf,-np.inf,-np.inf),(np.inf,np.inf,np.inf,np.inf)),maxfev=15000)
        popt, _ = curve_fit(weibull_cdf, x+np.abs(x.min())+2, y,bounds=(bounds_low,bounds_hi),method='trf',maxfev=30000)
    except Exception as e:
        print('neurMetric curve_fit error:')
        print(e)
        popt = [np.nan,np.nan,np.nan,np.nan]
    # Generate the fitted curve using the optimized parameters
    x_fit = np.linspace(x.min(), x.max(), 100)+np.abs(x.min())+2
    y_fit = weibull_cdf(x_fit, popt[0], popt[1],popt[2],popt[3]) # apply the optimized parameters : a_opt b_opt
    # print('a:'+str(popt[0])+'\nb:'+str(popt[1])+'\nc:'+str(popt[2])+'\nd:'+str(popt[3]))
    y_fit_discrete = weibull_cdf(x+np.abs(x.min())+2, popt[0], popt[1],popt[2], popt[3])
    # get threshold at threshx
    if y.min()>threshx:
        thresh = y.min()
    elif y.max()<threshx:
        thresh = np.nan
    else:
        thresh = x_fit[np.argmin(np.abs(y_fit-threshx))]-np.abs(x.min())-2
    popt[1] = thresh
    return x_fit-np.abs(x.min())-2,y_fit,popt,y_fit_discrete

def plotAnAVneuroMetric(data_temp,axessPosi,titlestr,Monkey,xy_populationfit,sesscls,keys,DVstr):
    # plot all sig neurometric fun 
    dprimMaxmin = 0.5
    slopeThres_df = pd.DataFrame()
    slope =[]
    thresh = []
    mod = []
    if data_temp['pvalslopeA']<0.05 and data_temp['y_raw_a'].max()>dprimMaxmin and data_temp['slopeA']>0:
        slope.append(data_temp['slopeA'])
        thresh.append(data_temp['threshA'])
        mod.append('a')         
        axessPosi.plot(data_temp['x_fit_a'], data_temp['y_fit_a'],linestyle='-',color='lightsteelblue',linewidth=1)
        xy_populationfit[keys]['a'][0] = np.concatenate((xy_populationfit[keys]['a'][0],data_temp['x_raw_a'].reshape((1,-1))),axis=0)
        xy_populationfit[keys]['a'][1] = np.concatenate((xy_populationfit[keys]['a'][1],data_temp['y_raw_a'].reshape((1,-1))),axis=0)
        xy_populationfit[keys]['a'][2] = np.concatenate((xy_populationfit[keys]['a'][2],data_temp['y_rawfr_a'].reshape((1,-1))),axis=0)
        xy_populationfit[keys]['a'][3].append([sesscls]*len(data_temp['x_raw_a']))
    else:
        slope.append(np.nan)
        thresh.append(np.nan)
        mod.append('a')
        xy_populationfit[keys]['a'][0] = np.concatenate((xy_populationfit[keys]['a'][0],data_temp['x_raw_a'].reshape((1,-1))),axis=0)
        xy_populationfit[keys]['a'][1] = np.concatenate((xy_populationfit[keys]['a'][1],np.full((1,len(data_temp['x_raw_a'])),np.nan)),axis=0)
        xy_populationfit[keys]['a'][2] = np.concatenate((xy_populationfit[keys]['a'][2],np.full((1,len(data_temp['x_raw_a'])),np.nan)),axis=0)
        xy_populationfit[keys]['a'][3].append([sesscls]*len(data_temp['x_raw_a']))  

    if data_temp['pvalslopeAV']<0.05 and data_temp['y_raw_av'].max()>dprimMaxmin and data_temp['slopeAV']>0:
        slope.append(data_temp['slopeAV'])
        thresh.append(data_temp['threshAV'])
        mod.append('av')     
        axessPosi.plot(data_temp['x_fit_av'], data_temp['y_fit_av'], linestyle='-',color='mistyrose',linewidth=1)
        xy_populationfit[keys]['av'][0] = np.concatenate((xy_populationfit[keys]['av'][0],data_temp['x_raw_av'].reshape((1,-1))),axis=0)
        xy_populationfit[keys]['av'][1] = np.concatenate((xy_populationfit[keys]['av'][1],data_temp['y_raw_av'].reshape((1,-1))),axis=0)
        xy_populationfit[keys]['av'][2] = np.concatenate((xy_populationfit[keys]['av'][2],data_temp['y_rawfr_av'].reshape((1,-1))),axis=0)
        xy_populationfit[keys]['av'][3].append([sesscls]*len(data_temp['x_raw_av']))
    else: 
        slope.append(np.nan)
        thresh.append(np.nan)
        mod.append('av') 
        xy_populationfit[keys]['av'][0] = np.concatenate((xy_populationfit[keys]['av'][0],data_temp['x_raw_av'].reshape((1,-1))),axis=0)
        xy_populationfit[keys]['av'][1] = np.concatenate((xy_populationfit[keys]['av'][1],np.full((1,len(data_temp['x_raw_av'])),np.nan)),axis=0)
        xy_populationfit[keys]['av'][2] = np.concatenate((xy_populationfit[keys]['av'][2],np.full((1,len(data_temp['x_raw_av'])),np.nan)),axis=0)
        xy_populationfit[keys]['av'][3].append([sesscls]*len(data_temp['x_raw_av'])) 

    axessPosi.legend([],frameon=False)    
    axessPosi.set_title(Monkey+titlestr) 
    # axessPosi.set_ylim([-0.1,1.1]) 
    axessPosi.set_xlabel('snr')
    # axessPosi.set_ylabel('fraction correct')
    axessPosi.set_ylabel(DVstr)

    slopeThres_df['slope'] = slope
    slopeThres_df['thresh'] = thresh
    slopeThres_df['mod'] = mod
    slopeThres_df['Monkey'] = Monkey
    slopeThres_df['cluster'] = sesscls
    return slopeThres_df,xy_populationfit  

def plotAnAVneuroMetric2(data_temp,axessPosi,titlestr,Monkey,xy_populationfit,sesscls,keys,DVstr):
    #  A and AV, if one of them have sig slope+ neurometric fit, plot the other regardless  
    dprimMaxmin = 0.5
    slopeThres_df = pd.DataFrame()
    slope =[]
    thresh = []
    mod = []
    if (data_temp['pvalslopeA']<0.05 and data_temp['y_raw_a'].max()>dprimMaxmin and data_temp['slopeA']>0) \
        or (data_temp['pvalslopeAV']<0.05 and data_temp['y_raw_av'].max()>dprimMaxmin and data_temp['slopeAV']>0):
        # a sig info for this neuron
        slope.append(data_temp['slopeA'])
        thresh.append(data_temp['threshA'])
        mod.append('a') 
        # av sig info for this neuron
        slope.append(data_temp['slopeAV'])
        thresh.append(data_temp['threshAV'])
        mod.append('av')        
        axessPosi.plot(data_temp['x_fit_a'], data_temp['y_fit_a'],linestyle='-',color='lightsteelblue',linewidth=1)
        xy_populationfit[keys]['a'][0] = np.concatenate((xy_populationfit[keys]['a'][0],data_temp['x_raw_a'].reshape((1,-1))),axis=0)
        xy_populationfit[keys]['a'][1] = np.concatenate((xy_populationfit[keys]['a'][1],data_temp['y_raw_a'].reshape((1,-1))),axis=0)
        xy_populationfit[keys]['a'][2] = np.concatenate((xy_populationfit[keys]['a'][2],data_temp['y_rawfr_a'].reshape((1,-1))),axis=0) 
        xy_populationfit[keys]['a'][3].append([sesscls]*len(data_temp['x_raw_a']))

        axessPosi.plot(data_temp['x_fit_av'], data_temp['y_fit_av'], linestyle='-',color='mistyrose',linewidth=1)
        xy_populationfit[keys]['av'][0] = np.concatenate((xy_populationfit[keys]['av'][0],data_temp['x_raw_av'].reshape((1,-1))),axis=0)
        xy_populationfit[keys]['av'][1] = np.concatenate((xy_populationfit[keys]['av'][1],data_temp['y_raw_av'].reshape((1,-1))),axis=0)
        xy_populationfit[keys]['av'][2] = np.concatenate((xy_populationfit[keys]['av'][2],data_temp['y_rawfr_av'].reshape((1,-1))),axis=0) 
        xy_populationfit[keys]['a'][3].append([sesscls]*len(data_temp['x_raw_av']))

    axessPosi.legend([],frameon=False)    
    axessPosi.set_title(Monkey+'_'+titlestr) 
    # axessPosi.set_ylim([-0.1,1.1]) 
    axessPosi.set_xlabel('snr')
    # axessPosi.set_ylabel('fraction correct')
    axessPosi.set_ylabel(DVstr)

    slopeThres_df['slope'] = slope
    slopeThres_df['thresh'] = thresh
    slopeThres_df['mod'] = mod
    slopeThres_df['Monkey'] = Monkey
    slopeThres_df['cluster'] = sesscls
    return slopeThres_df,xy_populationfit  

def plotAnAVneuroMetric3(data_temp,axessPosi,titlestr,Monkey,xy_populationfit,sesscls,keys,DVstr):
    # only plot if both A and AV condition have sig slope+   
    dprimMaxmin = 0.5
    slopeThres_df = pd.DataFrame()
    slope =[]
    thresh = []
    mod = []
    if (data_temp['pvalslopeA']<0.05 and data_temp['y_raw_a'].max()>dprimMaxmin and data_temp['slopeA']>0) \
        and (data_temp['pvalslopeAV']<0.05 and data_temp['y_raw_av'].max()>dprimMaxmin and data_temp['slopeAV']>0):
        # a sig info for this neuron
        slope.append(data_temp['slopeA'])
        thresh.append(data_temp['threshA'])
        mod.append('a') 
        # av sig info for this neuron
        slope.append(data_temp['slopeAV'])
        thresh.append(data_temp['threshAV'])
        mod.append('av')

        axessPosi.plot(data_temp['x_fit_a'], data_temp['y_fit_a'],linestyle='-',color='lightsteelblue',linewidth=1)
        xy_populationfit[keys]['a'][0] = np.concatenate((xy_populationfit[keys]['a'][0],data_temp['x_raw_a'].reshape((1,-1))),axis=0)
        xy_populationfit[keys]['a'][1] = np.concatenate((xy_populationfit[keys]['a'][1],data_temp['y_raw_a'].reshape((1,-1))),axis=0)
        xy_populationfit[keys]['a'][2] = np.concatenate((xy_populationfit[keys]['a'][2],data_temp['y_rawfr_a'].reshape((1,-1))),axis=0)
        xy_populationfit[keys]['a'][3].append([sesscls]*len(data_temp['x_raw_a']))

        axessPosi.plot(data_temp['x_fit_av'], data_temp['y_fit_av'], linestyle='-',color='mistyrose',linewidth=1)
        xy_populationfit[keys]['av'][0] = np.concatenate((xy_populationfit[keys]['av'][0],data_temp['x_raw_av'].reshape((1,-1))),axis=0)
        xy_populationfit[keys]['av'][1] = np.concatenate((xy_populationfit[keys]['av'][1],data_temp['y_raw_av'].reshape((1,-1))),axis=0)
        xy_populationfit[keys]['av'][2] = np.concatenate((xy_populationfit[keys]['av'][2],data_temp['y_rawfr_av'].reshape((1,-1))),axis=0)
        xy_populationfit[keys]['av'][3].append([sesscls]*len(data_temp['x_raw_av']))

    axessPosi.legend([],frameon=False)    
    axessPosi.set_title(Monkey+'_'+titlestr) 
    # axessPosi.set_ylim([-0.1,1.1]) 
    axessPosi.set_xlabel('snr')
    # axessPosi.set_ylabel('fraction correct')
    axessPosi.set_ylabel(DVstr)

    slopeThres_df['slope'] = slope
    slopeThres_df['thresh'] = thresh
    slopeThres_df['mod'] = mod
    slopeThres_df['Monkey'] = Monkey
    slopeThres_df['cluster'] = sesscls
    return slopeThres_df,xy_populationfit  

def plotAnAVneuroMetric4(data_temp,axessPosi,Monkey,xy_populationfit,sesscls,keys,DVstr,rawfrstr):
    #  plot A and AV regardless of the fit  
    slopeThres_df = pd.DataFrame()
    slope =[]
    thresh = []
    mod = []

    # a sig info for this neuron
    slope.append(data_temp['slopeA'])
    thresh.append(data_temp['threshA'])
    mod.append('a') 
    # av sig info for this neuron
    slope.append(data_temp['slopeAV'])
    thresh.append(data_temp['threshAV'])
    mod.append('av')        
    axessPosi.plot(data_temp['x_fit_a'], data_temp['y_fit_a'],linestyle='-',color='lightsteelblue',linewidth=1)
    xy_populationfit[keys]['a'][0] = np.concatenate((xy_populationfit[keys]['a'][0],data_temp['x_raw_a'].reshape((1,-1))),axis=0)
    xy_populationfit[keys]['a'][1] = np.concatenate((xy_populationfit[keys]['a'][1],data_temp['y_raw_a'].reshape((1,-1))),axis=0)
    xy_populationfit[keys]['a'][2] = np.concatenate((xy_populationfit[keys]['a'][2],data_temp[rawfrstr+'_a'].reshape((1,-1))),axis=0) 
    xy_populationfit[keys]['a'][3].append([sesscls]*len(data_temp['x_raw_a']))

    axessPosi.plot(data_temp['x_fit_av'], data_temp['y_fit_av'], linestyle='-',color='mistyrose',linewidth=1)
    xy_populationfit[keys]['av'][0] = np.concatenate((xy_populationfit[keys]['av'][0],data_temp['x_raw_av'].reshape((1,-1))),axis=0)
    xy_populationfit[keys]['av'][1] = np.concatenate((xy_populationfit[keys]['av'][1],data_temp['y_raw_av'].reshape((1,-1))),axis=0)
    xy_populationfit[keys]['av'][2] = np.concatenate((xy_populationfit[keys]['av'][2],data_temp[rawfrstr+'_av'].reshape((1,-1))),axis=0) 
    xy_populationfit[keys]['a'][3].append([sesscls]*len(data_temp['x_raw_av']))

    axessPosi.legend([],frameon=False)    
    axessPosi.set_title(Monkey,fontsize=fontsizeNo) 
    # axessPosi.set_ylim([-0.1,1.1]) 
    axessPosi.set_xlabel('SNR',fontsize=fontsizeNo)
    # axessPosi.set_ylabel('fraction correct')
    axessPosi.set_ylabel(DVstr,fontsize=fontsizeNo)
    axessPosi.tick_params(axis='both', which='major', labelsize=fontsizeNo-2)

    slopeThres_df['slope'] = slope
    slopeThres_df['thresh'] = thresh
    slopeThres_df['mod'] = mod
    slopeThres_df['Monkey'] = Monkey
    slopeThres_df['cluster'] = sesscls
    return slopeThres_df,xy_populationfit  


def fitPlotPopulation(xy_populationfit,keystr,Monkey,axessfitmean,axessRaw_scatter,axessRawfr_scatter,axessRaw,rawfrstr):
        # x_fit_a,y_fit_a,_,_ = applylogisticfit(np.mean(xy_populationfit[keystr]['a'][0],axis=0),np.mean(xy_populationfit[keystr]['a'][1],axis=0))
        x_fit_a,y_fit_a,pop_a,_ = applyweibullfit(np.nanmean(xy_populationfit[keystr]['a'][0],axis=0),np.nanmean(xy_populationfit[keystr]['a'][1],axis=0),0.5)
        axessfitmean.plot(x_fit_a,y_fit_a,'b',label='A')
        # x_fit_av,y_fit_av,_,_ = applylogisticfit (np.mean(xy_populationfit['av+snr/snr']['av'][0],axis=0),np.mean(xy_populationfit[keystr]['av'][1],axis=0))
        x_fit_av,y_fit_av,pop_av,_ = applyweibullfit(np.nanmean(xy_populationfit[keystr]['av'][0],axis=0),np.nanmean(xy_populationfit[keystr]['av'][1],axis=0),0.5)
        axessfitmean.plot(x_fit_av,y_fit_av,'r',label='AV') 
        axessfitmean.legend(frameon=False,loc='lower right')
        axessfitmean.set_ylim([0,1.1])
        raw_df_temp = pd.DataFrame({'snr':xy_populationfit[keystr]['a'][0].reshape(1,-1).tolist()[0]+xy_populationfit[keystr]['av'][0].reshape(1,-1).tolist()[0],
                                    DVstr:xy_populationfit[keystr]['a'][1].reshape(1,-1).tolist()[0]+xy_populationfit[keystr]['av'][1].reshape(1,-1).tolist()[0],
                                    'fr':xy_populationfit[keystr]['a'][2].reshape(1,-1).tolist()[0]+xy_populationfit[keystr]['av'][2].reshape(1,-1).tolist()[0],
                                    'mod':['a']*len(list(xy_populationfit[keystr]['a'][0].reshape(1,-1).tolist()[0]))+['av']*len(list(xy_populationfit[keystr]['av'][0].reshape(1,-1).tolist()[0])),
                                    'cls':list(chain.from_iterable(xy_populationfit[keystr]['a'][3]))+list(chain.from_iterable(xy_populationfit[keystr]['av'][3]))})
        sns.violinplot(raw_df_temp,x='snr',y=DVstr,hue='mod',hue_order=['av','a'],ax=axessRaw,palette=['salmon','cornflowerblue'],linewidth=0.5)
        sns.swarmplot(raw_df_temp,x='snr',y=DVstr,hue='mod',hue_order=['av','a'],ax=axessRaw,dodge=True,size=2.5,alpha=0.5)
        axessRaw.set_title(Monkey+'_'+keystr)
        axessRaw.set_ylim([0,1.1])
        # print(keystr+' cls info')
        # print(raw_df_temp[raw_df_temp['snr']==0].to_string())
        snrorder=['easy','medium','difficult']
        colors = ['mistyrose','lightcoral','maroon']   
        print(raw_df_temp.to_string())     
        raw_df_temp['snr']=raw_df_temp['snr'].replace({-15:'difficult',-10:'difficult',-5:'medium',0:'medium',5:'easy',10:'easy'})
        raw_df = raw_df_temp.groupby(by=['cls','mod','snr'])[['fr','FractionCorrect']].mean().reset_index()
        print(raw_df.to_string())    

        mapping = {string: i for i, string in enumerate(snrorder)}
        raw_df['sort_key'] = raw_df['snr'].map(mapping)
        raw_df.sort_values(by='sort_key',inplace=True)
        raw_df.drop(columns=['sort_key'],inplace=True)
        

        plotfitcorr(raw_df,axessRaw_scatter,'snr','FractionCorrect', 'mod', 'a', 'av', ['mistyrose','lightcoral','maroon'], 'o', ['cls'])
        axessRaw_scatter.set_title(Monkey+'_'+'FractionCorrect in neurometric fit')
        # twoway rm statistic tests
        try:
            statsRes = stats.wilcoxon(raw_df[raw_df['mod']=='a'].sort_values(['snr','cls'],kind='mergesort')['FractionCorrect'].values,
                raw_df[raw_df['mod']=='av'].sort_values(['snr','cls'],kind='mergesort')['FractionCorrect'].values,nan_policy='omit')
            axessRaw_scatter.text(0.8,0.2,' wilcoxonp='+str("{:.1e}".format(statsRes[1])),horizontalalignment='right',verticalalignment='center',fontsize=10,transform=axessRaw_scatter.transAxes)
        except:
            pass

        plotfitcorr(raw_df,axessRawfr_scatter,'snr','fr', 'mod', 'a', 'av', ['mistyrose','lightcoral','maroon'], 'o', ['cls'])
        axessRawfr_scatter.set_title(Monkey+'_'+'FiringRate(zscored)')
        try:
            statsRes = stats.wilcoxon(raw_df[raw_df['mod']=='a'].sort_values(['snr','cls'],kind='mergesort')['fr'].values,
                raw_df[raw_df['mod']=='av'].sort_values(['snr','cls'],kind='mergesort')['fr'].values,nan_policy='omit')
            axessRawfr_scatter.text(0.8,0.2,' wilcoxonp='+str("{:.1e}".format(statsRes[1])),horizontalalignment='right',verticalalignment='center',fontsize=10,transform=axessRaw_scatter.transAxes)
        except:
            pass      
        return raw_df

def plotfitcorr(slopeThres_df1,axess4,catstr, varStr, compCol, xvar, yvar, colorlist, markerSymb, sortlist):
    for modtest,colr in zip(slopeThres_df1[catstr].unique(),colorlist):
        slopeThres_temp = slopeThres_df1[slopeThres_df1[catstr]==modtest]
        axess4.scatter(slopeThres_temp[slopeThres_temp[compCol]==xvar].sort_values(sortlist,kind='mergesort')[varStr].values,
                                    slopeThres_temp[slopeThres_temp[compCol]==yvar].sort_values(sortlist,kind='mergesort')[varStr].values,
                                    alpha=0.5,c=colr,label=modtest,marker=markerSymb,edgecolors='none')   
    axisLoBound = min([axess4.get_xlim()[0],axess4.get_ylim()[0]])
    axisHiBound = max([axess4.get_xlim()[1],axess4.get_ylim()[1]])
    axess4.set_xlim([axisLoBound,axisHiBound])
    axess4.set_ylim([axisLoBound,axisHiBound])
    axess4.plot(np.linspace(axisLoBound,axisHiBound,100),np.linspace(axisLoBound,axisHiBound,100),'--',color='gray',linewidth=1.5)
    axess4.set_title(varStr,fontsize=fontsizeNo)
    if xvar=='a':
        axess4.set_xlabel('Auditory',fontsize=fontsizeNo)
        axess4.set_ylabel('Audiovisual',fontsize=fontsizeNo)
    axess4.legend(frameon=False,loc='upper left',fontsize=fontsizeNo)
    axess4.tick_params(axis='both', which='major', labelsize=fontsizeNo-2)


# MonkeyDate_all = {'Elay':['230420']} #

# ResPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/neuralMetric/'
# figSavePath = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Figures/neuroMetricFit/'
# AVmodPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/AVmodIndex/'
# behaveResPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/BehavPerform/'
# glmfitPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/glmfit/'

ResPathway = '/data/by-user/Huaizhen/Fitresults/neuralMetric/'
figSavePath = '/data/by-user/Huaizhen/Figures/neuroMetricFit/'
AVmodPathway = '/data/by-user/Huaizhen/Fitresults/AVmodIndex/'
behaveResPathway = '/data/by-user/Huaizhen/Fitresults/BehavPerform/'
glmfitPathway = '/data/by-user/Huaizhen/Fitresults/glmfit/'

MonkeyDate_all = {'Elay':['230420','230509','230531',
                        '230602','230606','230613','230616','230620','230627',
                        '230705','230711','230717','230718','230719','230726','230728',
                        '230802','230808','230810','230814','230818','230822','230829',
                        '230906','230908','230915','230919','230922','230927',
                         '231003','231004','231010'], 
                  'Wu':['230809','230815','230821','230830',
                        '230905','230911','230913','230918','230925',
                          '231002','231006','231009','231018','231020',
                          '231204','231211','231213','231225','231227','231229']}

namestr = '_hit_spkRateCRa2'
DVstr = 'FractionCorrect' # 'FractionCorrect','dprime'
rawfrstr = 'y_rawfr' #'y_rawfr' 'y_rawfr2'
fontsizeNo = 14

# filter neurons based on different rules
# df_avMod_all = pickle.load(open(AVmodPathway+'AVmodTTestDF_baselinecorrected.pkl','rb'))
df_avMod_all = pickle.load(open(glmfitPathway+'glmfitCoefDF_cooOnsetIndwithVirtual_2mod2lab.pkl','rb'))
df_avMod_all = df_avMod_all[(df_avMod_all['GLMmod']=="['snr-shift', 'V']") & (df_avMod_all['time']==0)]
df_avMod_all_sig = df_avMod_all[df_avMod_all['pval']<0.05] 
clscolstr = 'session_cls'

# filter out drifted neurons
dfinspect = pd.read_excel('AllClusters4inspectionSheet.xlsx')
driftedUnites = list(dfinspect[dfinspect['driftYES/NO/MAYBE(1,0,2)']==1]['session_cls'].values)
df_avMod_all_sig = df_avMod_all_sig[~df_avMod_all_sig['session_cls'].isin(driftedUnites)]

# plot neurometric fit according to neuron labels
fig, axess = plt.subplots(2,1,figsize=(6,6),sharex='col')  
fig2, axess2 = plt.subplots(2,1,figsize=(6,8),sharex='col') 
fig3, axess3 = plt.subplots(2,1,figsize=(6,8),sharex='col') 
fig4, axess4 = plt.subplots(2,1,figsize=(6,8),sharex='col') 

filterslopestr = 'glmfit'#'AnAV' #'AoAV' 'AuAV', 'glmfit'
figformat = 'png'

slopeThres_df1 = pd.DataFrame()
raw_df = {}
for mm,(Monkey,Date) in enumerate(MonkeyDate_all.items()):
    xy_populationfit = {filterslopestr:{'a':[np.empty((0,6)),np.empty((0,6)),np.empty((0,6)),[]],'av':[np.empty((0,6)),np.empty((0,6)),np.empty((0,6)),[]]}}
    
    # # filter based on sig nm slope 
    # for session in Date:  
    #     print(Monkey+session+' in plotting....')   
    #     #plot neurometric fun 
    #     data = pickle.load(open(ResPathway+Monkey+'_'+session+'_neuroMetric'+namestr+'.pkl','rb'))
    #     allcls =  [key for key in data if key not in ['Monkey','session']]

    #     for cc,cls in enumerate(allcls):
    #         data_temp = data[cls].copy()
    #         sesscls = session+'_'+cls
    #         if filterslopestr == 'AnAV': # sig NM slope in both A and AV conditions
    #             slopeThres_df_temp,xy_populationfit = plotAnAVneuroMetric3(data_temp,axess[mm],filterslopestr,Monkey,xy_populationfit,sesscls,filterslopestr,DVstr)
    #         if filterslopestr == 'AoAV': # all sig NM slope  
    #             slopeThres_df_temp,xy_populationfit = plotAnAVneuroMetric(data_temp,axess[mm],filterslopestr,Monkey,xy_populationfit,sesscls,filterslopestr,DVstr)
    #         if filterslopestr == 'AuAV': # save both A and AV NM info if slope in >=1 condition is sig
    #             slopeThres_df_temp,xy_populationfit = plotAnAVneuroMetric2(data_temp,axess[mm],filterslopestr,Monkey,xy_populationfit,sesscls,filterslopestr,DVstr)
    #         slopeThres_df_temp['modTest'] = filterslopestr
    #         slopeThres_df1 = pd.concat((slopeThres_df1,slopeThres_df_temp))
                               
    # filter based on glmfit 
    for session in Date:  
        print(Monkey+session+' in plotting....')   
        #plot neurometric fun 
        data = pickle.load(open(ResPathway+Monkey+'_'+session+'_neuroMetricFit'+namestr+'.pkl','rb'))
       
        df_avMod_sess = df_avMod_all_sig[(df_avMod_all_sig['Monkey']==Monkey) & (df_avMod_all_sig[clscolstr].str.contains(session))]   
        for cc,sesscls in enumerate(df_avMod_sess[clscolstr].unique().tolist()): 
            cls = sesscls[7:]
            data_temp = data[cls].copy()
            if data_temp['fr>1']=='yes':                             
                df_avMod_sess_temp = df_avMod_sess[df_avMod_sess[clscolstr]==sesscls] 
                mod = str(sorted(df_avMod_sess_temp['iv'].values))                
                slopeThres_df_temp,xy_populationfit = plotAnAVneuroMetric4(data_temp,axess[mm],Monkey,xy_populationfit,sesscls,filterslopestr,DVstr,rawfrstr)
                slopeThres_df_temp['modTest'] = filterslopestr
                slopeThres_df_temp['clsMod'] = mod
                slopeThres_df1 = pd.concat((slopeThres_df1,slopeThres_df_temp))

    # fit population neurometric
    raw_df1 = fitPlotPopulation(xy_populationfit,filterslopestr,Monkey,axess[mm],axess2[mm],axess3[mm],axess4[mm],rawfrstr)
    raw_df1['modTest'] = filterslopestr
    raw_df[Monkey] = raw_df1.copy()

fig.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
fig.savefig(figSavePath+'neuroMetric_'+filterslopestr+'fit'+namestr+'.'+figformat,format = figformat)
fig2.savefig(figSavePath+'neuroMetric_'+filterslopestr+'_FC_scatter'+namestr+'.'+figformat,format = figformat)
fig3.savefig(figSavePath+'neuroMetric_'+filterslopestr+'_RawFR_scatter'+namestr+'.'+figformat,format = figformat)
fig4.savefig(figSavePath+'neuroMetric_'+filterslopestr+'_FC_violin'+namestr+'.'+figformat,format = figformat)
plt.close(fig)
plt.close(fig2)
plt.close(fig3)
plt.close(fig4)

# # save info of units with sig slope in both A and AV condition
# slopeThres_df1.to_excel(AVmodPathway+'sigSlopeNeurons_'+filterslopestr+'_sig'+namestr+'.xlsx',index=False)
# pickle.dump(slopeThres_df1,open(AVmodPathway+'sigSlopeNeurons_'+filterslopestr+'_sig'+namestr+'.pkl','wb'))  

# # plot slope and threshold info   
# # all neuron has sig positive slope in both A and AV condition, colorlist ['royalblue','mediumorchid','lightcoral']
# print('plot slope and threshold info in A vs AV')
# fig4, axess4 = plt.subplots(1,2,figsize=(10,5)) 
# plotfitcorr(slopeThres_df1,axess4[0],'modTest','slope','mod','a','av',['royalblue'],'o',['cluster','Monkey'])
# statsRes = stats.wilcoxon(slopeThres_df1[slopeThres_df1['mod']=='a'].sort_values(['modTest','cluster','Monkey'],kind='mergesort')['slope'].values,
#                 slopeThres_df1[slopeThres_df1['mod']=='av'].sort_values(['modTest','cluster','Monkey'],kind='mergesort')['slope'].values,nan_policy='omit')
# axess4[0].text(0.8,0.5,' p='+str("{:.1e}".format(statsRes[1])),horizontalalignment='right',verticalalignment='center',fontsize=7,transform=axess4[0].transAxes)

# plotfitcorr(slopeThres_df1,axess4[1],'modTest','thresh','mod','a','av',['royalblue'],'o',['cluster','Monkey'])
# statsRes = stats.wilcoxon(slopeThres_df1[slopeThres_df1['mod']=='a'].sort_values(['modTest','cluster','Monkey'],kind='mergesort')['thresh'].values,
#                 slopeThres_df1[slopeThres_df1['mod']=='av'].sort_values(['modTest','cluster','Monkey'],kind='mergesort')['thresh'].values,nan_policy='omit')
# axess4[1].text(0.8,0.5,' p='+str("{:.1e}".format(statsRes[1])),horizontalalignment='right',verticalalignment='center',fontsize=7,transform=axess4[1].transAxes)

# fig4.tight_layout()
# fig4.savefig(figSavePath+'neuroMetric_'+filterslopestr+'_SlopeCorrNthreshCorr'+namestr+'.'+figformat,format = figformat)
# plt.close(fig4)   


# ## plot session by session, single neuron and population NM threshold/slope correlation to those of behave 
# print('plot session by session, single neuron and population NM threshold/slope correlation to behave....')
# [BehavefitParam,BehaveRTinfo] = pickle.load(open(behaveResPathway+'BehavPerformfit_2Monk.pkl','rb'))
# behavNM_df = pd.DataFrame()
# for sess_temp in (BehavefitParam.sess.unique()): 
#     print(sess_temp)
#     # print('Warning: assume that ONE sesison day only have data from ONE monkey!')
#     # behave info
#     monkey = BehavefitParam[BehavefitParam['sess']==sess_temp]['Monkey'].unique()[0]
#     BHAVslopeA_temp = BehavefitParam[(BehavefitParam['sess']==sess_temp)&(BehavefitParam['mod']=='A')]['slope'].unique()[0]
#     BHAVthreshA_temp = BehavefitParam[(BehavefitParam['sess']==sess_temp)&(BehavefitParam['mod']=='A')]['thresh'].unique()[0]
#     BHAVslopeAV_temp = BehavefitParam[(BehavefitParam['sess']==sess_temp)&(BehavefitParam['mod']=='AV')]['slope'].unique()[0]
#     BHAVthreshAV_temp = BehavefitParam[(BehavefitParam['sess']==sess_temp)&(BehavefitParam['mod']=='AV')]['thresh'].unique()[0]
#     #NM info
#     raw_df_temp = raw_df[monkey]
#     if raw_df_temp.shape[0]>0:
#         rawNM_df_temp = raw_df_temp[raw_df_temp['cls'].str.contains(sess_temp)]
#         if rawNM_df_temp.shape[0]>0:
#             rawNM_df_temp_group = rawNM_df_temp.groupby(['snr','mod','modTest'])['FractionCorrect'].mean().reset_index()
#             for modtest_temp in ['A+V','A','V']:
#                 rawNM_df_temp_group_temp = rawNM_df_temp_group[rawNM_df_temp_group['modTest']==modtest_temp]
#                 # fit for 3 neuron group separatly in a session
#                 if rawNM_df_temp_group_temp.shape[0]==0:
#                     behavNM_df = pd.concat((behavNM_df,pd.DataFrame({'Monkey':[monkey]*4,'sess':[sess_temp]*4,
#                                                         'modTest':[modtest_temp]*4,
#                                                         'mod':['a','av']*2,
#                                                         'fitDfrom':['NM','NM','BHAV','BHAV'],
#                                                         'slope':[np.nan,np.nan,np.nan,np.nan],
#                                                         'thresh':[np.nan,np.nan,np.nan,np.nan]})))
#                 else: 
#                     a_df = rawNM_df_temp_group_temp[rawNM_df_temp_group_temp['mod']=='a']
#                     x_fit_a,y_fit_a,pop_a,_ = applyweibullfit(a_df['snr'].values,a_df['FractionCorrect'].values,0.5)
#                     av_df = rawNM_df_temp_group_temp[rawNM_df_temp_group_temp['mod']=='av']
#                     x_fit_av,y_fit_av,pop_av,_ = applyweibullfit(av_df['snr'].values,av_df['FractionCorrect'].values,0.5)
#                     behavNM_df = pd.concat((behavNM_df,pd.DataFrame({'Monkey':[monkey]*4,'sess':[sess_temp]*4,
#                                                         'modTest':[modtest_temp]*4,
#                                                         'mod':['a','av']*2,
#                                                         'fitDfrom':['NM','NM','BHAV','BHAV'],
#                                                         'slope':[pop_a[0],pop_av[0],BHAVslopeA_temp,BHAVslopeAV_temp],
#                                                         'thresh':[pop_a[1],pop_av[1],BHAVthreshA_temp,BHAVthreshAV_temp]})))
#             # fit for all neuron in a session
#             rawNM_df_temp_group2 = rawNM_df_temp.groupby(['snr','mod'])['FractionCorrect'].mean().reset_index()
#             a_df = rawNM_df_temp_group2[rawNM_df_temp_group2['mod']=='a']
#             x_fit_a,y_fit_a,pop_a,_ = applyweibullfit(a_df['snr'].values,a_df['FractionCorrect'].values,0.5)
#             av_df = rawNM_df_temp_group2[rawNM_df_temp_group2['mod']=='av']
#             x_fit_av,y_fit_av,pop_av,_ = applyweibullfit(av_df['snr'].values,av_df['FractionCorrect'].values,0.5)
#             behavNM_df = pd.concat((behavNM_df,pd.DataFrame({'Monkey':[monkey]*4,'sess':[sess_temp]*4,
#                                                 'modTest':['ALL']*4,
#                                                 'mod':['a','av']*2,
#                                                 'fitDfrom':['NM','NM','BHAV','BHAV'],
#                                                 'slope':[pop_a[0],pop_av[0],BHAVslopeA_temp,BHAVslopeAV_temp],
#                                                 'thresh':[pop_a[1],pop_av[1],BHAVthreshA_temp,BHAVthreshAV_temp]})))

# fig4, axess4 = plt.subplots(1,2,figsize=(10,5)) 
# behavNM_df_temp =behavNM_df[behavNM_df['modTest'].isin(['ALL'])]
# plotfitcorr(behavNM_df_temp,axess4[0],'mod','slope','fitDfrom','NM','BHAV',['royalblue','lightcoral'],'o',['Monkey','sess','mod'])
# statsRes = stats.wilcoxon(behavNM_df_temp[behavNM_df_temp['fitDfrom']=='NM'].sort_values(['modTest','mod','sess','Monkey'],kind='mergesort')['slope'].values,
#                 behavNM_df_temp[behavNM_df_temp['fitDfrom']=='BHAV'].sort_values(['modTest','mod','sess','Monkey'],kind='mergesort')['slope'].values,nan_policy='omit')
# axess4[0].text(0.8,0.5,' p='+str("{:.1e}".format(statsRes[1])),horizontalalignment='right',verticalalignment='center',fontsize=7,transform=axess4[0].transAxes)

# plotfitcorr(behavNM_df_temp,axess4[1],'mod','thresh','fitDfrom','NM','BHAV',['royalblue','lightcoral'],'o',['Monkey','sess','mod'])
# statsRes = stats.wilcoxon(behavNM_df_temp[behavNM_df_temp['fitDfrom']=='NM'].sort_values(['modTest','mod','sess','Monkey'],kind='mergesort')['thresh'].values,
#                 behavNM_df_temp[behavNM_df_temp['fitDfrom']=='BHAV'].sort_values(['modTest','mod','sess','Monkey'],kind='mergesort')['thresh'].values,nan_policy='omit')
# axess4[1].text(0.8,0.5,' p='+str("{:.1e}".format(statsRes[1])),horizontalalignment='right',verticalalignment='center',fontsize=7,transform=axess4[1].transAxes)

# fig4.tight_layout()
# fig4.savefig(figSavePath+'NMvsBEHAV_'+filterslopestr+'_CorrNthreshCorr_allNeuron'+'.'+figformat,format = figformat)
# plt.close(fig4) 

# fig4, axess4 = plt.subplots(1,2,figsize=(10,5)) 
# behavNM_df_temp = behavNM_df[behavNM_df['modTest'].isin(['A','V','A+V'])].dropna()
# plotfitcorr(behavNM_df_temp[behavNM_df_temp['mod']=='a'],axess4[0],'modTest','slope','fitDfrom','NM','BHAV',['royalblue','mediumorchid','lightcoral'],'o',['Monkey','sess','mod'])
# plotfitcorr(behavNM_df_temp[behavNM_df_temp['mod']=='av'],axess4[0],'modTest','slope','fitDfrom','NM','BHAV',['royalblue','mediumorchid','lightcoral'],'v',['Monkey','sess','mod'])
# statsRes = stats.wilcoxon(behavNM_df_temp[behavNM_df_temp['fitDfrom']=='NM'].sort_values(['modTest','mod','sess','Monkey'],kind='mergesort')['slope'].values,
#                 behavNM_df_temp[behavNM_df_temp['fitDfrom']=='BHAV'].sort_values(['modTest','mod','sess','Monkey'],kind='mergesort')['slope'].values,nan_policy='omit')
# axess4[0].text(0.8,0.5,' p='+str("{:.1e}".format(statsRes[1])),horizontalalignment='right',verticalalignment='center',fontsize=7,transform=axess4[0].transAxes)

# plotfitcorr(behavNM_df_temp[behavNM_df_temp['mod']=='a'],axess4[1],'modTest','thresh','fitDfrom','NM','BHAV',['royalblue','mediumorchid','lightcoral'],'o',['Monkey','sess','mod'])
# plotfitcorr(behavNM_df_temp[behavNM_df_temp['mod']=='av'],axess4[1],'modTest','thresh','fitDfrom','NM','BHAV',['royalblue','mediumorchid','lightcoral'],'v',['Monkey','sess','mod'])
# statsRes = stats.wilcoxon(behavNM_df_temp[behavNM_df_temp['fitDfrom']=='NM'].sort_values(['modTest','mod','sess','Monkey'],kind='mergesort')['thresh'].values,
#                 behavNM_df_temp[behavNM_df_temp['fitDfrom']=='BHAV'].sort_values(['modTest','mod','sess','Monkey'],kind='mergesort')['thresh'].values,nan_policy='omit')
# axess4[1].text(0.8,0.5,' p='+str("{:.1e}".format(statsRes[1])),horizontalalignment='right',verticalalignment='center',fontsize=7,transform=axess4[1].transAxes)

# fig4.tight_layout()
# fig4.savefig(figSavePath+'NMvsBEHAV_'+filterslopestr+'_CorrNthreshCorr_singleNeuron'+'.'+figformat,format = figformat)
# plt.close(fig4) 


print('end')


