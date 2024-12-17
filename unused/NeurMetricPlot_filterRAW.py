import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from itertools import chain
from statsmodels.stats.anova import AnovaRM

def catSessCls(xy_populationfit,data_temp,sesscls,neurLab):
    xy_populationfit[neurLab]['a'][0] = np.concatenate((xy_populationfit[neurLab]['a'][0],data_temp['x_raw_a'].reshape((1,-1))),axis=0)
    xy_populationfit[neurLab]['a'][1] = np.concatenate((xy_populationfit[neurLab]['a'][1],data_temp['y_raw_a'].reshape((1,-1))),axis=0)
    xy_populationfit[neurLab]['a'][2] = np.concatenate((xy_populationfit[neurLab]['a'][2],data_temp['y_rawfr_a'].reshape((1,-1))),axis=0)
    xy_populationfit[neurLab]['a'][3].append([sesscls]*len(data_temp['x_raw_a']))

    xy_populationfit[neurLab]['av'][0] = np.concatenate((xy_populationfit[neurLab]['av'][0],data_temp['x_raw_av'].reshape((1,-1))),axis=0)
    xy_populationfit[neurLab]['av'][1] = np.concatenate((xy_populationfit[neurLab]['av'][1],data_temp['y_raw_av'].reshape((1,-1))),axis=0)
    xy_populationfit[neurLab]['av'][2] = np.concatenate((xy_populationfit[neurLab]['av'][2],data_temp['y_rawfr_av'].reshape((1,-1))),axis=0)
    xy_populationfit[neurLab]['av'][3].append([sesscls]*len(data_temp['x_raw_av']))
    return xy_populationfit

def PlotPopulation(xy_populationfit,keystr,axessRawNM_scatter,axessRawNM,axessRawFR_scatter):       
    raw_df = pd.DataFrame({'snr':xy_populationfit[keystr]['a'][0].reshape(1,-1).tolist()[0]+xy_populationfit[keystr]['av'][0].reshape(1,-1).tolist()[0],
                                DVstr:xy_populationfit[keystr]['a'][1].reshape(1,-1).tolist()[0]+xy_populationfit[keystr]['av'][1].reshape(1,-1).tolist()[0],
                                'mod':['a']*len(list(xy_populationfit[keystr]['a'][0].reshape(1,-1).tolist()[0]))+['av']*len(list(xy_populationfit[keystr]['av'][0].reshape(1,-1).tolist()[0])),
                                'FR':xy_populationfit[keystr]['a'][2].reshape(1,-1).tolist()[0]+xy_populationfit[keystr]['av'][2].reshape(1,-1).tolist()[0],
                                'cls':list(chain.from_iterable(xy_populationfit[keystr]['a'][3]))+list(chain.from_iterable(xy_populationfit[keystr]['av'][3]))})
    sns.violinplot(raw_df,x='snr',y=DVstr,hue='mod',hue_order=['av','a'],ax=axessRawNM,palette=['salmon','cornflowerblue'],linewidth=0.5)
    sns.swarmplot(raw_df,x='snr',y=DVstr,hue='mod',hue_order=['av','a'],ax=axessRawNM,dodge=True,size=2.5,alpha=0.5)
    axessRawNM.set_title(keystr)
    axessRawNM.set_ylim([0,1.1])
      
    plotfitcorr(raw_df,axessRawNM_scatter,'snr','FractionCorrect', 'mod', 'a', 'av', ['mistyrose','lightpink','lightcoral','indianred','brown','maroon'], 'o', ['cls'])
    axessRawNM_scatter.set_title(keystr)
    # twoway rm statistic tests
    try:
        testRes = AnovaRM(data=raw_df, depvar='FractionCorrect',
            subject='cls', within=['snr','mod']).fit().anova_table
        testRes_sig = testRes[testRes['Pr > F']<0.05] 
        for tt in range(testRes_sig.shape[0]):
            axessRawNM_scatter.text(0.7,0.3+tt*0.05,testRes_sig.index[tt]+' p='+str("{:.1e}".format(testRes_sig.iloc[tt,-1])),fontsize=7)
    except:
        pass
    plotfitcorr(raw_df,axessRawFR_scatter,'snr','FR', 'mod', 'a', 'av', ['mistyrose','lightpink','lightcoral','indianred','brown','maroon'], 'o', ['cls'])
    axessRawFR_scatter.set_title(keystr)
    # twoway rm statistic tests
    try:
        testRes = AnovaRM(data=raw_df, depvar='FR',
            subject='cls', within=['snr','mod']).fit().anova_table
        testRes_sig = testRes[testRes['Pr > F']<0.05] 
        for tt in range(testRes_sig.shape[0]):
            axessRawFR_scatter.text(0.7,0.3+tt*0.05,testRes_sig.index[tt]+' p='+str("{:.1e}".format(testRes_sig.iloc[tt,-1])),fontsize=7)
    except:
        pass
    
    return raw_df

def plotfitcorr(slopeThres_df1,axess4,catstr, varStr, compCol, xvar, yvar, colorlist, markerSymb, sortlist):
    for modtest,colr in zip(sorted(slopeThres_df1[catstr].unique()),colorlist):
        slopeThres_temp = slopeThres_df1[slopeThres_df1[catstr]==modtest]
        axess4.scatter(slopeThres_temp[slopeThres_temp[compCol]==xvar].sort_values(sortlist,kind='mergesort')[varStr].values,
                                    slopeThres_temp[slopeThres_temp[compCol]==yvar].sort_values(sortlist,kind='mergesort')[varStr].values,
                                    alpha=0.5,c=colr,label=modtest,marker=markerSymb,edgecolors='none')                                     
    axisLoBound = min([axess4.get_xlim()[0],axess4.get_ylim()[0]])
    axisHiBound = max([axess4.get_xlim()[1],axess4.get_ylim()[1]])
    axess4.set_xlim([axisLoBound,axisHiBound])
    axess4.set_ylim([axisLoBound,axisHiBound])
    axess4.plot(np.linspace(axisLoBound,axisHiBound,100),np.linspace(axisLoBound,axisHiBound,100),'--',color='gray',linewidth=1.5)
    axess4.set_title(varStr)
    axess4.set_xlabel(xvar)
    axess4.set_ylabel(yvar)
    axess4.legend(frameon=False,loc='upper left')


# MonkeyDate_all = {'Elay':['230420']} #

ResPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/neuralMetric/'
figSavePath = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Figures/neuroMetricFit/'
AVmodPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/AVmodIndex/'
behaveResPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/BehavPerform/'

# ResPathway = '/data/by-user/Huaizhen/Fitresults/neuralMetric/'
# figSavePath = '/data/by-user/Huaizhen/Figures/neuroMetricFit/'
# AVmodPathway = '/data/by-user/Huaizhen/Fitresults/AVmodIndex/'
# behaveResPathway = '/data/by-user/Huaizhen/Fitresults/BehavPerform/'
MonkeyDate_all = {'Elay':['230420','230503','230509','230525','230531',
                        '230602','230606','230608','230613','230616','230620','230627',
                        '230705','230711','230717','230718','230719','230726','230728',
                        '230802','230808','230810','230814','230818','230822','230829'], 
                  'Wu':['230809','230815','230821','230830']}

DVstr = 'FractionCorrect' # 'FractionCorrect','dprime'

# filter neurons based on different rules
df_avMod_all = pickle.load(open(AVmodPathway+'AVmodTTestDF.pkl','rb'))
df_avMod_all_sig = df_avMod_all[df_avMod_all['pval']<0.001] 
print(df_avMod_all_sig.to_string())

# plot neurometric fit according to neuron labels
fig2, axess2 = plt.subplots(2,3,figsize=(12,8)) 
fig3, axess3 = plt.subplots(2,3,figsize=(12,8)) 
fig4, axess4 = plt.subplots(2,3,figsize=(12,8)) 

slopeThres_df1 = pd.DataFrame()
raw_df = {}
for mm,(Monkey,Date) in enumerate(MonkeyDate_all.items()):
    xy_population = {'A+V':{'a':[np.empty((0,6)),np.empty((0,6)),np.empty((0,6)),[]],'av':[np.empty((0,6)),np.empty((0,6)),np.empty((0,6)),[]]},
                        'A':{'a':[np.empty((0,6)),np.empty((0,6)),np.empty((0,6)),[]],'av':[np.empty((0,6)),np.empty((0,6)),np.empty((0,6)),[]]},
                        'V':{'a':[np.empty((0,6)),np.empty((0,6)),np.empty((0,6)),[]],'av':[np.empty((0,6)),np.empty((0,6)),np.empty((0,6)),[]]}}

    for session in Date:  
        print(Monkey+session+' in plotting....')   
        # get cluster has significant AV modulation
        df_avMod_sess = df_avMod_all_sig[(df_avMod_all_sig['Monkey']==Monkey) & (df_avMod_all_sig['session_cls'].str.contains(session))]   
        #plot neurometric fun 
        data = pickle.load(open(ResPathway+Monkey+'_'+session+'_neuroMetric.pkl','rb'))

        for cc,sesscls in enumerate(df_avMod_sess['session_cls'].unique().tolist()):
            df_avMod_sess_temp = df_avMod_sess[df_avMod_sess['session_cls']==sesscls]
            cls = sesscls[sesscls.index('_')+1:]
            print(cls+' in plotting')
            data_temp = data[cls].copy()
            if len(data_temp)>0:
                # the units significantly modified by snr and vid  
                if sorted(df_avMod_sess_temp['mod'].tolist())==['A','V'] : #and cls=='cls91_ch10_good':
                    print('glmfit A+V  sig')
                    xy_population = catSessCls(xy_population,data_temp,sesscls,'A+V')
                elif sorted(df_avMod_sess_temp['mod'].tolist())==['A']:
                    print('glmfit A  sig')
                    xy_population = catSessCls(xy_population,data_temp,sesscls,'A')
                elif sorted(df_avMod_sess_temp['mod'].tolist())==['V']:
                    print('glmfit V  sig')
                    xy_population = catSessCls(xy_population,data_temp,sesscls,'V')

    # fit population neurometric
    raw_df1 = PlotPopulation(xy_population,'A+V',axess2[mm,0],axess3[mm,0],axess4[mm,0])
    raw_df1['modTest'] = 'A+V'
    raw_df2 = PlotPopulation(xy_population,'A',axess2[mm,1],axess3[mm,1],axess4[mm,1])
    raw_df2['modTest'] = 'A'
    raw_df3 = PlotPopulation(xy_population,'V',axess2[mm,2],axess3[mm,2],axess4[mm,2])
    raw_df3['modTest'] = 'V'
    raw_df[Monkey] = pd.concat((raw_df1,raw_df2,raw_df3))

fig3.tight_layout()
fig2.tight_layout()
fig4.tight_layout()
fig2.savefig(figSavePath+'AllstiModulated_RawNM_scatter.png')
fig4.savefig(figSavePath+'AllstiModulated_RawFR_scatter.png')
fig3.savefig(figSavePath+'AllstiModulated_RawNM_violin.png')
plt.close(fig2)
plt.close(fig3)
plt.close(fig4)

print('end')


