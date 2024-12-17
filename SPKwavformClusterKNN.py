import pandas as pd
import numpy as np
import seaborn as sns
import seaborn.objects as so
import mat73
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import re
from scipy import stats
# import umap
from sharedparam import getMonkeyDate_all,getGranularChan_all,neuronfilterDF
from scipy.stats import pearsonr,zscore, mannwhitneyu, chi2_contingency
from scipy.stats.contingency import expected_freq
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from scipy.signal import find_peaks,find_peaks_cwt
from matplotlib import colors
from collections import Counter
from matplotlib.legend_handler import HandlerLine2D
import statsmodels.api as sm
from statsmodels.formula.api import ols
from matplotlib.sankey import Sankey
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from statsmodels.stats.anova import AnovaRM
import statsmodels.formula.api as smf
import mat73

def getpeak2troughDur(cls,wavform):
    spikefs = 24414.0625/1000 #kHz
    clsNum = int(cls[3:cls.find('_')])
    clsRow = np.where(wavform['unitIDs']==clsNum)[0]
    wavfmean_temp = wavform['waveFormsMean_1chan'][clsRow,:]
    baselinewav = np.concatenate((wavfmean_temp[0][:20],wavfmean_temp[0][-20:]))
    baselinemean = np.mean(baselinewav)

    positivepeaks,_ = find_peaks(wavfmean_temp[0])
    negativepeaks = np.where(wavfmean_temp[0]==wavfmean_temp[0].min())[0]

    if len(positivepeaks)>=2:
        if (positivepeaks<negativepeaks).all():
            positivepeaks = np.array([positivepeaks.max()])
        elif (positivepeaks>negativepeaks).all():
            positivepeaks = np.array([positivepeaks.min()])
        else:
            positivepeaks_beforeTrough = positivepeaks[positivepeaks<negativepeaks]
            positivepeaks_afterTrough = positivepeaks[positivepeaks>negativepeaks]
            positivepeaks = np.array([positivepeaks_beforeTrough[-1],positivepeaks_afterTrough[0]])

    firstpeak2trough_ratio = np.nan
    secondpeak2trough_ratio = np.nan
    halftroughAmpDur = np.nan
    peak2peakdur = np.nan
    troughAmp = np.nan
    maxpeakAmp = np.nan
    spkLabel = 'RS/FS' 

    maxpeakAmp = wavfmean_temp[0].max()-baselinemean # find global maximum amplitude
    troughAmp = wavfmean_temp[0][negativepeaks[0]]-baselinemean
    troughwav = wavfmean_temp[0][(negativepeaks[0]-15):(negativepeaks[0]+15)]-baselinemean
    halfAmpDurIndst = np.where(np.abs(troughwav[:15]-troughAmp/2)==np.min(np.abs(troughwav[:15]-troughAmp/2)))[0][0]
    halfAmpDurIndend = np.where(np.abs(troughwav[15:]-troughAmp/2)==np.min(np.abs(troughwav[15:]-troughAmp/2)))[0][0]+15
    halftroughAmpDur = (halfAmpDurIndend - halfAmpDurIndst)/spikefs # in ms
    if maxpeakAmp>np.abs(troughAmp):
        spkLabel = 'positive'   
    else:     
        if len(positivepeaks)>=1:
            # obtain first peak2trough ratio
            firstpeak2trough_ratio = np.abs((wavfmean_temp[0][positivepeaks[0]]-baselinemean)/troughAmp)
            # obtain second peak2trough ratio
            if len(positivepeaks) == 2:
                secondpeak2trough_ratio = np.abs((wavfmean_temp[0][positivepeaks[1]]-baselinemean)/troughAmp)
                peak2peakdur = (positivepeaks[1]-positivepeaks[0])/spikefs # in ms
                if firstpeak2trough_ratio>0.2:
                    if peak2peakdur<=1:
                        spkLabel = 'triphasic'
                    else:
                        spkLabel = 'compound'
        else:
            print(cls+' no peaks detected!')
            spkLabel = 'NaN'


    # obatain trough-peak time
    troughTim = negativepeaks[0]
    peak2troughTim = np.where(wavfmean_temp[0,troughTim:]==wavfmean_temp[0,troughTim:].max())[0][0]/spikefs # in ms

    return peak2troughTim,halftroughAmpDur,firstpeak2trough_ratio,secondpeak2trough_ratio,peak2peakdur,maxpeakAmp,troughAmp,spkLabel,positivepeaks,negativepeaks

def estiAVmodindex(snra,fra,fca,snrav,frav,fcav):
    #fra and frav should save at the same snr order
    modind_df = pd.DataFrame()
    AVmoddf_temp = pd.DataFrame()
    for ii,snr in enumerate(snra):
        modind_df = pd.concat((modind_df,pd.DataFrame({'snr':[snr]*2,
                                                       'AVmodInd':[(frav[ii]-fra[ii])/(frav[ii]+fra[ii])]*2,
                                                       'mod':['a','av'],
                                                       'zscore_fr':[fra[ii],frav[ii]],
                                                       'FractionCorrect':[fca[ii],fcav[ii]]})))
        AVmoddf_temp = pd.concat((AVmoddf_temp,pd.DataFrame({'AVmod'+str(int(snr)):[(frav[ii]-fra[ii])/(frav[ii]+fra[ii])]})),axis=1)

    return modind_df,AVmoddf_temp

def plotumap(u,numericCat,cmap,labels,figSavePath,figname):
    fig, axess = plt.subplots(1,1,figsize=(10,10)) #  
    axess.scatter(u[:,0],u[:,1],c=numericCat,cmap=cmap)
    # Annotating each point with its label
    for i, label1 in enumerate(labels):
        axess.text(u[i, 0], u[i, 1], label1, fontsize=5, ha='right', va='bottom')
    fig.tight_layout()
    fig.savefig(figSavePath+figname, dpi=600)
    plt.close(fig) 

def plotumapWavform(u,SPKwavform,numericCat,colorGrouplabel,cmapstr,figSavePath,figname, vminmax = [-0.45,0.45]):
    fig, axess = plt.subplots(1,1,figsize=(23,15)) #  
    parentAxis = plt.gca()
    # axess.scatter(u[:,0],u[:,1],c='white',alpha=0)
    axess.scatter(u[:,0],u[:,1],c=numericCat,cmap=cmapstr)

    # generate colormap according to numericCat data
    norm = mcolors.Normalize(vmin=vminmax[0], vmax=vminmax[1])
    # Choose a colormap
    cmap = getattr(plt.cm, cmapstr, plt.cm.viridis)  # Defaults to 'viridis' if cmapstr is invalid
    # Apply colormap to normalized data
    colors = cmap(norm(numericCat))

    # Annotating each point with its wavform
    # Loop through each scatter plot point and add a mini line plot
    for x, y, y_mini,lincolor in zip(u[:,0], u[:,1], SPKwavform,colors):
        # Creating inset axes (mini plot) at each scatter plot point
        ax_inset = inset_axes(parentAxis, width="150%", height="70%", loc='lower left',
                            bbox_to_anchor=(x-0.1, y-0.1, 0.3, 0.3),
                            bbox_transform=parentAxis.transData) #
        ax_inset.plot(y_mini, color=lincolor,linewidth=0.7)
        ax_inset.set_xlim([0,102])
        ax_inset.axis('off')  # Turn off axis for mini plot  
    # Add a color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # You can technically pass your numericCat array here, but it's not necessary
    fig.colorbar(sm, ax=axess, orientation='vertical', shrink=0.6, label=colorGrouplabel)

    # fig.tight_layout()
    fig.savefig(figSavePath+figname, dpi=600)
    plt.close(fig) 

def calculate_corr_pval(group,colA,colB):
    correlation, p_value = pearsonr(group[colA], group[colB])
    return pd.Series({'correlation': correlation, 'p_value': p_value})

# run in monty
MonkeyDate_all = getMonkeyDate_all()
wavformPathway = '/home/huaizhen/Documents/MonkeyAVproj/data/Fitresults/wavformStruct/'
AVmodPathway = '/home/huaizhen/Documents/MonkeyAVproj/data/Fitresults/AVmodIndex/'
figSavePath = '/home/huaizhen/Documents/MonkeyAVproj/data/Figures/SPKwavshape/'
NMPathway = '/home/huaizhen/Documents/MonkeyAVproj/data/Fitresults/neuralMetric/'
STRFexcelPath = '/home/huaizhen/Documents/MonkeyAVproj/data/Fitresults/STRF/'

# debug in local pc
# MonkeyDate_all = {'Wu':['240710'],'Elay':['230616'],} #
# wavformPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/wavformStruct/'
# AVmodPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/AVmodIndex/'
# NMPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/neuralMetric/'
# figSavePath = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Figures/SPKwavshape/'
# STRFexcelPath = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/STRF/'


filterules = 'all' # 'ttest' 'glm'
STRFstr = 'addSTRFcol'

# param for nm
bintim = 200 #200
signalStrlist = ['hit'] #['hit'] ['hit','miss'] ---signal method in NM
noiseStr='aCRnoise' #'aCRnoise' 'aCRvCRnoise' 'aMISSnoise' ---noise method in NM
bintimstr = '_'+str(bintim)+'msbinRawPSTH'
extrastr = 'align2coo'#'align2coo' # 'align2js'
pklnamestr = ('+').join(signalStrlist)+'_'+noiseStr+bintimstr+'_allcls_'+extrastr 

figformat = 'svg'
fontsizeNo = 20
fontnameStr = 'DejaVu Sans'#DejaVu Sans'

clsInfo = pd.DataFrame()
SPKwavform_ALLmonk = np.empty((0,102))
for Monkey,Date in MonkeyDate_all.items():
    df_avMod_all_sig,(clscolstr,_,_) = neuronfilterDF(AVmodPathway,STRFexcelPath,filterules,Monkey,STRFstr)

    for Date_temp in Date:  
        fig, axess = plt.subplots(1,1,figsize=(12,12))
        print('.............get spkwavform in '+Monkey+Date_temp+'.............')
        wavformStruct = mat73.loadmat(wavformPathway+Monkey+'-'+Date_temp+'_spkWavShape.mat')['wf_filter']
        NMdata = pickle.load(open(NMPathway+Monkey+'_'+Date_temp+'_neuroMetricFit'+pklnamestr+'.pkl','rb'))

        for cc, cls in enumerate(wavformStruct['clsName']):
            if (df_avMod_all_sig['session_cls']==Date_temp+'_'+cls[0]).any(): 
                print(cls[0])
                df_avMod_sess_temp = df_avMod_all_sig[df_avMod_all_sig[clscolstr]==Date_temp+'_'+cls[0]] 
                mod = "+".join(sorted(df_avMod_sess_temp['iv'].values)) 
                strf = df_avMod_sess_temp['STRFsig'].unique()[0]

                data_temp = NMdata[cls[0]].copy()               
                # spkwavformArr_temp = wavformStruct['waveForms_1chan'][cc,:,:] #numSPKs X time
                spkwavformArr_temp = wavformStruct['waveFormsMean_1chan'][cc,:].reshape((1,-1)) #1 X time
                # av modindex
                _, AVmoddf_temp= estiAVmodindex(data_temp['x_raw_a'],data_temp['y_rawfr2_a'],
                                                    data_temp['y_raw_a'],data_temp['x_raw_av'],
                                                    data_temp['y_rawfr2_av'],data_temp['y_raw_av']) # raw fr average                
                if np.isnan(spkwavformArr_temp).any():
                    print('nan wavform in '+cls[0])
                    pass
                else:
                    # get waveform shape parameters
                    peak2troughTim,halftroughAmpDur,firstpeak2trough_ratio,secondpeak2trough_ratio,\
                        peak2peakdur,maxpeakAmp,troughAmp,spkLabel,positivepeaks,negativepeaks = getpeak2troughDur(cls[0],wavformStruct)
                    # plot waveform session by session
                    axess.plot((cc-1)*20+np.arange(0,len(spkwavformArr_temp[0]),1),(cc-1)*40+spkwavformArr_temp[0])
                    allpeakInds = np.concatenate((positivepeaks,negativepeaks))
                    axess.scatter((cc-1)*20+allpeakInds,(cc-1)*40+spkwavformArr_temp[0][allpeakInds],s=16,c='black')
                    axess.text((cc-1)*20+180,(cc-1)*40,cls[0],horizontalalignment='right',verticalalignment='center',fontsize=fontsizeNo)

                    SPKwavform_ALLmonk = np.concatenate((SPKwavform_ALLmonk,spkwavformArr_temp),axis=0)
                    numSPKs = spkwavformArr_temp.shape[0]
                    clsInfo_temp = pd.DataFrame({'Monkey':[Monkey]*numSPKs,'session_cls':[Date_temp+'_'+cls[0]]*numSPKs,'sess':[Date_temp]*numSPKs,
                                                'chan':[np.int64(re.search(r'_ch(.*?)_',Date_temp+'_'+cls[0]).group(1))]*numSPKs,
                                                'iv':[mod]*numSPKs,'strfSig':[strf]*numSPKs,
                                                'peak2troughDur':[peak2troughTim]*numSPKs,'halfAmpDur':[halftroughAmpDur]*numSPKs,
                                                'firstpeak2trough_ratio':[firstpeak2trough_ratio]*numSPKs,'secondpeak2trough_ratio':[secondpeak2trough_ratio]*numSPKs,
                                                'peak2peakdur':[peak2peakdur]*numSPKs,'maxpeakAmp':[maxpeakAmp]*numSPKs,
                                                'troughAmp':[troughAmp]*numSPKs,'spkLabel':[spkLabel]*numSPKs})
                    AVmoddf_temp = AVmoddf_temp.loc[[0]*numSPKs]
                    clsInfo = pd.concat([clsInfo,pd.concat((clsInfo_temp,AVmoddf_temp),axis=1)],axis=0).reset_index(drop=True)
        axess.set_title(Monkey+'_'+Date_temp)
        fig.tight_layout()
        fig.savefig(figSavePath+Monkey+'_'+Date_temp+'_wavformParamCheck.'+figformat,format = figformat)
        plt.close(fig)
    print('number of units for '+Monkey+': '+str(len(clsInfo['session_cls'].unique())))
pickle.dump([clsInfo,SPKwavform_ALLmonk],open(wavformPathway+'AllUnits_spkwavform_Wuless_'+extrastr+'_'+str(bintim)+'ms.pkl','wb'))  

clsInfo,SPKwavform_ALLmonk = pickle.load(open(wavformPathway+'AllUnits_spkwavform_Wuless_'+extrastr+'_'+str(bintim)+'ms.pkl','rb')) 


print(clsInfo)
clsInfo_part = clsInfo[~clsInfo['strfSig'].isna()]
gchan_all = getGranularChan_all()
clsInfo_part_realign = pd.DataFrame()
# realign chan relative to the granular chan
for sess in list(clsInfo_part.sess.unique()):
    clsInfo_part_temp = clsInfo_part[clsInfo_part['sess']==sess]
    clsInfo_part_temp['chan'] = clsInfo_part_temp['chan'].values-gchan_all[sess]
    clsInfo_part_realign = pd.concat([clsInfo_part_realign,clsInfo_part_temp])




########################## plot recorded channel of strf vs nonstrf neurons
fig, axess = plt.subplots(1,1,figsize=(6,6)) #  
df4chisqu = clsInfo_part_realign[['chan','strfSig']]
contingency_table = pd.crosstab(df4chisqu['chan'],df4chisqu['strfSig'])
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
sns.countplot(clsInfo_part_realign,x='chan',hue='strfSig',dodge=True,stat='percent',ax=axess,hue_order=[False,True],palette=['darkgray','dimgray'])
axess.text(0.98, 0.77, r'$\chi^2$ : p = '+str(np.round(p_value,decimals=5),), 
    transform=axess.transAxes,  # Specify the position in axis coordinates
    fontsize=fontsizeNo-8,             # Font size
    fontname=fontnameStr,
    verticalalignment='top', # Align text at the top
    horizontalalignment='right') # Align text to the right
font_properties = {'fontname': fontnameStr, 'fontsize': fontsizeNo-4}
axess.set_xlabel('Depth',**font_properties)
axess.set_ylabel('Percent of Units',**font_properties)
xticks = axess.get_xticks() 
xticklabels = [label.get_text() for label in axess.get_xticklabels()]
print(xticklabels)
zero_tick = xticks[xticklabels.index('0')]
axess.set_xticks([0,zero_tick,len(clsInfo_part_realign['chan'].unique())])
axess.set_xticklabels(['superficial','granular','deep'])
axess.tick_params(axis='both', which='major', labelsize=fontsizeNo-8)
new_labels = ['non-significant', 'significant']
handles, labels = axess.get_legend_handles_labels()
axess.legend(handles=handles[:2], labels=new_labels, loc='upper right',
            handler_map={mlines.Line2D: HandlerLine2D(numpoints=1)},frameon=False,
            bbox_to_anchor=(1, 1), fontsize=fontsizeNo-8,title='STRF',title_fontsize=fontsizeNo-6)  
fig.tight_layout()
fig.savefig(figSavePath+'recordedChannel_Cmp.'+figformat, dpi=600)
plt.close(fig) 

####################### compare spkwavform width in STRF vs non-TSRF
clsInfo_part_part = clsInfo_part[clsInfo_part['spkLabel'].isin(['RS/FS','triphasic','compound','positive'])]
####combine two monkey  
fig, axess = plt.subplots(1,1,figsize=(6,6)) #  
sns.boxplot(clsInfo_part_part,y='strfSig',x='peak2troughDur',orient='h',
            palette={False:'darkgray',True:'dimgray'},
            linewidth=2,dodge=True,ax=axess)
axess.set_yticklabels(['non-significant', 'significant'],rotation=90, verticalalignment='center')
mv_stats,mv_pval = stats.mannwhitneyu(clsInfo_part_part[clsInfo_part_part['strfSig']==0]['peak2troughDur'].values,
                                    clsInfo_part_part[clsInfo_part_part['strfSig']==1]['peak2troughDur'].values,alternative='greater')

# if mv_pval<0.0001:
#     axess.text(0.5, 0.62, 'p < 0.0001 ', 
#         transform=axess.transAxes,  # Specify the position in axis coordinates
#         fontsize=fontsizeNo-10,             # Font size
#         fontname=fontnameStr,
#         verticalalignment='top', # Align text at the top
#         horizontalalignment='right') # Align text to the right

font_properties = {'fontname': fontnameStr, 'fontsize': fontsizeNo-4}
axess.set_xlabel('Trough to Peak Duration (ms)',**font_properties)
axess.set_ylabel('STRF',**font_properties)
axess.tick_params(axis='both', which='major', labelsize=fontsizeNo-8)
axess.set_xticks(list(np.arange(0,2.5,0.5)))
# ######################## plot strf vs nonstrf wavforms
clsInfo_part = clsInfo[~clsInfo['strfSig'].isna()]
axess2 = axess.twinx()
for aa,(strfSig,strfSigStr) in enumerate(zip(sorted(list(clsInfo_part['strfSig'].unique())),['non-significant STRF','significant STRF'])):
    clsInfo_cc = clsInfo_part[clsInfo_part['strfSig']==strfSig]
    rows_cc = list(clsInfo_cc.index)
    SPKwavform_ALLmonk_cc = SPKwavform_ALLmonk[rows_cc,25:75]
    yshift = 450#100
    xshift =0
    axess2.plot(np.arange(0,SPKwavform_ALLmonk_cc.shape[1])/(2*SPKwavform_ALLmonk_cc.shape[1])-0.51+xshift,SPKwavform_ALLmonk_cc.T-200-yshift*aa,color='gray',linewidth=0.2)
    # axess_flat.plot(np.arange(0,SPKwavform_ALLmonk_cc.shape[1]),np.mean(SPKwavform_ALLmonk_cc-yshift*aa,axis=0),color='red',linewidth=2)
    axess2.plot(np.arange(0,SPKwavform_ALLmonk_cc.shape[1])/(2*SPKwavform_ALLmonk_cc.shape[1])-0.51+xshift,np.median(SPKwavform_ALLmonk_cc-200-yshift*aa,axis=0),color='red',linewidth=3)
    # axess2.text(0.58,0.91-0.2*aa, strfSigStr,horizontalalignment='left',verticalalignment='center',fontsize=fontsizeNo-8,fontname = fontnameStr,transform=axess2.transAxes)
    axess2.set_ylim([-900,40])
    axess2.set_yticks([])
    axess2.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
fig.tight_layout()
fig.savefig(figSavePath+'peak2troughDur_Cmp+wavforms4eachSTRFsig.'+figformat, dpi=600)
plt.close(fig) 



 

    


       
    






          

                


