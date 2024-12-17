import mat73
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.lines as mlines


figformat = 'svg'
fontsizeNo = 16
fontnameStr = 'Arial'#DejaVu Sans'

wavformPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/wavformStruct/'
RTFfilepath = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/STRF/'
figSavePath = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Figures/STRFfig/'

# wavformPathway = '/home/huaizhen/Documents/MonkeyAVproj/data/Fitresults/wavformStruct/'
# RTFfilepath = '/home/huaizhen/Documents/MonkeyAVproj/data/Fitresults/STRF/'
# figSavePath = '/home/huaizhen/Documents/MonkeyAVproj/data/Figures/STRFfig/'

# read all neurons used in final analysis
clsInfo,SPKwavform_ALLmonk = pickle.load(open(wavformPathway+'AllUnits_spkwavform_Wuless.pkl','rb')) 
clsInfo_part = clsInfo[~clsInfo['strfSig'].isna()]
# read FM and RD parameter in the dmr stimulus
FMRD = mat73.loadmat(RTFfilepath+'DMR_FM+RD.mat') 
FMall_full = FMRD['FM']
RDall_full = FMRD['RD']
#RTFhist binedge
FMall = [-50,-30,-10,10,30,50]#[FMall_full[i] for i in np.arange(0,len(FMall_full),4000,dtype=int)]
RDall = [0,1,2,3,4]#[RDall_full[i] for i in np.arange(0,len(RDall_full),5000,dtype=int)]

#load data saved from matlab for each monkey
data = mat73.loadmat(RTFfilepath+'Elay_RTFHistStruct.mat')['RTFstruct']
datadf_Elay = pd.DataFrame({'Monkey':np.array(data['Monkey']).flatten().tolist(),
                            'session':np.array(data['session']).flatten().tolist(),
                            'blockname':np.array(data['blockname']).flatten().tolist(),
                            'cls':np.array(data['cls']).flatten().tolist(),
                            'PLI':np.array(data['PLI']).flatten().tolist(),
                            'RDHistA1': [arr[0] if isinstance(arr,(int,float)) else list(arr)[0] for arr in data['RDHistA1']],
                            'FMHistA1': [arr[0] if isinstance(arr,(int,float)) else list(arr)[0] for arr in data['FMHistA1']],
                            'RDHistB1': [arr[0] if isinstance(arr,(int,float)) else list(arr)[0] for arr in data['RDHistB1']],
                            'FMHistB1': [arr[0] if isinstance(arr,(int,float)) else list(arr)[0] for arr in data['FMHistB1']]})  

data = mat73.loadmat(RTFfilepath+'Wu_RTFHistStruct.mat')['RTFstruct']
datadf_Wu = pd.DataFrame({'Monkey':np.array(data['Monkey']).flatten().tolist(),
                            'session':np.array(data['session']).flatten().tolist(),
                            'blockname':np.array(data['blockname']).flatten().tolist(),
                            'cls':np.array(data['cls']).flatten().tolist(),
                            'PLI':np.array(data['PLI']).flatten().tolist(),
                            'RDHistA1': [arr[0] if isinstance(arr,(int,float)) else list(arr)[0] for arr in data['RDHistA1']],
                            'FMHistA1': [arr[0] if isinstance(arr,(int,float)) else list(arr)[0] for arr in data['FMHistA1']],
                            'RDHistB1': [arr[0] if isinstance(arr,(int,float)) else list(arr)[0] for arr in data['RDHistB1']],
                            'FMHistB1': [arr[0] if isinstance(arr,(int,float)) else list(arr)[0] for arr in data['FMHistB1']]})  
# combine two monkeys data
datadf = pd.concat([datadf_Elay,datadf_Wu],axis=0).reset_index(drop=True)   
clsInfo_part1monk = clsInfo_part.copy()

datadf['session_cls'] = datadf['session']+'_'+datadf['cls']

font_properties = {'fontname': fontnameStr, 'fontsize': fontsizeNo-4}
fig, axess_all = plt.subplots(2,2,figsize=(10,10),sharex='col',sharey='col')
PLI_all = pd.DataFrame()

#loop through sigSTRF and nsigSTRF
for aa,strfsig in enumerate([True,False]):
    PLI_temp = []
    #obtain neuron names used in all other analysis
    clsInfo_temp = clsInfo_part1monk[clsInfo_part1monk['strfSig']==strfsig].reset_index(drop=True)
    
    histAll = np.zeros((len(RDall)-1,len(FMall)-1))
    histAllmax = np.zeros((len(RDall)-1,len(FMall)-1))
    # loop through neurons
    for cls in clsInfo_temp.session_cls.unique():
        datadf_temp = datadf[datadf['session_cls']==cls].sort_values('blockname').reset_index(drop=True).loc[1]
        # obtain 2d hist for trialA, trialB, and dmr stimulus
        histA,RDedge,FMedge = np.histogram2d(datadf_temp['RDHistA1'],datadf_temp['FMHistA1'],bins=[RDall,FMall])
        histB,RDedge,FMedge = np.histogram2d(datadf_temp['RDHistB1'],datadf_temp['FMHistB1'],bins=[RDall,FMall])       
        histS,RDedge,FMedge=np.histogram2d(RDall_full,FMall_full,bins=[RDall,FMall])    #Stimulus distribution - used to remove bia
        #normalize 2d hist
        hist = (histA+histB)/np.sum(histA+histB)
        hist_rmbias = hist/histS
        hist_rmbias = hist_rmbias/np.sum(hist_rmbias)
        # add current hist to population hist (histAll)
        histAll = histAll+hist_rmbias 
        histAllmax = histAllmax+np.where(hist==np.max(hist),1,0)
        PLI_all = pd.concat([PLI_all,pd.DataFrame({'cls':[cls],'strfSig':[strfsig],'PLI':[datadf_temp['PLI']]})]) 
 
    print('strfsig '+str(strfsig))
    print(histAllmax)
    # plot 2d histogram
    axess_all[aa,0].imshow(histAllmax,origin='lower')
    axess_all[aa,0].set_xlabel('Temporal modulation (Hz)',**font_properties)
    axess_all[aa,0].set_ylabel('Spectral modulation (cycle/oct)',**font_properties)
    axess_all[aa,0].tick_params(axis='both', which='major', labelsize=fontsizeNo-8)
    axess_all[aa,0].set_yticks([0,len(RDall)-2])
    axess_all[aa,0].set_yticklabels([str(np.round(RDedge[0],decimals=0)), str(np.round(RDedge[-1],decimals=0))])
    axess_all[aa,0].set_xticks([0,len(FMall)-2])
    axess_all[aa,0].set_xticklabels([str(np.round(FMedge[0],decimals=0)), str(np.round(FMedge[-1],decimals=0))])
# plot PLI histgram
font_properties = {'fontname': fontnameStr, 'fontsize': fontsizeNo-4}
mv_stats,mv_pval = stats.mannwhitneyu(PLI_all[PLI_all['strfSig']==0]['PLI'].values,
                                    PLI_all[PLI_all['strfSig']==1]['PLI'].values,alternative='less')

PLI_all['strfSig_label'] = PLI_all['strfSig'].map({False: 'nSTRF', True: 'STRF'})
sns.histplot(PLI_all,x='PLI',hue='strfSig_label',stat = 'probability',bins=10,ax = axess_all[1,1]
             ,palette={'nSTRF':'darkgray','STRF':'black'})
# Access and modify the legend
legend = axess_all[1,1].get_legend()
if legend is not None:
    legend.set_title(None)  # Remove the title 
axess_all[1,1].set_xlabel('Phase Locking Index',**font_properties)
axess_all[1,1].set_ylabel('Probability',**font_properties)
axess_all[1,1].tick_params(axis='both', which='major', labelsize=fontsizeNo-4) 
axess_all[1,1].yaxis.set_ticks([0,0.1,0.2,0.3])
fig.tight_layout()
fig.savefig(figSavePath+'RTF4STRF+NSTRF.'+figformat, dpi=200)
fig.savefig(figSavePath+'RTF4STRF+NSTRF.png', dpi=100)
plt.close(fig)


 