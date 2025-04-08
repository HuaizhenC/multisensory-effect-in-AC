import time
from datetime import timedelta
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from matplotlib import pyplot as plt 
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from spikeUtilities import decodrespLabel2str,SortfilterDF
from sharedparam import getMonkeyDate_all,neuronfilterDF
import random
cpus = 10

#pick random neurons from all neurons recorded from one monkey
def pickneuronpopulations(Monkey,STRFstr):
    clsInfo,_ = pickle.load(open(wavformPathway+'AllUnits_spkwavform_Wuless.pkl','rb')) 
    clsInfo_temp = clsInfo[clsInfo['Monkey']==Monkey]                   
    # filter neurons based on different rules
    df_avMod_all_sig_ori,_ = neuronfilterDF(AVmodPathway,STRFexcelPath,'ttest',Monkey,STRF=STRFstr)
    df_avMod_all_sig_ori = df_avMod_all_sig_ori[df_avMod_all_sig_ori['session_cls'].isin(clsInfo_temp.session_cls.values)]
    # randomly pick clusters  
    df_avMod_all_sig_unique = df_avMod_all_sig_ori.groupby(['Monkey','session','cls','session_cls','STRFsig']).size().reset_index()                    
    df_avMod_all_sig = df_avMod_all_sig_unique.groupby('STRFsig').sample(30,replace=False).sort_values(by=['Monkey','session'])
    # print(Monkey+' '+STRFstr)
    # print(df_avMod_all_sig_unique.groupby('STRFsig').size().to_string())
    return df_avMod_all_sig
# sample trials of each condition for each neuron
def sampTrials(Monkey,Date,alignKeys,df_avMod_all_sig):
    # initialize dataframe to save psth time series data of each neuron
    AllSU_psth_condAve = pd.DataFrame()
    AllSU_psth_trialBYtrial = pd.DataFrame()
    for Date_temp in Date: 
        print('session date:'+Date_temp)
        ######## columns in rawPSTH_df ['Monkey','sess','cls','respLabel','AVoffset', 'snr', 'trialMod', 'snr-shift', 'trialNum','baselineFR']+frate
        ######## modify line106-107 in reshapePSTH when switch input files
        rawPSTH_df_all = pickle.load(open(Pathway+Monkey+'_'+Date_temp+'_allSU+MUA_alltri'+alignKeys+'_nonoverlaptimwin_50msbinRawPSTH_df.pkl','rb'))  # Nneurons X clsSamp X trials X bootstrapTimes
        psthCol_all  = [s for s in list(rawPSTH_df_all.columns) if 'frate' in s]

        # get clips of psth during window defined by timwinStart 
        binedge_all = np.arange(-1.5,0,bin) #align2coo psth window is [-1,1] align2js psth window is [-1.5,0]                       
        frateNO = np.arange(min(range(len(binedge_all)), key=lambda i: abs(binedge_all[i] - timwinStart[0])) ,
                            min(range(len(binedge_all)), key=lambda i: abs(binedge_all[i] - timwinStart[1]))+1,1)     
        psthCol =  [s for s in psthCol_all if float(s[5:]) in frateNO]
        binedge = binedge_all[frateNO]
        rawPSTH_df = rawPSTH_df_all.drop(columns=list(set(psthCol_all)-set(psthCol)))
        #filter trial conditions 
        rawPSTH_df_temp,_ = SortfilterDF(decodrespLabel2str(rawPSTH_df),filterlable =filterdict)  # filter out trials 
        
        # preprocess each cls separately in the current session 
        sess_cls_namelist = list(df_avMod_all_sig[df_avMod_all_sig['session']==Date_temp]['cls'])
        for cc,cls in enumerate(sess_cls_namelist):                         
            rawPSTH_df_temp_cls = rawPSTH_df_temp[rawPSTH_df_temp['cls']==cls].copy().reset_index()
            # in case sample the same cluster multiple times, so add differentiation here
            clnamenew = cls+'_'+str(cc)
            rawPSTH_df_temp_cls['cls'] = clnamenew 
            # if need to balance trials across conditions by bootstrapping  
            if balanceSamples:                           
                rawPSTH_df_temp_cls = rawPSTH_df_temp_cls.groupby(by=['sess','cls','trialMod','snr','respLabel','AVoffset']).sample(50,random_state=random.randint(1, 10000),replace=True) 

            # zscored psth of the current cluster for each trial
            rawPSTH_df_temp_cls_tbyt = rawPSTH_df_temp_cls.copy()
            rawPSTH_df_temp_cls_tbyt[psthCol] = (rawPSTH_df_temp_cls_tbyt[psthCol] - rawPSTH_df_temp_cls_tbyt[psthCol].mean(skipna=True))\
                                                    /rawPSTH_df_temp_cls_tbyt[psthCol].std(skipna=True)
            # concatenate trial by trial psth of all clusters
            AllSU_psth_trialBYtrial = pd.concat([AllSU_psth_trialBYtrial,rawPSTH_df_temp_cls_tbyt])  

            # average psth of the current cluster in each condition
            psth_df_CondAve = rawPSTH_df_temp_cls.groupby(psthGroupCols)[psthCol].mean().reset_index().sort_values(by=psthGroupCols)
            # gaussian smooth averaged psth , sigma=40ms , binlen 50ms 
            psth_df_CondAve[psthCol] = gaussian_filter1d(psth_df_CondAve[psthCol].values, sigma=0.8, axis=-1)            
            # zscore averaged psth of the current cluster
            psth_df_CondAve[psthCol] = (psth_df_CondAve[psthCol]-psth_df_CondAve[psthCol].mean(skipna=True))\
                                        /psth_df_CondAve[psthCol].std(skipna=True)
            # transfer averaged psth of the current cluster in one row dataframe
            colnames = []
            for (_,dfrow) in psth_df_CondAve[psthGroupCols].iterrows():
                colnames = colnames+['_'.join([str(vv) for vv in list(dfrow.values)])+'_'+tt for tt in psthCol]
            psth_concate_df = pd.DataFrame(psth_df_CondAve[psthCol].values.reshape(1,-1),columns=colnames)        
            # concatenate condition averaged psth of all clusters
            info_temp_df = pd.DataFrame({'Monkey':[Monkey],'sess':[Date_temp],'cls':[clnamenew]})  
            AllSU_psth_condAve = pd.concat([AllSU_psth_condAve,pd.concat([info_temp_df,psth_concate_df],axis=1) ],axis=0) 
    return AllSU_psth_trialBYtrial,AllSU_psth_condAve,psthCol,binedge
# get denoised coefficients of IVs for each cls
def getBeta(AllSU_psth_trialBYtrial,linregCols,psthCol,denoiseMat,AllSU_psth_condAve):
    #get coefficients of IVs of linear regression on PSTH for each cls
    beta_df = pd.DataFrame()
    betaColname = []
    for col_temp in linregCols:
        labels,_ = pd.factorize(pd.to_numeric(AllSU_psth_trialBYtrial[col_temp],errors='ignore'),sort=True)
        AllSU_psth_trialBYtrial[col_temp] = labels.astype(int)
        betaColname = betaColname+[col_temp+'_'+tim for tim in psthCol]
    # estimate each cls in a loop: cls should be saved in the same order as that in AllSU_psth_condAve,psthall
    for (_,dfrow) in AllSU_psth_condAve[['sess','cls']].iterrows():    
        grpdf = AllSU_psth_trialBYtrial[(AllSU_psth_trialBYtrial['sess']==dfrow['sess'])
                                         &(AllSU_psth_trialBYtrial['cls']==dfrow['cls'])]
        reg = LinearRegression(fit_intercept=True,n_jobs = CPUs).fit(grpdf[linregCols].values,grpdf[psthCol].values)# could include interaction terms
        beta = reg.coef_.T#IV coefficients: IVnum X timebin (IVnum in the order as in linregCols)
        beta_df_temp = pd.DataFrame(beta.reshape(1,-1),columns=betaColname)
        info_temp_df = pd.DataFrame({'Monkey':[grpdf.Monkey.values[0]],
                                    'sess':[dfrow['sess']],'cls':[dfrow['cls']]})  
        beta_df = pd.concat([beta_df,pd.concat([info_temp_df,beta_df_temp],axis=1)],axis=0)
    # denoise beta with subspace spanned by population repsonse
    beta_df[betaColname] = np.dot(denoiseMat,beta_df[betaColname].values) # units X (IVnum X timebin)
    return beta_df,betaColname
# get maximum beta coefficient over time for each cls
def rmTimdim(beta_df_denoise,betaColname,linregCols):
    beta_df_max = beta_df_denoise.iloc[:,:3].copy()
    betaarray = beta_df_denoise[betaColname].values
    for colstr in linregCols:
        subcolnum = [i for i,ele in enumerate(betaColname) if colstr in ele]
        subarray = betaarray[:,subcolnum]
        beta_df_max[colstr] = subarray[np.arange(subarray.shape[0]),np.argmax(np.abs(subarray),axis=1)]
    return beta_df_max  
# reshape psth dataframe to array for later project
def reshapePSTH(AllSU_psth,timpnts,psthstcol):
    AllSU_psth_data = AllSU_psth.iloc[:,psthstcol:] 
    cond = int(AllSU_psth_data.shape[1]/timpnts)
    psth = np.empty((AllSU_psth.shape[0],timpnts,cond))
    condstrlist = []
    for cc in range(cond):
        psth[:,:,cc] = AllSU_psth_data.values[:,cc*timpnts:(cc+1)*timpnts] # psth: units X time X cond 
        condstrlist = condstrlist+['_'.join(AllSU_psth_data.columns[cc*timpnts].split('_')[:-1])] # a_5.0_frate0
        # condstrlist = condstrlist+['_'.join(AllSU_psth_data.columns[cc*timpnts].split('_')[:-2])]# a_5.0_fr_0
    return psth,condstrlist
# conduct targeted dimension reduction procedure
def conductTargDimRed(AllSU_psth_trialBYtrial,AllSU_psth_condAve,psthCol):
    psthall = AllSU_psth_condAve.iloc[:,3:].values # units X (cond X timebin)
    AllSU_psth_condAve_denoise = AllSU_psth_condAve.copy()   
    # denoise population PSTH response
    pca = PCA()
    pca.fit(psthall.T)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.where(cumulative_variance >= 0.80)[0][0] + 1  # Add 1 for 1-based index
    pca_optimal = PCA(n_components=num_components,svd_solver='full')
    pca_optimal.fit(psthall.T)
    print('explained variance:'+ str(np.sum(pca_optimal.explained_variance_ratio_[:num_components])))
    denoiseMat = pca_optimal.components_.T @ pca_optimal.components_#get denoise Matrix to project input to the subspace spanned by the retained PCs
    AllSU_psth_condAve_denoise.iloc[:,3:] = denoiseMat @ psthall # units X (cond X timebin)
    # separate the cond and time dimension in the psth data for following projecting
    psth,condstrlist = reshapePSTH(AllSU_psth_condAve_denoise,len(psthCol),psthstcol=3)# psth: units X time X cond 
    # get denoised coefficients of IVs for each cls
    beta_df_denoise,betaColname = getBeta(AllSU_psth_trialBYtrial,linregCols,psthCol,denoiseMat,AllSU_psth_condAve) # units X (IVnum X timebin)
    # remove time dimension from beta
    beta_df_max = rmTimdim(beta_df_denoise,betaColname,linregCols) 
    # orthogonalize beta
    Q,R = np.linalg.qr(beta_df_max[linregCols].values,mode='reduced')# Q: units X linregCond
    # project averaged psth in each condition to orthogonalized beta
    psth_proj = np.tensordot(Q.T,psth,axes=([1],[0])) #psth_proj: linregCond X time X condition

    return psth_proj,condstrlist
#form dataframe for psth project
def formdf(psth_proj,linregCols,condstrlist,binedge,bsref='off'):
    psth_proj_df = pd.DataFrame()
    snrcolor = {'0.0':'cyan','5.0':'orchid','10.0':'magenta','15.0':'limegreen','20.0':'green','25.0':'cornflowerblue','30.0':'royalblue'}
    # snrcolor = {'0.0':'darkgray','5.0':'darkgray','10.0':'darkgray','15.0':'dimgrey','20.0':'dimgrey','25.0':'black','30.0':'black'}
    modsym = {'a':'circle','av':'circle-open','v':'diamond'}
    modline = {'a':'solid','av':'dot','v':'solid'}
    respsym = {'hit':6,'miss':6}

    for tt in range(psth_proj.shape[1]):
        psth_proj_df_ttall = pd.DataFrame()
        for cc in range(psth_proj.shape[2]):                 
            condstrlist_temp = condstrlist[cc]
            try:
                mod,snr,resp = condstrlist_temp.split('_')
            except ValueError:
                mod,snr = condstrlist_temp.split('_')
                resp = 'hit'

            if not mod=='v': #correct snr to the real value for a and av condition
                snr_c = int(float(snr)-20)
            else:
                snr_c = 0          
            psth_proj_df_temp = pd.concat([pd.DataFrame({'time':[binedge[tt]],'mod':[mod],'snr':[snr_c],'resp':[resp],
                                                         'color':[snrcolor[snr]],'symbol':[modsym[mod]],'width':[respsym[resp]],'line':[modline[mod]]}),
                                            pd.DataFrame(psth_proj[:,tt,cc].reshape(1,-1),columns=linregCols)],axis=1)
            psth_proj_df_ttall = pd.concat([psth_proj_df_ttall,psth_proj_df_temp])
        if bsref=='on':
            # refer projections to snr-shift axis in all conditions at the current time point to the average value in snr=-15 conditions
            # so that in each space all value reflect the absolute distance differences between all other conditions and the -15dB
            #### to the snr axis
            psth_proj_df_ttref = psth_proj_df_ttall[psth_proj_df_ttall['snr']==-15][['snr-shift']].mean().values
            psth_proj_df_tt10dB = psth_proj_df_ttall[psth_proj_df_ttall['snr']==10][['snr-shift']].mean().values
            if psth_proj_df_tt10dB>=psth_proj_df_ttref:
                psth_proj_df_ttall['snr-shift_bscrct'] = psth_proj_df_ttall['snr-shift'].values-psth_proj_df_ttref                           
            else:
                psth_proj_df_ttall['snr-shift_bscrct'] = psth_proj_df_ttref-psth_proj_df_ttall['snr-shift'].values
  
            ### to the modality axis
            psth_proj_df_ttA = psth_proj_df_ttall[psth_proj_df_ttall['mod']=='a'][['trialMod']].mean().values
            psth_proj_df_ttAV = psth_proj_df_ttall[psth_proj_df_ttall['mod']=='av'][['trialMod']].mean().values
            if psth_proj_df_ttAV>=psth_proj_df_ttA:
                psth_proj_df_ttall['trialMod_bscrct'] = psth_proj_df_ttall['trialMod'].values-psth_proj_df_ttA
            else:
                psth_proj_df_ttall['trialMod_bscrct'] = psth_proj_df_ttA-psth_proj_df_ttall['trialMod'].values

        psth_proj_df = pd.concat([psth_proj_df,psth_proj_df_ttall])
   
    return psth_proj_df 
# plot trajectories for each monkey
def plot2DTraj(psth_proj_df_BSall,namstrallSU):
    ################### plot projection on the av axis difference between a and av condition separatly for each SNR level, combine the same monkey's plot in one plot
    fig, axess = plt.subplots(2,1,figsize=(8,8),sharex='col',sharey='col')  
    psth_proj_df_BSall_av = psth_proj_df_BSall.copy() 
    psth_proj_df_BSall_av['snr'] = psth_proj_df_BSall_av['snr'].replace({-15:'difficult',-10:'difficult',-5:'medium',0:'medium',5:'easy',10:'easy'})
    psth_proj_df_BSall_mean = psth_proj_df_BSall_av.groupby(by=['Monkey','time','mod','resp','bs','shuffled','snr'])[['trialMod']].mean().reset_index()
    snrcolor = ['limegreen','orange','lightcoral'] # in the order of 'easy','medium','difficult'
    psth_proj_mod_a = psth_proj_df_BSall_mean[psth_proj_df_BSall_mean['mod']=='a'].sort_values(by=['Monkey','time','resp','bs','snr'])
    psth_proj_mod_av = psth_proj_df_BSall_mean[psth_proj_df_BSall_mean['mod']=='av'].sort_values(by=['Monkey','time','resp','bs','snr'])
    psth_proj_moddiff_snr = psth_proj_mod_a[['Monkey','time','resp','bs','snr','shuffled']].copy()
    psth_proj_moddiff_snr['trialModDiff'] = np.abs(psth_proj_mod_a['trialMod'].values-psth_proj_mod_av['trialMod'].values)

    for mm,monk in enumerate(list(psth_proj_moddiff_snr.Monkey.unique())):
        psth_proj_moddiff_mm = psth_proj_moddiff_snr[(psth_proj_moddiff_snr['Monkey']==monk)&(psth_proj_moddiff_snr['shuffled']=='true label')]

        font_properties = {'fontname': fontnameStr, 'fontsize': fontsizeNo-4}
        sns.lineplot(psth_proj_moddiff_mm,x='time',y='trialModDiff',hue='snr',hue_order=['easy','medium','difficult'],palette=snrcolor,ax=axess[mm],estimator='mean', errorbar=('ci', 95))                              
        axess[mm].set_xlabel('Time (s; relative to audiory target onset)',**font_properties)
        axess[mm].text(0.25, 0.9, 'monkey '+monk[0], 
                    transform=axess[mm].transAxes,  # Specify the position in axis coordinates
                    fontsize=fontsizeNo-4,             # Font size
                    fontname=fontnameStr,
                    verticalalignment='top', # Align text at the top
                    horizontalalignment='right') # Align text to the right
        axess[mm].legend(frameon=False, loc='upper right', bbox_to_anchor=(1, 1.03), title='SNR', fontsize=fontsizeNo-6, title_fontsize=fontsizeNo-4)
        axess[mm].plot([0,0],[0,2],linestyle='--',color='black')  
        axess[mm].plot([-0.52,-0.52],[0,1.5],linestyle='--',color='gray')         
        axess[mm].set_ylabel('A/AV separation',**font_properties)
        axess[mm].tick_params(axis='both', which='major', labelsize=fontsizeNo-4) 
    fig.savefig(figSavePath+'Popprjcton_ModalityAxis_A-AVdistanceBYsnr_'+namstrallSU+'_2monk_2swin.'+figformat)
    plt.close(fig)

    ################### plot projection on the snr axis
    fig, axess = plt.subplots(2,1,figsize=(8,8),sharex='col',sharey='col')  
    font_properties = {'fontname': fontnameStr, 'fontsize': fontsizeNo-4}
    ycolnames = 'snr-shift_bscrct' #'snr-shift','snr-shift_bscrct','snr-slope'
    snrcolor = ['limegreen','orange','lightcoral'] # in the order of 'easy','medium','difficult'
    psth_proj_df_BSall_temp = psth_proj_df_BSall.copy()
    psth_proj_df_BSall_temp['snr'] = psth_proj_df_BSall_temp['snr'].replace({-15:'difficult',-10:'difficult',-5:'medium',0:'medium',5:'easy',10:'easy'})
    psth_proj_df_BSsnr = psth_proj_df_BSall_temp.groupby(by=['Monkey','time','snr','bs','shuffled'])[ycolnames].mean().reset_index().reset_index(drop=True)

    for mm, monk in enumerate(list(psth_proj_df_BSsnr.Monkey.unique())):
        psth_proj_df_BSsnr_monk = psth_proj_df_BSsnr[(psth_proj_df_BSsnr['Monkey']==monk)&(psth_proj_df_BSsnr['shuffled']=='true label')].reset_index()
        sns.lineplot(psth_proj_df_BSsnr_monk,x='time',y=ycolnames,hue='snr',
                    hue_order=['easy','medium','difficult'],palette=snrcolor,
                    ax=axess[mm],estimator='mean', errorbar=('ci', 95)) 
        psth_proj_df_BSsnr_E_shuffled = psth_proj_df_BSsnr[(psth_proj_df_BSsnr['Monkey']==monk)&(psth_proj_df_BSsnr['shuffled']=='shuffled label')].groupby(by=['time','bs'])[['snr-shift_bscrct']].mean().reset_index()
        sns.lineplot(psth_proj_df_BSsnr_E_shuffled,x='time',y=ycolnames,color='gray',
                    ax=axess[mm],estimator='mean', errorbar=('ci', 95))                    
        # Add a manual legend entry for the gray shuffled plot
        handles, labels = axess[mm].get_legend_handles_labels()
        handles.append(plt.Line2D([0], [0], color='gray', label='Shuffled'))
        labels.append('shuffled')
        # Add the combined legend to the plot
        axess[mm].set_xlabel('Time (s; relative to audiory target onset)',**font_properties)
        axess[mm].legend(handles=handles, labels=labels,frameon=False, loc='upper right', bbox_to_anchor=(1, 1.03), title='SNR', fontsize=fontsizeNo-6, title_fontsize=fontsizeNo-4)
        axess[mm].plot([0,0],[-2,8],linestyle='--',color='black')  
        axess[mm].text(0.25, 0.9, 'monkey '+monk[0], 
                    transform=axess[mm].transAxes,  # Specify the position in axis coordinates
                    fontsize=fontsizeNo-4,             # Font size
                    fontname=fontnameStr,
                    verticalalignment='top', # Align text at the top
                    horizontalalignment='right') # Align text to the right                
        axess[mm].set_ylabel('Distance to -15 dB',**font_properties)
        axess[mm].tick_params(axis='both', which='major', labelsize=fontsizeNo-4) 
    fig.savefig(figSavePath+'Popprjcton_SNRAxis_'+namstrallSU+'_2swin.'+figformat)
    plt.close(fig)     

# debug in local pc
MonkeyDate_all = {'Elay':['230420','230616']}
Pathway='/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/PSTHdataframe/'
AVmodPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/AVmodIndex/'
figSavePath = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Figures/PSTHtraj/'
ResPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/PSTHtraj/'
STRFexcelPath = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/STRF/'
wavformPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/wavformStruct/'

# # run in monty
# MonkeyDate_all = getMonkeyDate_all()
# Pathway = '/home/huaizhen/Documents/MonkeyAVproj/data/PSTHdataframe/'
# ResPathway = '/home/huaizhen/Documents/MonkeyAVproj/data/Fitresults/PSTHtraj/'
# figSavePath = '/home/huaizhen/Documents/MonkeyAVproj/data/Figures/PSTHtraj/'
# AVmodPathway = '/home/huaizhen/Documents/MonkeyAVproj/data/Fitresults/AVmodIndex/'
# STRFexcelPath = '/home/huaizhen/Documents/MonkeyAVproj/data/Fitresults/STRF/'
# wavformPathway = '/home/huaizhen/Documents/MonkeyAVproj/data/Fitresults/wavformStruct/'

#################################bootstrap parameters
balanceSamples = True # whether balance samples across conditions by adding bootstraped samples to each condition
popSampTimes =5
resolutionred = 5 # reduce temporal resolotion of the decoding by this scale, original temporal resolution:0.01
# 2 task related variables
psthGroupCols = ['trialMod','snr-shift']
filterdict = {'trialMod':['a','av'],'respLabel':['hit'],'AVoffset':[90,120]}
linregCols = ['snr-shift','trialMod']
#tim for each datapoint in psth, 0 is cooonset
timwinStart = [-1,0.5]#align2coo [-0.9,1] align2js [-1.2,0] align2DT [-1,0.5]
bin = 50/1000 
fontsizeNo = 20# 38 for tranjectory plot
fontnameStr = 'Arial'#'DejaVu Sans'
figformat = 'svg'
################################

start_time = time.monotonic()
if __name__ == '__main__':
    for alignKeys in ['_align2coo']: # '_align2coo' '_align2DT' '_align2js'
        for STRFstr in ['notsig','sig',]:#['all','notsig','sig']:   
            namstrallSU = alignKeys+'_Hit=A+AV_ttestfilteredUnits_ProjonModXsnr_STRF'+ STRFstr+'_spkwavShape-all_nonoverlaptimwin50msbinRawPSTH'+'_AVoffset90+120'+'_sample30noreplacement_lessWu-sess'       
            psth_proj_df_BSall = pd.DataFrame()
            for bs in range(popSampTimes):
                for Monkey,Date in MonkeyDate_all.items():    
                    # randomly pick 30 usable strf/nstrf nuerons from the current monkey
                    df_avMod_all_sig = pickneuronpopulations(Monkey,STRFstr)
                    # concatenate spike time series data of all selected neurons, sample equal number of trials for all conditions 
                    AllSU_psth_trialBYtrial,AllSU_psth_condAve,psthCol,binedge = sampTrials(Monkey,Date,alignKeys,df_avMod_all_sig)
                    # conduct targeted dimension reduction for the selected neural populations                                                   
                    psth_proj,condstrlist = conductTargDimRed(AllSU_psth_trialBYtrial,AllSU_psth_condAve,psthCol) #psth_proj: linregCond X time X condition
                    # add centered projection on each axis and at each time points separately, form psth_proj dataframe for meaningful avergaing across separate projections
                    psth_proj_df = formdf(psth_proj,linregCols,condstrlist,binedge,bsref='on')
                    psth_proj_df['shuffled'] = 'true label'
                    psth_proj_df_shuffled = formdf(psth_proj,linregCols,list(np.random.choice(condstrlist,size=len(condstrlist),replace=False)),binedge,bsref='on')
                    psth_proj_df_shuffled['shuffled'] = 'shuffled label'
                    psth_proj_df_2comb = pd.concat([psth_proj_df,psth_proj_df_shuffled])
                    psth_proj_df_2comb['bs'] = bs
                    psth_proj_df_2comb['Monkey'] = Monkey
                    psth_proj_df_BSall = pd.concat([psth_proj_df_BSall,psth_proj_df_2comb]) 
            pickle.dump(psth_proj_df_BSall,open(ResPathway+'twoMonk_bs_2swin'+namstrallSU+'.pkl','wb'))                               
            psth_proj_df_BSall = pickle.load(open(ResPathway+'twoMonk_bs_2swin'+namstrallSU+'.pkl','rb'))  # psth_proj_df_BSall rows=Nneurons X clsSamp X trials X bootstrapTimes
            # plot projections on snr axis and condition axis for each monkey separately
            plot2DTraj(psth_proj_df_BSall,namstrallSU)
                            
times2 = time.monotonic()
print('total time spend for 2 monkeys:')
print(timedelta(seconds= times2- start_time)) 


