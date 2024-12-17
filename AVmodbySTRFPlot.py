import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from itertools import chain
from statsmodels.stats.anova import AnovaRM
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from sharedparam import getMonkeyDate_all,neuronfilterDF
from spikeUtilities import estiAVmodindex
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.formula.api import mixedlm
from statsmodels.stats.anova import AnovaRM
import statsmodels.formula.api as smf
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerLine2D

def estiAVmodindex(snra,fra_raw,fra_zs,fca,snrav,frav_raw,frav_zs,fcav):
    #fra and frav should save at the same snr order
    modind_df = pd.DataFrame()
    for ii,snr in enumerate(snra):
        modind_df = pd.concat((modind_df,pd.DataFrame({'snr':[snr]*2,
                                                       'AVmodInd_fromraw':[(frav_raw[ii]-fra_raw[ii])/(frav_raw[ii]+fra_raw[ii])]*2,
                                                       'AVmodInd_fromzs':[(frav_zs[ii]-fra_zs[ii])/(frav_zs[ii]+fra_zs[ii])]*2,
                                                       'mod':['a','av'],
                                                       'raw_fr':[fra_raw[ii],frav_raw[ii]],
                                                       'zscore_fr':[fra_zs[ii],frav_zs[ii]],
                                                       'FractionCorrect':[fca[ii],fcav[ii]]})))
    return modind_df


# MonkeyDate_all = {'Elay':['230620'],'Wu':['240607']} #
# ResPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/neuralMetric/'
# figSavePath = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Figures/AVmodIndex/'
# AVmodPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/AVmodIndex/'
# behaveResPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/BehavPerform/'
# STRFexcelPath = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/STRF/'


MonkeyDate_all = getMonkeyDate_all()
ResPathway = '/home/huaizhen/Documents/MonkeyAVproj/data/Fitresults/neuralMetric/'
figSavePath = '/home/huaizhen/Documents/MonkeyAVproj/data/Figures/AVmodIndex/'
AVmodPathway = '/home/huaizhen/Documents/MonkeyAVproj/data/Fitresults/AVmodIndex/'
STRFexcelPath = '/home/huaizhen/Documents/MonkeyAVproj/data/Fitresults/STRF/'

bintim = 300#200
filterules = 'ttest' # 'ttest' 'glm' 'all'
noiseCR = 'spkRateCRa2' # only matters when noiseStr='aCRnoise' or 'aCRvCRnoise', has to be one of the string in CRcolnames
signalStrlist = ['hit'] #['hit'] ['hit','miss'] ---signal method in NM
noiseStr='aCRnoise' #'aCRnoise' 'aCRvCRnoise' 'aMISSnoise' ---noise method in NM
bintimstr = '_'+str(bintim)+'msbinRawPSTH'

DVstr = 'FractionCorrect' # 'FractionCorrect','dprime'
figformat = 'svg'
fontsizeNo = 20
fontnameStr = 'DejaVu Sans'#'DejaVu Sans'
AVmodIndVar = 'AVmodInd_fromraw' #'AVmodInd_fromraw' 'AVmodInd_fromzs'
font_properties = {'fontname': fontnameStr, 'fontsize': fontsizeNo}
for extrastr in ['align2coo']:
    pklnamestr = ('+').join(signalStrlist)+'_'+noiseStr+bintimstr+'_allcls_'+extrastr 

    for strfunits in ['all']:#['sig','notsig','all']:   
        namstr = pklnamestr+'_STRF'+strfunits+'_'+filterules+'_'+extrastr
        AVmodindex = pd.DataFrame()

        for mm,(Monkey,Date) in enumerate(MonkeyDate_all.items()):           
            # filter neurons based on different rules
            df_avMod_all_sig,_ = neuronfilterDF(AVmodPathway,STRFexcelPath,filterules,Monkey,STRF=strfunits)        
            for session in Date:  
                # print(Monkey+session+' in plotting....')   
                # get cluster has significant AV modulation
                df_avMod_sess = df_avMod_all_sig[(df_avMod_all_sig['Monkey']==Monkey) & (df_avMod_all_sig['session']==session)]  
                # print(df_avMod_sess.to_string()) 
                #plot neurometric fun 
                data = pickle.load(open(ResPathway+Monkey+'_'+session+'_neuroMetricFit'+pklnamestr+'.pkl','rb'))

                for cc,sesscls in enumerate(df_avMod_sess['session_cls'].unique().tolist()):
                    df_avMod_sess_temp = df_avMod_sess[df_avMod_sess['session_cls']==sesscls]
                    cls = sesscls[sesscls.index('_')+1:]
                    # print(cls)
                    data_temp = data[cls].copy()
                    if len(data_temp)>0:
                        # raw fr average:y_rawfr2_a;  zcrored fr average: y_rawfr_a
                        AVmodindex_temp = estiAVmodindex(data_temp['x_raw_a'],data_temp['y_rawfr2_a'],data_temp['y_rawfr_a'],data_temp['y_raw_a'],data_temp['x_raw_av'],data_temp['y_rawfr2_av'],data_temp['y_rawfr_av'],data_temp['y_raw_av'])
                        mod = "+".join(sorted(df_avMod_sess_temp['iv'].values))                           
                        AVmodindex_temp['Umod'] = mod  
                        AVmodindex_temp['Monkey'] = Monkey
                        AVmodindex_temp['session_cls'] = sesscls
                        AVmodindex_temp['session'] = session
                        # print(df_avMod_sess_temp.to_string())
                        AVmodindex_temp['STRFsig'] = df_avMod_sess_temp['STRFsig'].unique()[0]
                        AVmodindex = pd.concat((AVmodindex,AVmodindex_temp))  
        
        # relabel snr for later grouping process
        AVmodindex_tt = AVmodindex.copy()
        print(AVmodindex_tt)
        AVmodindex_tt['snr']=AVmodindex_tt['snr'].replace({-15:'difficult',-10:'difficult',-5:'medium',0:'medium',5:'easy',10:'easy'})

        ##############group average measurements
        # average across all snrs
        AVmodindex_ave = AVmodindex.groupby(by=['session_cls','mod','Monkey','STRFsig'])[['FractionCorrect','zscore_fr','raw_fr','AVmodInd_fromraw','AVmodInd_fromzs']].mean().reset_index()
        #average within snr group
        AVmodindex_tt_ave = AVmodindex_tt.groupby(by=['session_cls','mod','snr','Monkey','STRFsig'])[['FractionCorrect','zscore_fr','raw_fr','AVmodInd_fromraw','AVmodInd_fromzs']].mean().reset_index()

        # fig, axess = plt.subplots(2,1,figsize=(4,7)) 
        snrorder=['easy','medium','difficult']
        colors = ['limegreen','orange','lightcoral']

        ##################### plot av modindex in different STRF neurons
        if strfunits=='all':            
            AVmodindex_tt_ave_A = AVmodindex_tt_ave[AVmodindex_tt_ave['mod']=='a']
            AVmodindex_tt_ave_A['AVmodInd_fromraw']=AVmodindex_tt_ave_A['AVmodInd_fromraw'].abs()
            # Define the model: DV ~ IV1 * IV2 (this also includes the interaction term between IV1 and IV2)
            model = smf.mixedlm('AVmodInd_fromraw ~ snr*STRFsig', 
                    data=AVmodindex_tt_ave_A, 
                    groups=AVmodindex_tt_ave_A['session_cls'],
                    re_formula='~STRFsig')
            # Fit the model
            result = model.fit()
            # Print the results
            print(result.summary()) 

            print('modindex strf: median--'+str(np.median(AVmodindex_tt_ave_A[AVmodindex_tt_ave_A['STRFsig']==1].AVmodInd_fromraw))
                                +'\n    [25 75]percentile--['+str(np.percentile(AVmodindex_tt_ave_A[AVmodindex_tt_ave_A['STRFsig']==1].AVmodInd_fromraw,25))+'   '
                                                             +str(np.percentile(AVmodindex_tt_ave_A[AVmodindex_tt_ave_A['STRFsig']==1].AVmodInd_fromraw,75))+']') 
            print('modindex nstrf: median--'+str(np.median(AVmodindex_tt_ave_A[AVmodindex_tt_ave_A['STRFsig']==0].AVmodInd_fromraw))
                                +'\n    [25 75]percentile--['+str(np.percentile(AVmodindex_tt_ave_A[AVmodindex_tt_ave_A['STRFsig']==0].AVmodInd_fromraw,25))+'   '
                                                            +str(np.percentile(AVmodindex_tt_ave_A[AVmodindex_tt_ave_A['STRFsig']==0].AVmodInd_fromraw,75))+']')  
            
            print('difficulty snr')
            print('modindex strf: median--'+str(np.median(AVmodindex_tt_ave_A[(AVmodindex_tt_ave_A['STRFsig']==1)&(AVmodindex_tt_ave_A['snr']=='difficult')].AVmodInd_fromraw))
                                +'\n    [25 75]percentile--['+str(np.percentile(AVmodindex_tt_ave_A[(AVmodindex_tt_ave_A['STRFsig']==1)&(AVmodindex_tt_ave_A['snr']=='difficult')].AVmodInd_fromraw,25))+'   '
                                                             +str(np.percentile(AVmodindex_tt_ave_A[(AVmodindex_tt_ave_A['STRFsig']==1)&(AVmodindex_tt_ave_A['snr']=='difficult')].AVmodInd_fromraw,75))+']') 
            print('modindex nstrf: median--'+str(np.median(AVmodindex_tt_ave_A[(AVmodindex_tt_ave_A['STRFsig']==0)&(AVmodindex_tt_ave_A['snr']=='difficult')].AVmodInd_fromraw))
                                +'\n    [25 75]percentile--['+str(np.percentile(AVmodindex_tt_ave_A[(AVmodindex_tt_ave_A['STRFsig']==0)&(AVmodindex_tt_ave_A['snr']=='difficult')].AVmodInd_fromraw,25))+'   '
                                                             +str(np.percentile(AVmodindex_tt_ave_A[(AVmodindex_tt_ave_A['STRFsig']==0)&(AVmodindex_tt_ave_A['snr']=='difficult')].AVmodInd_fromraw,75))+']') 

            print('medium snr')
            print('modindex strf: median--'+str(np.median(AVmodindex_tt_ave_A[(AVmodindex_tt_ave_A['STRFsig']==1)&(AVmodindex_tt_ave_A['snr']=='medium')].AVmodInd_fromraw))
                                +'\n    [25 75]percentile--['+str(np.percentile(AVmodindex_tt_ave_A[(AVmodindex_tt_ave_A['STRFsig']==1)&(AVmodindex_tt_ave_A['snr']=='medium')].AVmodInd_fromraw,25))+'   '
                                                             +str(np.percentile(AVmodindex_tt_ave_A[(AVmodindex_tt_ave_A['STRFsig']==1)&(AVmodindex_tt_ave_A['snr']=='medium')].AVmodInd_fromraw,75))+']') 
            print('modindex nstrf: median--'+str(np.median(AVmodindex_tt_ave_A[(AVmodindex_tt_ave_A['STRFsig']==0)&(AVmodindex_tt_ave_A['snr']=='medium')].AVmodInd_fromraw))
                                +'\n    [25 75]percentile--['+str(np.percentile(AVmodindex_tt_ave_A[(AVmodindex_tt_ave_A['STRFsig']==0)&(AVmodindex_tt_ave_A['snr']=='medium')].AVmodInd_fromraw,25))+'   '
                                                             +str(np.percentile(AVmodindex_tt_ave_A[(AVmodindex_tt_ave_A['STRFsig']==0)&(AVmodindex_tt_ave_A['snr']=='medium')].AVmodInd_fromraw,75))+']') 

            print('easy snr')
            print('modindex strf:  median--'+str(np.median(AVmodindex_tt_ave_A[(AVmodindex_tt_ave_A['STRFsig']==1)&(AVmodindex_tt_ave_A['snr']=='easy')].AVmodInd_fromraw))
                                +'\n    [25 75]percentile--['+str(np.percentile(AVmodindex_tt_ave_A[(AVmodindex_tt_ave_A['STRFsig']==1)&(AVmodindex_tt_ave_A['snr']=='easy')].AVmodInd_fromraw,25))+'   '
                                                             +str(np.percentile(AVmodindex_tt_ave_A[(AVmodindex_tt_ave_A['STRFsig']==1)&(AVmodindex_tt_ave_A['snr']=='easy')].AVmodInd_fromraw,75))+']') 
            print('modindex nstrf: median--'+str(np.median(AVmodindex_tt_ave_A[(AVmodindex_tt_ave_A['STRFsig']==0)&(AVmodindex_tt_ave_A['snr']=='easy')].AVmodInd_fromraw))
                                +'\n    [25 75]percentile--['+str(np.percentile(AVmodindex_tt_ave_A[(AVmodindex_tt_ave_A['STRFsig']==0)&(AVmodindex_tt_ave_A['snr']=='easy')].AVmodInd_fromraw,25))+'   '
                                                             +str(np.percentile(AVmodindex_tt_ave_A[(AVmodindex_tt_ave_A['STRFsig']==0)&(AVmodindex_tt_ave_A['snr']=='easy')].AVmodInd_fromraw,75))+']') 

            ### plot diff at each snr
            fig, axess = plt.subplots(1,1,figsize=(10,6))
            sns.barplot(data=AVmodindex_tt_ave_A.reset_index(drop=True), x='snr', y='AVmodInd_fromraw', hue='STRFsig',
                        palette={0:'darkgray', 1:'dimgray'}, 
                        errorbar=('ci',95),capsize=0.1,errwidth=1, ax=axess)
            axess.set_ylabel('Neural Modulation Index',**font_properties)
            axess.set_xlabel('SNR',**font_properties)
            axess.tick_params(axis='both', which='major', labelsize=fontsizeNo-8)        
            new_labels = ['non-significant', 'significant']
            handles, labels = axess.get_legend_handles_labels()
            axess.legend(handles=handles[:2], labels=new_labels, loc='upper right',
                        handler_map={mlines.Line2D: HandlerLine2D(numpoints=1)},frameon=False,
                        bbox_to_anchor=(1.01, 1), fontsize=fontsizeNo-8,title='STRF',title_fontsize=fontsizeNo-6) 
            fig.tight_layout()
            fig.savefig(figSavePath+'AVmodIndbySTRF_Cmp'+namstr+'.'+figformat, dpi=600)
            plt.close(fig)          

       

print('end')







