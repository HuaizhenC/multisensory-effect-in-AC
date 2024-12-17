import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# need run under psychopy environment
ResPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/glmfit/'
figSavePath = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Figures/glmfit/'

df = pd.read_pickle(ResPathway+'glmfitresDF_align2Coo.pkl')
df['clusterID'] = df['cluster'].factorize(sort=True)[0].copy()  
df2 = df.sort_values(['Monkey','session','clusterID','startTim'],kind='mergesort') 

for monkey in df2.Monkey.unique():
    df2_monk = df2[df2['Monkey']==monkey]
    for sess in df2_monk.session.unique():
        df2_monk_sess = df2_monk[df2_monk['session']==sess]
        df2_monk_sess_cls_all = pd.DataFrame()
        for cls in df2_monk_sess.clusterID.unique():
            df2_monk_sess_cls = df2_monk_sess[df2_monk_sess['clusterID']==cls]
            df2_monk_sess_cls_temp = pd.concat([df2_monk_sess_cls,pd.DataFrame(df2_monk_sess_cls['pval_snr'].copy().apply(lambda x: 1 if x*len(df2_monk_sess_cls.startTim.unique())<0.05 else np.nan)).rename(columns={'pval_snr':'pval_snrSIG'})],axis=1) 
            df2_monk_sess_cls_temp = pd.concat([df2_monk_sess_cls_temp,pd.DataFrame(df2_monk_sess_cls['pval_trialMod'].copy().apply(lambda x: 1 if x*len(df2_monk_sess_cls.startTim.unique())<0.05 else np.nan)).rename(columns={'pval_trialMod':'pval_trialModSIG'})],axis=1) 
            df2_monk_sess_cls_temp['coef_snr'] = df2_monk_sess_cls_temp['coef_snr'].values*df2_monk_sess_cls_temp['pval_snrSIG'].values
            df2_monk_sess_cls_temp['coef_trialMod'] = df2_monk_sess_cls_temp['coef_trialMod'].values*df2_monk_sess_cls_temp['pval_trialModSIG'].values
            df2_monk_sess_cls_all = pd.concat([df2_monk_sess_cls_all,df2_monk_sess_cls_temp]) 
        fig, axes = plt.subplots(2,1, figsize=(14,8)) 
        sns.scatterplot(df2_monk_sess_cls_all,x='startTim',y='coef_snr',hue='cluster',style='cluster',ax=axes[0])
        axes[0].legend(frameon=False,loc='upper right',bbox_to_anchor=(1.1, 1), borderaxespad=0.,fontsize=7)
        sns.scatterplot(df2_monk_sess_cls_all,x='startTim',y='coef_trialMod',hue='cluster',style='cluster',ax=axes[1])
        axes[1].legend(frameon=False,loc='upper right',bbox_to_anchor=(1.1, 1), borderaxespad=0.,fontsize=7)
        plt.tight_layout()
        plt.savefig(figSavePath+monkey+sess+'_glmCoeffSigbyTime.png')

            
print('ff')