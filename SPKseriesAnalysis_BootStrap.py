import time
from datetime import timedelta
import numpy as np
import seaborn as sns
import pickle
import pandas as pd
from matplotlib import pyplot as plt 
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from spikeUtilities import decodrespLabel2str,SortfilterDF,sampBalanceGLM
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from multiprocessing import Pool
from scipy.spatial.distance import pdist, squareform
from scipy.signal import correlate
from sharedparam import getMonkeyDate_all,neuronfilterDF
import random
from spikeUtilities import glmfit
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerLine2D
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
from scipy.stats import friedmanchisquare


# from statsmodels.tsa.stattools import grangercausalitytests

CPUs = 10

###########Targeted	dimensionality reduction in this paper: https://www.nature.com/articles/nature12742

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

# get denoise matrix from selected pca components
def getdenoiseMat(pcomponents):
    denoiseMat = np.zeros((pcomponents.shape[1],pcomponents.shape[1]))
    for pcnum in range(pcomponents.shape[0]):
        denoiseMat = denoiseMat + pcomponents[pcnum,:].reshape(-1,1)*pcomponents[pcnum,:].reshape(1,-1)
    return denoiseMat
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
            try :
                coeff_temp,pval_temp,evalparam = glmfit(psth_proj_df_ttall,['snr-shift'],['snr'],'gaussian') 
            except RuntimeWarning:
                coeff_temp = pd.DataFrame([[np.nan]*2],columns=['coef_snr'])
                pval_temp = pd.DataFrame([[np.nan]*2],columns=['pval_snr'])
            psth_proj_df_ttall['snr-slope'] = coeff_temp['coef_snr'].values[0]  
            psth_proj_df_ttall['snr-slope-Pval'] = pval_temp['pval_snr'].values[0]   
            ### to the modality axis
            psth_proj_df_ttA = psth_proj_df_ttall[psth_proj_df_ttall['mod']=='a'][['trialMod']].mean().values
            psth_proj_df_ttAV = psth_proj_df_ttall[psth_proj_df_ttall['mod']=='av'][['trialMod']].mean().values
            if psth_proj_df_ttAV>=psth_proj_df_ttA:
                psth_proj_df_ttall['trialMod_bscrct'] = psth_proj_df_ttall['trialMod'].values-psth_proj_df_ttA
            else:
                psth_proj_df_ttall['trialMod_bscrct'] = psth_proj_df_ttA-psth_proj_df_ttall['trialMod'].values
            try :
                coeff_temp,pval_temp,evalparam = glmfit(psth_proj_df_ttall,['trialMod'],['mod'],'gaussian') 
            except RuntimeWarning:
                coeff_temp = pd.DataFrame([[np.nan]*2],columns=['coef_mod'])
                pval_temp = pd.DataFrame([[np.nan]*2],columns=['pval_mod'])
            psth_proj_df_ttall['mod-slope'] = coeff_temp['coef_mod'].values[0]  
            psth_proj_df_ttall['mod-slope-Pval'] = pval_temp['pval_mod'].values[0]   

        psth_proj_df = pd.concat([psth_proj_df,psth_proj_df_ttall])
   
    return psth_proj_df 
#plot 3d animation 
def plotTraj(psth_proj_df,linregCols,alignKeys,htmlfilename):
   
    psth_proj_df = psth_proj_df.sort_values(by=['time','mod','snr','resp'])
    # psth_proj_df['category'] = psth_proj_df['mod']+ ' ' + psth_proj_df['snr'].astype(str)+' '+ psth_proj_df['resp']
    psth_proj_df['category'] = psth_proj_df['mod']
    # psth_proj_df['category'] = psth_proj_df['snr'].astype(str) 

    times= sorted(psth_proj_df['time'].unique())
    categories = psth_proj_df['category'].unique()
    cooOntime = times[np.argmin(np.abs(np.array(times)))]
    vidOntime = times[np.argmin(np.abs(np.array(times)+636/1220.703125))]    
    # Create a figure
    fig = go.Figure()
    if len(linregCols)==3:
        # Create a trace for each category
        for category in categories:
            category_df = psth_proj_df[psth_proj_df['category'] == category]
            fig.add_trace(go.Scatter3d(
                x=[category_df.iloc[0][linregCols[0]]],
                y=[category_df.iloc[0][linregCols[1]]],
                z=[category_df.iloc[0][linregCols[2]]],
                mode='lines+markers',
                line=dict(width=5, color=category_df.iloc[0]['color'], dash=category_df.iloc[0]['line']),
                marker=dict(size=0,color='white', symbol='circle',opacity=0),
                name=category
            ))

        # Creating frames for the animation
        frames = []
        for time in times:
            frame_data = []
            for category in categories:
                category_df = psth_proj_df[(psth_proj_df['category'] == category) & (psth_proj_df['time'] <= time)]
                if alignKeys=='align2coo':
                    near_zero_indices = category_df['time'].apply(lambda x: 1 if np.abs(x-vidOntime)<0.001 else 2 if np.abs(x-cooOntime)<0.001 else 0)
                else:
                    near_zero_indices = category_df['time'].apply(lambda x: 2 if np.abs(x-cooOntime)<0.001 else 0)

                marker_sizes= near_zero_indices.apply(lambda x: 10 if x>0 else 0).tolist()
                if category_df['mod'].apply(lambda x:1 if 'v' in x else 0).all():
                    marker_colors= near_zero_indices.apply(lambda x: 'black' if x==1 else 'red' if x==2 else 'white').tolist()
                else:
                    marker_colors= near_zero_indices.apply(lambda x: 'dimgrey' if x==1 else 'goldenrod' if x==2 else 'white').tolist()

                frame_data.append(go.Scatter3d(
                    x=category_df[linregCols[0]],
                    y=category_df[linregCols[1]],
                    z=category_df[linregCols[2]],
                    mode='lines+markers',
                    line=dict(width=category_df.iloc[0]['width'], color=category_df.iloc[0]['color'], dash=category_df.iloc[0]['line']),
                    marker=dict(size=marker_sizes, color=marker_colors,opacity=1),
                    name=category
                ))                              
            frames.append(go.Frame(data=frame_data, name=str(time)))
        fig.frames = frames
        # Adding buttons to control the animation
        xmin, xmax = psth_proj_df[linregCols[0]].values.min(),psth_proj_df[linregCols[0]].values.max()
        ymin, ymax = psth_proj_df[linregCols[1]].values.min(),psth_proj_df[linregCols[1]].values.max()
        zmin, zmax = psth_proj_df[linregCols[2]].values.min(),psth_proj_df[linregCols[2]].values.max()   
        # Add a slider and buttons for animation control
        sliders = [{
            'steps': [
                {
                    'method': 'animate',
                    'label': str(time),
                    'args': [[str(time)], {'frame': {'duration': 500, 'redraw': True}, 'mode': 'immediate'}]
                } for time in times
            ]
        }]    
        fig.update_layout(
            sliders=sliders,
            scene=dict(
                        xaxis=dict(range=[xmin, xmax]),  # Replace xmin and xmax with your desired values
                        yaxis=dict(range=[ymin, ymax]),  # Replace ymin and ymax with your desired values
                        zaxis=dict(range=[zmin, zmax]),   # Replace zmin and zmax with your desired values
                        xaxis_title='SNR',
                        yaxis_title='A/AV',
                        zaxis_title='Hit/Miss'
                    ),
            updatemenus=[{
                "buttons": [
                    {"args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate"},
                    {"args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"}],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }]
        )

    if len(linregCols)==2:
        # Create a trace for each category
        for category in categories:
            category_df = psth_proj_df[psth_proj_df['category'] == category]
            fig.add_trace(go.Scatter(
                x=[category_df.iloc[0][linregCols[0]]],
                y=[category_df.iloc[0][linregCols[1]]],
                mode='lines+markers',
                line=dict(width=2, color=category_df.iloc[0]['color'], dash=category_df.iloc[0]['line']),
                marker=dict(size=0,color='white', symbol='triangle-right',opacity=1),#modify symbol to add time direction in trajectories
                name=category
            ))

        # Creating frames for the animation
        frames = []
        for time in times:
            frame_data = []
            for category in categories:
                category_df = psth_proj_df[(psth_proj_df['category'] == category) & (psth_proj_df['time'] <= time)]
                if alignKeys=='align2coo':
                    near_zero_indices = category_df['time'].apply(lambda x: 1 if np.abs(x-vidOntime)<0.001 else 2 if np.abs(x-cooOntime)<0.001 else 0)
                else:
                    near_zero_indices = category_df['time'].apply(lambda x: 2 if np.abs(x-cooOntime)<0.001 else 0)

                marker_sizes= near_zero_indices.apply(lambda x: 30 if x>0 else 0).tolist()
                if category_df['mod'].apply(lambda x:1 if 'v' in x else 0).all():
                    marker_colors= near_zero_indices.apply(lambda x: 'black' if x==1 else 'red' if x==2 else 'white').tolist()
                else:
                    marker_colors= near_zero_indices.apply(lambda x: 'dimgrey' if x==1 else 'goldenrod' if x==2 else 'white').tolist()

                frame_data.append(go.Scatter(
                    x=category_df[linregCols[0]],
                    y=category_df[linregCols[1]],
                    mode='lines+markers',
                    line=dict(width=category_df.iloc[0]['width'], color=category_df.iloc[0]['color'], dash=category_df.iloc[0]['line']),
                    marker=dict(size=marker_sizes, color=category_df.iloc[0]['color'],opacity=1),
                    name=category
                ))                              
            frames.append(go.Frame(data=frame_data, name=str(time)))
        fig.frames = frames
        # Adding buttons to control the animation
        xmin, xmax = psth_proj_df[linregCols[0]].values.min()-0.5,psth_proj_df[linregCols[0]].values.max()+0.5
        # ymin, ymax = psth_proj_df[linregCols[1]].values.min()-0.5,psth_proj_df[linregCols[1]].values.max()+0.5
        ymin, ymax = -2,2
        # Add a slider and buttons for animation control
        sliders = [{'steps': [
                {'method': 'animate',
                'label': str(time),
                'args': [[str(time)], {'frame': {'duration': 500, 'redraw': True}, 'mode': 'immediate'}]} 
                for time in times],
                'x': 0,  # Adjust horizontal position if needed
                'y': -0.5,  # Lower position (reduce this value to move the slider down)                
        }]    
        fig.update_layout(
            sliders=sliders,            
            xaxis=dict(title='SNR',range=[xmin, xmax], 
                       linecolor='black',linewidth=4,showline=True,
                       title_font={"size": fontsizeNo,"family":fontnameStr},tickfont={"size": fontsizeNo-2,"family":fontnameStr}),                      
            yaxis=dict(title='Modality',range=[ymin, ymax], 
                       linecolor='black',linewidth=4,showline=True,
                       title_font={"size": fontsizeNo,"family":fontnameStr},tickfont={"size": fontsizeNo-2,"family":fontnameStr},
                       tickvals=[-4,-2,0,2,4],ticktext=['-4','-2','0','2','4']),
            legend=dict(font=dict(family=fontnameStr,size=fontsizeNo-3),
                        x = 0.96,y=1),
            plot_bgcolor='white',
            updatemenus=[{
                "buttons": [
                    {"args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate"},
                    {"args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"}],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": -0.4,
                "yanchor": "top"
            }]
        )

    # Save the plot as an interactive HTML file
    fig.write_html(htmlfilename+'.html')

# plot raw psth
def plotpsth(AllSU_psth_trialBYtrial,binedge,grpbylist,psthCol,pngfilename):
    AllSU_psth_df_condAve = pd.DataFrame()
    # AllSU_psth_trialBYtrial = sampBalanceGLM(AllSU_psth_trialBYtrial,grpbylist)
    AllSU_psth_condAve = AllSU_psth_trialBYtrial.groupby(grpbylist).mean(numeric_only=True)[psthCol].reset_index()
    for cat,grp in AllSU_psth_condAve.groupby(grpbylist):
        AllSU_psth_df_temp = pd.DataFrame()
        AllSU_psth_df_temp['frate'] = grp[psthCol].values.flatten()
        AllSU_psth_df_temp['time'] = binedge
        for cc,col in enumerate(grpbylist):
            AllSU_psth_df_temp[col] = cat[cc]
        AllSU_psth_df_condAve = pd.concat([AllSU_psth_df_condAve,AllSU_psth_df_temp])

    if len(AllSU_psth_df_condAve['respLabel'].unique())==1: # when only hit trials are analyzed
        fig, axess = plt.subplots(1,1,figsize=(10,4),sharex='col') # 
        sns.lineplot(AllSU_psth_df_condAve.reset_index(),x='time',y='frate',style='trialMod',hue='snr',
                    palette=sns.color_palette('flare',n_colors=6,as_cmap=True),
                    ax=axess,estimator='mean', errorbar=('ci', 95))
        axess.legend(frameon=False, framealpha=0,loc='upper right',bbox_to_anchor=(1.12, 1.), borderaxespad=0.)
    else:
        fig, axess = plt.subplots(2,1,figsize=(10,8),sharex='col') # when hit and miss trials are analyzed separately 
        sns.lineplot(AllSU_psth_df_condAve[AllSU_psth_df_condAve['respLabel']=='hit'].reset_index(),x='time',y='frate',style='trialMod',hue='snr',
                    palette=sns.color_palette('flare',n_colors=6,as_cmap=True),
                    ax=axess[0],estimator='mean', errorbar=('ci', 95)) 
        sns.lineplot(AllSU_psth_df_condAve[AllSU_psth_df_condAve['respLabel']=='miss'].reset_index(),x='time',y='frate',style='trialMod',hue='snr',
                    palette=sns.color_palette('flare',n_colors=6,as_cmap=True),
                    ax=axess[1],estimator='mean', errorbar=('ci', 95))  
        axess[0].set_title('hit')    
        axess[1].set_title('miss')  
        axess[0].legend(frameon=False, framealpha=0,loc='upper right',bbox_to_anchor=(1.12, 1.), borderaxespad=0.)
        axess[1].legend(frameon=False, framealpha=0,loc='upper right',bbox_to_anchor=(1.12, 1.), borderaxespad=0.)
    fig.tight_layout()
    fig.savefig(pngfilename+'.png')
    plt.close(fig)        

    return AllSU_psth_df_condAve

def getpairDist(tt,tcol,condstrdf,AllSU_psth_condAve,AllSU_psth_trialBYtrial):
    DisMat_temp = np.zeros((condstrdf.shape[0],condstrdf.shape[0]))
    for i,row in condstrdf.iterrows():
        frate_tempI = AllSU_psth_condAve[(AllSU_psth_condAve['trialMod']==row['trialMod'])&
                                                            (AllSU_psth_condAve['snr']==row['snr'])].sort_values(by=['sess_cls'])[tcol].values 
        frate_tempI_alltrial_df = AllSU_psth_trialBYtrial[(AllSU_psth_trialBYtrial['trialMod']==row['trialMod'])&
                                                            (AllSU_psth_trialBYtrial['snr']==row['snr'])].sort_values(by=['sess_cls','trialNum'])
        frate_tempI_alltrial = np.array([frate_tempI_alltrial_df[frate_tempI_alltrial_df['sess_cls']==clstemp][tcol].values for clstemp in sorted(list(frate_tempI_alltrial_df['sess_cls'].unique()))]) # units X trials
        for j,subrow in condstrdf.loc[i:].iterrows():
            frate_tempJ = AllSU_psth_condAve[(AllSU_psth_condAve['trialMod']==subrow['trialMod'])&
                                                            (AllSU_psth_condAve['snr']==subrow['snr'])].sort_values(by=['sess_cls'])[tcol].values 
            frate_tempJ_alltrial_df = AllSU_psth_trialBYtrial[(AllSU_psth_trialBYtrial['trialMod']==subrow['trialMod'])&
                                                            (AllSU_psth_trialBYtrial['snr']==subrow['snr'])].sort_values(by=['sess_cls','trialNum'])  
            frate_tempJ_alltrial = np.array([frate_tempJ_alltrial_df[frate_tempJ_alltrial_df['sess_cls']==clstemp][tcol].values for clstemp in sorted(list(frate_tempJ_alltrial_df['sess_cls'].unique()))]) # units X trials

            distemp = np.tensordot(np.tensordot((frate_tempI-frate_tempJ).reshape(-1,1).T,
                                                        np.linalg.pinv((np.cov(frate_tempI_alltrial)+np.cov(frate_tempJ_alltrial))/2),axes=([1],[0])),
                                                        (frate_tempI-frate_tempJ).reshape(-1,1),axes=([1],[0]))
            if distemp.size ==1:
                DisMat_temp[i,j] = distemp.item()
                DisMat_temp[j,i] = distemp.item()
            else:
                raise ValueError("The result of the tensordot operation is not a scalar.") 
    # print(tt)
    # print(DisMat_temp) 
    return DisMat_temp,tt
            
# get Mahalanobis distance metric across condition pairs
def getMahalanobisDis(AllSU_psth_trialBYtrial_BS,grpbylist,psthCol,binedge,pngfilename):
    AllSU_psth_trialBYtrial_BS['sess_cls'] = AllSU_psth_trialBYtrial_BS['sess']+AllSU_psth_trialBYtrial_BS['cls']
    AllSU_psth_trialBYtrial = AllSU_psth_trialBYtrial_BS.drop_duplicates().reset_index(drop=True)
    AllSU_psth_trialBYtrial = sampBalanceGLM(AllSU_psth_trialBYtrial,grpbylist) # balance trials across sessionsXclsXcond
    AllSU_psth_condAve = AllSU_psth_trialBYtrial.groupby(grpbylist+['sess_cls']).mean(numeric_only=True)[psthCol].reset_index()
    condstrdf = AllSU_psth_condAve.groupby(by=['trialMod','snr'])[['trialMod','snr']].size().reset_index() # stim cond

    MahalanobisDisMat = np.zeros((condstrdf.shape[0],condstrdf.shape[0],len(psthCol)))# cond X cond X timepnts
    # estimate mahalanobis distance over time
    argItems = [(tt,tcol,condstrdf,AllSU_psth_condAve,AllSU_psth_trialBYtrial) for tt,tcol in enumerate(psthCol)]
     
    with Pool(processes=CPUs) as p:
        # spikeTime is a list: trails X spikeNum
        for MahalanobisDisMat_temp,tt in p.starmap(getpairDist,argItems):
            MahalanobisDisMat[:,:,tt] = MahalanobisDisMat_temp
    p.close()
    p.join()
    
    plotheatMatAnimation(MahalanobisDisMat,binedge,condstrdf,pngfilename,zminV=0,zmaxV=400)
    return MahalanobisDisMat

def plotheatMatAnimation(MahalanobisDisMat,binedge,condstrdf,pngfilename,zminV=0,zmaxV=100):
    # plot animation
    frames = [go.Frame(data=[go.Heatmap(z=(MahalanobisDisMat[:, :, i]),zmin=zminV,zmax=zmaxV)], name=str(tt)) for i,tt in enumerate(binedge)]
    ticktextlist = [row['trialMod']+'_'+row['resp']+'_'+str(int(row['snr'])) for i,row in condstrdf.iterrows()]
    # Create the figure with the initial frame
    fig = go.Figure(
        data=[go.Heatmap(z=MahalanobisDisMat[:, :, 0])],
        layout=go.Layout(
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[dict(label="Play", method="animate", args=[None, {"frame": {"duration": 500, "redraw": True}}]),
                        dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])]
            )],
            xaxis=dict(scaleanchor="y", scaleratio=1,tickvals=np.arange(MahalanobisDisMat.shape[0]),ticktext=ticktextlist,range=[-0.5,MahalanobisDisMat.shape[0]-0.5]),
            yaxis=dict(scaleanchor="x", scaleratio=1,tickvals=np.arange(MahalanobisDisMat.shape[0]),ticktext=ticktextlist,range=[-0.5,MahalanobisDisMat.shape[1]-0.5],side='left'),
            margin=dict(t=20, l=20, r=20, b=20),
            uirevision='constant',
            height=850,width=900
        ),
        frames=frames
    )
    # Add slider for control
    sliders = [{
        'steps': [{'method': 'animate', 'args': [[f'{tt}'], {'mode': 'immediate', 'frame': {'duration': 500, 'redraw': True}}], 'label': str(tt)} for i,tt in enumerate(binedge)],
        'transition': {'duration': 400},
        'x': 0.1, 'len': 0.9, 'xanchor': 'left', 'y': -0.1, 'yanchor': 'top'
    }]
    fig.update_layout(sliders=sliders)
    fig.write_html(pngfilename+".html")

def setlegend(legendaxis,titlestr=''):
    legendaxis.set_frame_on(False)
    legendaxis.set_title(titlestr,prop={'size': fontsizeNo-4})
    legendaxis.get_frame().set_alpha(0)
    legendaxis.set_bbox_to_anchor((0.8, 1))
    legendaxis._loc('upper left') 
    for text in legendaxis.get_texts():
        text.set_fontsize(fontsizeNo-6)  

# debug in local pc
# MonkeyDate_all = {'Elay':['230420','231004']}
# Pathway='/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/PSTHdataframe/'
# AVmodPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/AVmodIndex/'
# figSavePath = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Figures/PSTHtraj/'
# ResPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/PSTHtraj/'
# STRFexcelPath = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/STRF/'
# wavformPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/wavformStruct/'

# run in monty
MonkeyDate_all = getMonkeyDate_all()
Pathway = '/home/huaizhen/Documents/MonkeyAVproj/data/PSTHdataframe/'
ResPathway = '/home/huaizhen/Documents/MonkeyAVproj/data/Fitresults/PSTHtraj/'
figSavePath = '/home/huaizhen/Documents/MonkeyAVproj/data/Figures/PSTHtraj/'
AVmodPathway = '/home/huaizhen/Documents/MonkeyAVproj/data/Fitresults/AVmodIndex/'
STRFexcelPath = '/home/huaizhen/Documents/MonkeyAVproj/data/Fitresults/STRF/'
wavformPathway = '/home/huaizhen/Documents/MonkeyAVproj/data/Fitresults/wavformStruct/'


baselinecorrectFlag = False # whether correct psth at each time point with fr before chorus onset
balanceSamples = True # whether balance samples across conditions by adding bootstraped samples to each condition
bootstrapTims =100#100#00# 50 #100
resolutionred = 5#5 # reduce temporal resolotion of the decoding by this scale, original temporal resolution:0.01

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

start_time = time.monotonic()
clsInfo,_ = pickle.load(open(wavformPathway+'AllUnits_spkwavform_Wuless.pkl','rb')) 
if __name__ == '__main__':
    for alignKeys in ['_align2coo']: # '_align2coo' '_align2DT' '_align2js'
        snr_moddiff_peakAlignment_all = pd.DataFrame()
        for STRFstr in ['notsig','sig',]:#['all','notsig','sig']:
            for spkLabel in ['all']: #['all','RS/FS','positive','triphasic']:   
                #for clsNo in [0,1]: #      
                    namstrallSU = alignKeys+'_Hit=A+AV_ttestfilteredUnits_ProjonModXsnr_STRF'+ STRFstr+'_spkwavShape-'+spkLabel.replace('/','or')+'_nonoverlaptimwin50msbinRawPSTH'+'_AVoffset90+120'+'_sample30noreplacement_lessWu-sess'
                
                    psth_proj_df_BSall = pd.DataFrame()
                    for bs in range(bootstrapTims):
                        for Monkey,Date in MonkeyDate_all.items():                   
                            # clsInfo_temp = clsInfo[(clsInfo['spkLabel']==spkLabel)&(clsInfo['Monkey']==Monkey)&(clsInfo['clsNo']==clsNo)]
                            if spkLabel=='all':
                                clsInfo_temp = clsInfo[clsInfo['Monkey']==Monkey]
                            else:
                                clsInfo_temp = clsInfo[(clsInfo['spkLabel']==spkLabel)&(clsInfo['Monkey']==Monkey)]
                            # filter neurons based on different rules
                            df_avMod_all_sig_ori,_ = neuronfilterDF(AVmodPathway,STRFexcelPath,'ttest',Monkey,STRF=STRFstr)
                            df_avMod_all_sig_ori = df_avMod_all_sig_ori[df_avMod_all_sig_ori['session_cls'].isin(clsInfo_temp.session_cls.values)]
                            # randomly pick clusters with replacement
                            df_avMod_all_sig_unique = df_avMod_all_sig_ori.groupby(['Monkey','session','cls','session_cls','STRFsig']).size().reset_index()                    
                            df_avMod_all_sig = df_avMod_all_sig_unique.groupby('STRFsig').sample(30,replace=False).sort_values(by=['Monkey','session'])
                            print(Monkey+' '+STRFstr+' '+spkLabel)
                            print(df_avMod_all_sig_unique.groupby('STRFsig').size().to_string())

                            AllSU_psth_condAve = pd.DataFrame()
                            AllSU_psth_trialBYtrial = pd.DataFrame()
                            AllSU_psth_condAve_shuffled = pd.DataFrame()

                            for Date_temp in Date: 
                                print('session date:'+Date_temp)
                                ######## columns in rawPSTH_df ['Monkey','sess','cls','respLabel','AVoffset', 'snr', 'trialMod', 'snr-shift', 'trialNum','baselineFR']+frate
                                ######## modify line106-107 in reshapePSTH when switch input files
                                rawPSTH_df_all = pickle.load(open(Pathway+Monkey+'_'+Date_temp+'_allSU+MUA_alltri'+alignKeys+'_nonoverlaptimwin_50msbinRawPSTH_df.pkl','rb'))  # Nneurons X clsSamp X trials X bootstrapTimes
                                psthCol_all  = [s for s in list(rawPSTH_df_all.columns) if 'frate' in s]

                                # only use psth during window defined by timwinStart 
                                binedge_all = np.arange(-1.5,0,bin) #align2coo psth window is [-1,1] align2js psth window is [-1.5,0]
                                
                                frateNO = np.arange(min(range(len(binedge_all)), key=lambda i: abs(binedge_all[i] - timwinStart[0])) ,
                                                    min(range(len(binedge_all)), key=lambda i: abs(binedge_all[i] - timwinStart[1]))+1,1)     
                                psthCol =  [s for s in psthCol_all if float(s[5:]) in frateNO]
                                binedge = binedge_all[frateNO]
                                rawPSTH_df = rawPSTH_df_all.drop(columns=list(set(psthCol_all)-set(psthCol)))
                                #filter trial conditions 
                                rawPSTH_df_temp,_ = SortfilterDF(decodrespLabel2str(rawPSTH_df),filterlable =filterdict)  # filter out trials 
                                
                                # rawPSTH_df_all = pickle.load(open(Pathway+Monkey+'_'+Date_temp+'_allSU+MUA_alltri'+alignKeys+'_overlaptimwin_200msbinRawPSTH_df.pkl','rb'))  # Nneurons X clsSamp X trials X bootstrapTimes
                                # psthCol_all  = [s for s in list(rawPSTH_df_all.columns) if 'fr_' in s]                           
                                # # reduce temporal resolution
                                # frcolfull = [psthCol_all[i] for i in np.arange(0,len(psthCol_all),resolutionred,dtype=int)]
                                # # only use psth during window defined by timwinStart 
                                # psthCol =  [s for s in frcolfull if float(s[3:])>=timwinStart[0] and float(s[3:])<=timwinStart[1]]
                                # binedge = [float(s[3:]) for s in psthCol]
                                # rawPSTH_df = rawPSTH_df_all.drop(columns=list(set(psthCol_all)-set(psthCol)))
                                # # filter trial conditions 
                                # rawPSTH_df_temp,_ = SortfilterDF(rawPSTH_df,filterlable =filterdict)  # filter out trials 

                                # rawPSTH_df_temp = rawPSTH_df_filter[rawPSTH_df_filter['sess']==Date_temp].reset_index(drop=True)       
                                # preprocess each cls separately
                                df_avMod_sess_sig = df_avMod_all_sig[df_avMod_all_sig['session']==Date_temp]
                                for cc,cls in enumerate(list(df_avMod_sess_sig['cls'])):                         
                                        rawPSTH_df_temp_cls = rawPSTH_df_temp[rawPSTH_df_temp['cls']==cls].copy().reset_index()
                                        # might sample the same cluster multiple times, so need differentiation
                                        clnamenew = cls+'_'+str(cc)
                                        rawPSTH_df_temp_cls['cls'] = clnamenew 
                                        df_avMod_sess_temp = df_avMod_all_sig[df_avMod_all_sig['session_cls']==Date_temp+'_'+cls]                    
                                        # if need to balance trials across conditions by adding bootstraped samples 
                                        if balanceSamples:                           
                                            # rawPSTH_df_temp_cls = sampBalanceGLM(rawPSTH_df_temp_cls,['cls']+psthGroupCols,random.randint(1, 10000),method='samplewithreplacement',samples=200)[0]
                                            rawPSTH_df_temp_cls = rawPSTH_df_temp_cls.groupby(by=['sess','cls','trialMod','snr','respLabel','AVoffset']).sample(50,random_state=random.randint(1, 10000),replace=True) 

                                        # if need to baseline correct firing rate of each trial of this cls
                                        if baselinecorrectFlag: 
                                            PSTH_arr = rawPSTH_df_temp_cls[psthCol].values - rawPSTH_df_temp_cls['baselineFR'].values.reshape(-1,1) 
                                        else:
                                            PSTH_arr = rawPSTH_df_temp_cls[psthCol].values 

                                        rawPSTH_df_temp_cls[psthCol] = PSTH_arr  
                                        ############process trial label ordered the same as in the task 
                                        # concatenate trial by trial zscored psth of all cluster  
                                        rawPSTH_df_temp_cls_tbyt = rawPSTH_df_temp_cls.copy()
                                        meanovertime = np.nanmean(rawPSTH_df_temp_cls_tbyt[psthCol].values)
                                        stdovertime = np.nanstd(rawPSTH_df_temp_cls_tbyt[psthCol].values)      
                                        rawPSTH_df_temp_cls_tbyt[psthCol] = (rawPSTH_df_temp_cls_tbyt[psthCol].values-meanovertime)/stdovertime
                                        AllSU_psth_trialBYtrial = pd.concat([AllSU_psth_trialBYtrial,rawPSTH_df_temp_cls_tbyt])                                                
                                        # average psth across conditions
                                        psth_df_CondAve = rawPSTH_df_temp_cls.groupby(psthGroupCols)[psthCol].mean().reset_index().sort_values(by=psthGroupCols)
                                        # gaussian smooth averaged psth in each condition, sigma=40ms , binlen 50ms 
                                        psth_df_CondAve[psthCol] = gaussian_filter1d(psth_df_CondAve[psthCol].values, sigma=0.8, axis=-1)            
                                        # zscore smoothed psth within a cluster
                                        PSTH_arr = psth_df_CondAve[psthCol].values
                                        psth_df_CondAve[psthCol] = (PSTH_arr-np.nanmean(PSTH_arr))/np.nanstd(PSTH_arr)
                                        # concatenate all condition averaged psth of this cluster in one row dataframe
                                        colnames = []
                                        for (_,dfrow) in psth_df_CondAve[psthGroupCols].iterrows():
                                            colnames = colnames+['_'.join([str(vv) for vv in list(dfrow.values)])+'_'+tt for tt in psthCol]
                                        psth_concate_df = pd.DataFrame(psth_df_CondAve[psthCol].values.reshape(1,-1),columns=colnames)        
                                        # concatenate condition averaged psth of all clusters
                                        info_temp_df = pd.DataFrame({'Monkey':[Monkey],'sess':[Date_temp],'cls':[clnamenew]})  
                                        AllSU_psth_condAve = pd.concat([AllSU_psth_condAve,
                                                                        pd.concat([info_temp_df,psth_concate_df],axis=1) ],axis=0) 
                                                                                
                            print('cls total: '+str(AllSU_psth_condAve.groupby(['sess','cls']).size().reset_index().shape[0]))
                            print(AllSU_psth_condAve.columns)
                            # print(AllSU_psth_condAve[AllSU_psth_condAve.isna().any(axis=1)].to_string())
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
                            denoiseMat = getdenoiseMat(pca_optimal.components_)
                            AllSU_psth_condAve_denoise.iloc[:,3:] = np.dot(denoiseMat,psthall) # units X (cond X timebin)
                            # reshape psth for following projecting
                            psth,condstrlist = reshapePSTH(AllSU_psth_condAve_denoise,len(psthCol),psthstcol=3)# psth: units X time X cond 
                            # get denoised coefficients of IVs for each cls
                            beta_df_denoise,betaColname = getBeta(AllSU_psth_trialBYtrial.copy(),linregCols,psthCol,denoiseMat,AllSU_psth_condAve) # units X (IVnum X timebin)
                            # remove time dimension from beta
                            beta_df_max = rmTimdim(beta_df_denoise,betaColname,linregCols) 
                            # orthogonalize beta
                            Q,R = np.linalg.qr(beta_df_max[linregCols].values,mode='reduced')# Q: units X linregCond
                            # project raw psth to orthogonalized beta
                            psth_proj = np.tensordot(Q.T,psth,axes=([1],[0])) #psth_proj: linregCond X time X condition
                            # form psth_proj dataframe
                            psth_proj_df = formdf(psth_proj,linregCols,condstrlist,binedge,bsref='on')
                            psth_proj_df['shuffled'] = 'true label'
                            psth_proj_df_shuffled = formdf(psth_proj,linregCols,list(np.random.choice(condstrlist,size=len(condstrlist),replace=False)),binedge,bsref='on')
                            psth_proj_df_shuffled['shuffled'] = 'shuffled label'
                            psth_proj_df_2comb = pd.concat([psth_proj_df,psth_proj_df_shuffled])
                            psth_proj_df_2comb['bs'] = bs
                            psth_proj_df_2comb['Monkey'] = Monkey
                            psth_proj_df_BSall = pd.concat([psth_proj_df_BSall,psth_proj_df_2comb])
                            # plotpsth(AllSU_psth_trialBYtrial.reset_index(drop=True),binedge,['cls','trialMod','snr','respLabel'],psthCol,figSavePath+Monkey+'_rawpsth'+namstrallSU)

                    pickle.dump([psth_proj_df_BSall,EuclidenaDistMat_BSall],open(ResPathway+'twoMonk_bs_2swin'+namstrallSU+'.pkl','wb'))                   
                    
                    psth_proj_df_BSall,EuclidenaDistMat_BSall = pickle.load(open(ResPathway+'twoMonk_bs_2swin'+namstrallSU+'.pkl','rb'))  # Nneurons X clsSamp X trials X bootstrapTimes

                   
                    ################### plot projection on the av axis difference between a and av condition averaged across SNR              
                    psth_proj_df_BSall_mean = psth_proj_df_BSall.groupby(by=['Monkey','time','mod','resp','bs','shuffled'])[['trialMod','mod-slope']].mean().reset_index()
                    psth_proj_mod_a = psth_proj_df_BSall_mean[psth_proj_df_BSall_mean['mod']=='a'].sort_values(by=['Monkey','time','resp','bs','shuffled'])
                    psth_proj_mod_av = psth_proj_df_BSall_mean[psth_proj_df_BSall_mean['mod']=='av'].sort_values(by=['Monkey','time','resp','bs','shuffled'])
                    psth_proj_moddiff_SNRave = psth_proj_mod_a[['Monkey','time','mod','resp','bs','shuffled']].copy()
                    psth_proj_moddiff_SNRave['trialModDiff'] = np.abs(psth_proj_mod_a['trialMod'].values-psth_proj_mod_av['trialMod'].values)
                    #plot distance between A and AV condition on modality axis 
                    fig, axess = plt.subplots(1,1,figsize=(8,4),sharex='col',sharey='col')  
                    font_properties = {'fontname': fontnameStr, 'fontsize': fontsizeNo-4}
                    sns.lineplot(psth_proj_moddiff_SNRave[psth_proj_moddiff_SNRave['shuffled']=='true label'],x='time',y='trialModDiff',hue='Monkey',hue_order=['Elay','Wu'],palette=['magenta','royalblue'],ax=axess,estimator='mean', errorbar=('ci', 95))               
                    sns.lineplot(psth_proj_moddiff_SNRave[psth_proj_moddiff_SNRave['shuffled']=='shuffled label'],x='time',y='trialModDiff',hue='Monkey',hue_order=['Elay','Wu'],palette=['plum','lightsteelblue'],ax=axess,estimator='mean', errorbar=('ci', 95))               
                    new_labels = ['monkey E (True)', 'monkey W (True)','monkey E (Shuffled)',  'monkey W (Shuffled)']
                    handles, labels = axess.get_legend_handles_labels()
                    
                    axess.legend(handles=handles[:4], labels=new_labels, loc='upper right',
                                handler_map={mlines.Line2D: HandlerLine2D(numpoints=1)},frameon=False,
                                bbox_to_anchor=(1.02, 1.03), fontsize=fontsizeNo-8)                
                    axess.set_xlabel('Time (s; relative to audiory target onset)',**font_properties)
                    axess.plot([0,0],[0,2],linestyle='--',color='black')  
                    axess.plot([-0.52,-0.52],[0,1.5],linestyle='--',color='gray') 
                    #                     
                    # axess.legend(handles=handles[:4], labels=new_labels, loc='upper left',
                    #             handler_map={mlines.Line2D: HandlerLine2D(numpoints=1)},frameon=False,
                    #             bbox_to_anchor=(0, 1.03), fontsize=fontsizeNo-8)                     
                    # axess.set_xlabel('Time (s; relative to DT onset)',**font_properties)

                    axess.set_ylabel('A/AV separation',**font_properties)
                    axess.tick_params(axis='both', which='major', labelsize=fontsizeNo-4) 
                    fig.tight_layout()
                    fig.savefig(figSavePath+'Popprjcton_ModalityAxis_A-AVdistance_'+namstrallSU+'_2monk_2swin.'+figformat)
                    plt.close(fig)

                    ################### plot projection on the av axis difference between a and av condition separatly for each SNR, 
                    ################## combine the same monkey's plot in one plot
                    fig, axess = plt.subplots(2,1,figsize=(8,8),sharex='col',sharey='col')  
                    psth_proj_df_BSall_av = psth_proj_df_BSall.copy() 
                    psth_proj_df_BSall_av['snr'] = psth_proj_df_BSall_av['snr'].replace({-15:'difficult',-10:'difficult',-5:'medium',0:'medium',5:'easy',10:'easy'})
                    psth_proj_df_BSall_mean = psth_proj_df_BSall_av.groupby(by=['Monkey','time','mod','resp','bs','shuffled','snr'])[['trialMod','mod-slope']].mean().reset_index()
                    # psth_proj_df_BSall_mean = psth_proj_df_BSall_mean[psth_proj_df_BSall_mean['shuffled']=='true label'].copy()
                    snrcolor = ['limegreen','orange','lightcoral'] # in the order of 'easy','medium','difficult'
                    psth_proj_mod_a = psth_proj_df_BSall_mean[psth_proj_df_BSall_mean['mod']=='a'].sort_values(by=['Monkey','time','resp','bs','snr'])
                    psth_proj_mod_av = psth_proj_df_BSall_mean[psth_proj_df_BSall_mean['mod']=='av'].sort_values(by=['Monkey','time','resp','bs','snr'])
                    psth_proj_moddiff_snr = psth_proj_mod_a[['Monkey','time','resp','bs','snr','shuffled']].copy()
                    psth_proj_moddiff_snr['trialModDiff'] = np.abs(psth_proj_mod_a['trialMod'].values-psth_proj_mod_av['trialMod'].values)

                    for mm,monk in enumerate(list(psth_proj_df_BSall_mean.Monkey.unique())):
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
                        
                        # axess[mm].set_xlabel('Time (s; relative to DT onset)',**font_properties)
                        # axess[mm].text(0.98, 0.9, 'monkey '+monk[0], 
                        #             transform=axess[mm].transAxes,  # Specify the position in axis coordinates
                        #             fontsize=fontsizeNo-4,             # Font size
                        #             fontname=fontnameStr,
                        #             verticalalignment='top', # Align text at the top
                        #             horizontalalignment='right') # Align text to the right
                        # axess[mm].legend(frameon=False, loc='upper left', bbox_to_anchor=(0, 1.03), title='SNR', fontsize=fontsizeNo-6, title_fontsize=fontsizeNo-4)
                         
                        axess[mm].set_ylabel('A/AV separation',**font_properties)
                        axess[mm].tick_params(axis='both', which='major', labelsize=fontsizeNo-4) 
                    fig.savefig(figSavePath+'Popprjcton_ModalityAxis_A-AVdistanceBYsnr_'+namstrallSU+'_2monk_2swin.'+figformat)
                    plt.close(fig)

                    ################### plot projection on the snr axis
                    fig, axess = plt.subplots(2,1,figsize=(8,8),sharex='col',sharey='col')  
                    font_properties = {'fontname': fontnameStr, 'fontsize': fontsizeNo-4}
                    ycolnames = 'snr-shift_bscrct' #'snr-shift','snr-shift_bscrct','snr-slope'
                    snrcolor = ['limegreen','orange','lightcoral'] # in the order of 'easy','medium','difficult'
                    # snrcolor_bsline = ['violet','lightgreen','forestgreen','lightskyblue','dodgerblue']
                    psth_proj_df_BSall_temp = psth_proj_df_BSall.copy()
                    psth_proj_df_BSall_temp['snr'] = psth_proj_df_BSall_temp['snr'].replace({-15:'difficult',-10:'difficult',-5:'medium',0:'medium',5:'easy',10:'easy'})
                    psth_proj_df_BSsnr = psth_proj_df_BSall_temp.groupby(by=['Monkey','time','snr','bs','shuffled'])[ycolnames].mean().reset_index().reset_index(drop=True)
                    #psth_proj_df_BSsnr = psth_proj_df_BSsnr[psth_proj_df_BSsnr['snr']!=-15].copy()

                    for mm, monk in enumerate(['Elay','Wu']):
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
 
                        # axess[mm].set_xlabel('Time (s; relative to DT onset)',**font_properties)
                        # axess[mm].legend(handles=handles, labels=labels,frameon=False, loc='upper left', bbox_to_anchor=(0, 1.03), title='SNR', fontsize=fontsizeNo-6, title_fontsize=fontsizeNo-4)
                        # axess[mm].text(0.98, 0.9, 'monkey '+monk[0], 
                        #             transform=axess[mm].transAxes,  # Specify the position in axis coordinates
                        #             fontsize=fontsizeNo-4,             # Font size
                        #             fontname=fontnameStr,
                        #             verticalalignment='top', # Align text at the top
                        #             horizontalalignment='right') # Align text to the right
                        
                        axess[mm].set_ylabel('Distance to -15 dB',**font_properties)
                        axess[mm].tick_params(axis='both', which='major', labelsize=fontsizeNo-4) 
                    
                    # fig.tight_layout()
                    fig.savefig(figSavePath+'Popprjcton_SNRAxis_'+namstrallSU+'_2swin.'+figformat)
                    plt.close(fig) 

                    # ### find peak correlation between snr traj and a/av separation traj   
                    psth_proj_df_SNRave = psth_proj_df_BSsnr[psth_proj_df_BSsnr['snr']=='medium'].reset_index(drop=True)                   
                    
                    psth_proj_df_moddiff_SNRave = psth_proj_moddiff_SNRave.copy()
         
                    psth_proj_moddiff_SNRave_peaktime = psth_proj_df_moddiff_SNRave.loc[psth_proj_df_moddiff_SNRave.groupby(by=['Monkey','bs','shuffled'])['trialModDiff'].idxmax(),['Monkey','time','bs','shuffled']].reset_index(drop=True)
                    psth_proj_df_SNRave_peaktime = psth_proj_df_SNRave.loc[psth_proj_df_SNRave.groupby(by=['Monkey','bs','shuffled'])['snr-shift_bscrct'].idxmax(),['Monkey','time','bs','shuffled']].reset_index(drop=True)
                    psth_proj_moddiff_SNRave_peaktime = psth_proj_moddiff_SNRave_peaktime[psth_proj_moddiff_SNRave_peaktime['shuffled']=='true label'].sort_values(by=['Monkey','bs']).reset_index(drop=True)
                    psth_proj_df_SNRave_peaktime = psth_proj_df_SNRave_peaktime[psth_proj_df_SNRave_peaktime['shuffled']=='true label'].sort_values(by=['Monkey','bs']).reset_index(drop=True)
                    snr_moddiff_peakAlignment = psth_proj_moddiff_SNRave_peaktime[['Monkey','bs']]
                    snr_moddiff_peakAlignment['peakTimeDiff'] = np.abs(psth_proj_moddiff_SNRave_peaktime['time'].values-psth_proj_df_SNRave_peaktime['time'].values)
                    snr_moddiff_peakAlignment['strfSig'] = STRFstr
                    snr_moddiff_peakAlignment_all = pd.concat([snr_moddiff_peakAlignment_all,snr_moddiff_peakAlignment],axis=0).reset_index(drop=True)

        DVstr = 'peakTimeDiff'
        snr_moddiff_peakAlignment_all['strfSig'] = snr_moddiff_peakAlignment_all['strfSig'].replace({'notsig':'nSTRF','sig':'STRF'})
        snr_moddiff_peakAlignment_all['Monkey'] = snr_moddiff_peakAlignment_all['Monkey'].replace({'Wu':'monkey W','Elay':'monkey E'})
        fig, axess = plt.subplots(1,1,figsize=(8,4),sharex='col',sharey='col')
        sns.boxplot(snr_moddiff_peakAlignment_all,x = 'Monkey',y=DVstr,hue='strfSig',hue_order=['nSTRF','STRF'],
                    palette={'nSTRF':'darkgray','STRF':'dimgray'},dodge=True)
        font_properties = {'fontname': fontnameStr, 'fontsize': fontsizeNo-4}
        axess.set_xlabel(' ',**font_properties)
        axess.set_ylabel('Trajectory peak coherence',**font_properties)
        axess.tick_params(axis='both', which='major', labelsize=fontsizeNo-8)
        axess.set_yticks(list(np.arange(0,1.6,0.4)))  
        axess.legend(['nSTRF','STRF'],frameon=False, loc='upper left', bbox_to_anchor=(0, 1.03), title='', fontsize=fontsizeNo-8, title_fontsize=fontsizeNo-6)

        mv_stats_E,mv_pval_E = stats.mannwhitneyu(snr_moddiff_peakAlignment_all[(snr_moddiff_peakAlignment_all['strfSig']=='nSTRF')
                                                                            &(snr_moddiff_peakAlignment_all['Monkey']=='monkey E')][DVstr].values,
                                            snr_moddiff_peakAlignment_all[(snr_moddiff_peakAlignment_all['strfSig']=='STRF')
                                                                            &(snr_moddiff_peakAlignment_all['Monkey']=='monkey E')][DVstr].values,alternative='less')
        mv_stats_W,mv_pval_W = stats.mannwhitneyu(snr_moddiff_peakAlignment_all[(snr_moddiff_peakAlignment_all['strfSig']=='nSTRF')
                                                                            &(snr_moddiff_peakAlignment_all['Monkey']=='monkey W')][DVstr].values,
                                            snr_moddiff_peakAlignment_all[(snr_moddiff_peakAlignment_all['strfSig']=='STRF')
                                                                            &(snr_moddiff_peakAlignment_all['Monkey']=='monkey W')][DVstr].values,alternative='less')
        print('Monkey E: nstrf<strf-pval '+str(mv_pval_E)
              +'\nMonkey W: nstrf<strf-pval '+str(mv_pval_W))
        
        fig.savefig(figSavePath+'Popprjcton_SNR-AVsep-coherenct_'+namstrallSU+'_2swin.'+figformat)
        plt.close(fig) 
                             

times2 = time.monotonic()
print('total time spend for 2 monkeys:')
print(timedelta(seconds= times2- start_time)) 


