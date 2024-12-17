import time
start_time = time.monotonic()
from datetime import timedelta
from eegUtilities import loadTimWindowedLFP
from spikeUtilities import loadBehavMat,decodrespLabel2str
import numpy as np
import pandas as pd
import pickle
import imageio.v3 as iio
import imageio
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mne.stats import permutation_cluster_test

numFreq = 80
morletFreq = np.logspace(np.log10(1),np.log10(150),numFreq)

def getcondMeanSpectrum(xx_data,filter=["'a'","'hit'"],input='phase',logicstr='all'):
    # filter keys include all filter trings
    if logicstr=='all':
        allkeys_filtered = [element for element in list(xx_data.keys()) if all(subs in element for subs in filter)]
    if logicstr=='any':
        allkeys_filtered = [element for element in list(xx_data.keys()) if any(subs in element for subs in filter)]
    
    trialKeys = [element for element in allkeys_filtered if 'trials' in element]
    arrayKeys = list(set(allkeys_filtered)^set(trialKeys))
    contrials = sum([xx_data[key] for key in trialKeys]) # number of trials summed up in total
    xx_temp = np.stack([xx_data[key] for key in arrayKeys])
    if input=='power':
        xx_data_updated = np.sum(xx_temp,axis=0)/contrials # mean power
    if input =='phase':
        xx_data_updated = np.angle(np.sum(xx_temp,axis=0))
    return xx_data_updated # chan X freq X time

def add0Str2xtick(xt,x0str):            
    xt = np.delete(xt,np.where(xt==0)[0])
    xt = np.append(xt,0)
    xtl = xt.tolist()
    xtl = [np.round(xtl[i],1) for i in range(len(xtl))]
    xtl[-1] = x0str
    return xt,xtl

MonkeyDate_all = {'Elay':[['230420',[-3,3]]]} #

# MonkeyDate_all = {'Elay':['230711','230705','230718']} #

FigPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Figures/LFP/'
ResPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/LFP/'

# MonkeyDate_all = {'Elay':[['230420',[-3,3]],['230503',[2,2]],['230509',[1,2]],
#                           ['230531',[1,1]],['230602',[1,0]],['230606',[0,2]],
#                           ['230613',[2,1]],['230616',[2,0]],['230620',[2,-1]],
#                           ['230627',[2,-1]],['230705',[3,0]],['230711',[2,2]],
#                           ['230717',[0,1]],['230718',[3,-2]],['230719',[0,-1]]]} # [A/P,M/L] one monkey each time

# Pathway='/data/by-user/Huaizhen/preprocNeuralMatfiles/EEG+Vprobe/'
# ResPathway = '/data/by-user/Huaizhen/LFPcut/2Dpower+phase/'
# FigPathway = '/data/by-user/Huaizhen/LFPcut/2Dpower+phase/Figures/'

# timeRange_all = [[-0.5,0.5],[-0.9,0.5],[-0.5,0.5]] #fs 1220.703125
# AllalignStr = ['align2cho','align2coo','align2js']
# x0str_all = ['chOn','cooOn','JSOn']
# alignkeys_all = ['chorusOnsetInd','cooOnsetIndwithVirtual','joystickOnsetInd']

timeRange_all = [[-0.5,0.5]] #fs 1220.703125
AllalignStr = ['align2cho']
x0str_all = ['chOn']
alignkeys_all = ['chorusOnsetInd']

for alignkeys,alignStr,timeRange,x0str in zip(alignkeys_all,AllalignStr,timeRange_all,x0str_all): 
    for chan in range(24):  
        fig, axes = plt.subplots(8,8,figsize=(24, 24))
        for Monkey,DateNaxcord in MonkeyDate_all.items():
            for DateNaxcord_temp in DateNaxcord:
                dd = DateNaxcord_temp[0]
                ax2plot = DateNaxcord_temp[1]
                print('in session: '+dd)
                gif_path = ResPathway+Monkey+'_2Dpowerdiff'+alignStr+'_map.gif'
                F_obs,F_obs_plot_all,xx_data_cond_diff = pickle.load(open(ResPathway+Monkey+dd+'_PermTestFobs2D_'+alignStr+'.pkl','rb')) 
                F_obs_plot = F_obs_plot_all[chan,:,:]

                axcol = int(-1*ax2plot[1]+4)
                asrow = int(-1*ax2plot[0]+4)
                #plot contrast and sigmap
                # vminval = xx_data_cond_diff['power'].min()
                # vmaxval = xx_data_cond_diff['power'].max()                    
                # c=axes[ff].imshow(xx_data_cond_diff['power'][chan,:,:],
                #                     cmap='gray',vmin=vminval,vmax=vmaxval,
                #                     extent=[timeRange[0], timeRange[-1], morletFreq[0],morletFreq[-1]],
                #                             origin='lower',aspect='auto')

                max_F = np.nanmax(abs(F_obs_plot))
                c = axes[asrow,axcol].imshow(F_obs_plot,
                    extent=[timeRange[0], timeRange[-1], morletFreq[0],morletFreq[-1]],
                    aspect="auto",
                    origin="lower",
                    cmap="RdBu_r",
                    vmin=-max_F,
                    vmax=max_F,alpha=1)
                axes[asrow,axcol].set_title('chan'+str(chan),fontsize=8)
                # Set the x-axis and y-axis tick labels
                xt = axes[asrow,axcol].get_xticks()  
                xt,xtl = add0Str2xtick(xt,x0str)
                axes[asrow,axcol].set_xticks(xt)
                axes[asrow,axcol].set_xticklabels(xtl)
                axes[asrow,axcol].set_xlim(-0.9,0.5)
                y_values = morletFreq[[0,20,40,60,numFreq-1]]
                axes[asrow,axcol].yaxis.set_major_locator(ticker.LinearLocator(numticks=len(y_values)))
                axes[asrow,axcol].set_yticks(axes[asrow,axcol].get_yticks())
                axes[asrow,axcol].set_yticklabels([f'{label:.2f}' for label in np.round(y_values,decimals=1)],fontsize=8)
        fig.tight_layout()
        plt.savefig(FigPathway+Monkey+alignStr+f"{chan}.jpg")
        plt.close()
        
    frames = np.stack([iio.imread(FigPathway+Monkey+alignStr+f"{cc}.jpg") for cc in range(24)], axis=0)
    imageio.mimsave(gif_path, frames,format='GIF',duration=300)# duration in ms/frame
    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))








