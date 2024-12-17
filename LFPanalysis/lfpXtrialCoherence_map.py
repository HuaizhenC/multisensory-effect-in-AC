import numpy as np
import pandas as pd
import os
import imageio.v3 as iio
import imageio
import matplotlib.pyplot as plt
import pickle
import re

numFreq = 80
morletFreq = np.logspace(np.log10(1),np.log10(150),numFreq)

def getChanNum(df_avMod_all_sig, MonkeyDate):
   def getchan(session_clslist):
       # Regular expression pattern to match 'ch' followed by one or more digits
       pattern = r'ch(\d+)'
       # Extract numbers after 'ch' for each element
       ch_numbers = np.unique(np.array([int(re.search(pattern, element).group(1)) for element in session_clslist if re.search(pattern, element)]))
       return ch_numbers
       
   chanAudNeuDF = pd.DataFrame()
   for Monkey,DateNaxcord in MonkeyDate.items():
        for DateNaxcord_temp in DateNaxcord:
            chanAudNeuDF_temp = pd.DataFrame()
            dd = DateNaxcord_temp[0]  
            df_avMod_sess = df_avMod_all_sig[(df_avMod_all_sig['Monkey']==Monkey) & (df_avMod_all_sig['session_cls'].str.contains(dd))]   
            chanAudNeuDF_temp['Sigchan'] = getchan(df_avMod_sess.session_cls.values.tolist())
            chanAudNeuDF_temp['sess'] = dd
            chanAudNeuDF_temp['Monkey'] = Monkey
            chanAudNeuDF = pd.concat((chanAudNeuDF,chanAudNeuDF_temp))
   return chanAudNeuDF

# MonkeyDate = {'Elay':[['230420',[-3,3]]]} #
# chanfigPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Figures/lfpXtrialCoh/'
# figPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Figures/lfpXtrialCoh/allchanfig/'
# AVmodPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/AVmodIndex/'
# input = 'phase'

MonkeyDate = {'Elay':[['230420',[-3,3]],['230509',[1,2]],
                          ['230531',[1,1]],['230602',[1,0]],['230606',[0,2]],
                          ['230613',[2,1]],['230616',[2,0]],['230620',[2,-1]],
                          ['230627',[2,-2]],['230705',[3,0]],['230711',[2,2]],
                          ['230717',[0,1]],['230718',[3,-2]],['230719',[0,-1]],['230726',[-1,1]],['230728',[1,-1]],
                          ['230802',[-1,2]],['230808',[-1,0]],['230810',[-1,-1]],['230814',[0,-2]],['230818',[-1,-2]],['230822',[-2,0]],['230829',[-2,-1]],
                          ['230906',[-2,1]],['230908',[-2,2]],['230915',[-1,3]],['230919',[0,3]],['230922',[1,3]]]} # [A/P,M/L] one monkey each time

chanfigPathway = '/data/by-user/Huaizhen/Figures/lfpXtrialCoh/'
figPathway = '/data/by-user/Huaizhen/Figures/lfpXtrialCoh/allchanfig/'
AVmodPathway = '/data/by-user/Huaizhen/Fitresults/AVmodIndex/'
input_all = ['phase','power']

# filter neurons based on different rules
df_avMod_all = pickle.load(open(AVmodPathway+'AVmodTTestDF.pkl','rb'))
df_avMod_all_sig = df_avMod_all[df_avMod_all['pval']<0.001] 
print(df_avMod_all_sig.to_string())
# get channels with sig auditory neurons
chanAudNeuDF = getChanNum(df_avMod_all_sig,MonkeyDate)

for input in input_all:
    for chan in np.arange(0,24,1):  
        fig, axes = plt.subplots(8,8,figsize=(24, 19))
        for Monkey,DateNaxcord in MonkeyDate.items():
            gif_path = figPathway+Monkey+'_lfpXtrialCoherence_'+input+'_map.gif'
            for DateNaxcord_temp in DateNaxcord:
                dd = DateNaxcord_temp[0]
                ax2plot = DateNaxcord_temp[1]
                axcol = -1*ax2plot[1]+4
                asrow = -1*ax2plot[0]+4
                print('gridsite[a/p,l/m]'+str(ax2plot[0])+' '+str(ax2plot[1])+'    plotaxis[row,col]'+str(asrow)+' '+str(axcol))
                chanfigPathway_temp = os.path.join(chanfigPathway,Monkey+dd)
                fig_nametemp = Monkey+dd+'align2coo_lfpXtrialCoh_'+input+'_diff_ch'+str(chan)+'.jpg'

                data_temp = iio.imread(os.path.join(chanfigPathway_temp,fig_nametemp))
                data_tempcut = data_temp.copy()
                axes[asrow,axcol].imshow(data_tempcut[25:550,35:590])
                # check if this channel of this session have sig auditory neurons
                chanAudNeu_temp = chanAudNeuDF[(chanAudNeuDF['sess']==dd) & (chanAudNeuDF['Monkey']==Monkey)].Sigchan.values            
                chanAudNeu_temp_range = np.arange(chanAudNeu_temp.min(),chanAudNeu_temp.max()+1,1)
                # highlight chan plot
                if chan in chanAudNeu_temp:
                    axes[asrow,axcol].set_xticks([])
                    axes[asrow,axcol].set_yticks([])
                    axes[asrow,axcol].spines['top'].set_color('red')     # Top border
                    axes[asrow,axcol].spines['bottom'].set_color('red')  # Bottom border
                    axes[asrow,axcol].spines['left'].set_color('red')    # Left border
                    axes[asrow,axcol].spines['right'].set_color('red')   # Right border
                # elif chan not in chanAudNeu_temp and chan in chanAudNeu_temp_range:
                #     axes[asrow,axcol].axis('off')
                else:
                    axes[asrow,axcol].axis('off')
                    # axes[asrow,axcol].clear()                
                axes[asrow,axcol].set_title('chan'+str(chan)+' '+dd,fontsize=8)
            fig.tight_layout()
            plt.savefig(figPathway+Monkey+'_'+input+'_'+f"{chan}.jpg")
            plt.close()   

    frames = np.stack([iio.imread(figPathway+Monkey+'_'+input+'_'+f"{cc}.jpg") for cc in np.arange(0,24,1)], axis=0)
    imageio.mimsave(gif_path, frames,format='GIF',duration=1000)# duration in ms/frame








