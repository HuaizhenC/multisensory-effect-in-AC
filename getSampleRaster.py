import numpy as np
import seaborn as sns
import mat73
import os
import pandas as pd
from matplotlib import pyplot as plt 
from spikeUtilities import getClsRasterMultiprocess,SortfilterDF,loadPreprocessMat,getPSTH
from sharedparam import getMonkeyDate_all,neuronfilterDF
from datetime import datetime
######################self defined functions

def plotraster2(spikeTimedf_temp,timwin,bintimeLen,axes,extraStr,behavefs):
    rasterdotsize = 5
    
    linecolors = ['lime','orange','red'] # in the order of 'easy','medium','difficult'
    dotcolors= ['lime','orange','red']
    
    yaxmax = []
    # plot raster
    if extraStr=='Chorus':     
        ax1 =  axes[0] 
        # plot raster 
        spikeTimedf_temp_sorted = spikeTimedf_temp.sort_values(by=['trialNum'],ascending=True).reset_index(drop=True)
        spikeTimedf_temp_sorted['trialNumGlob'] =pd.factorize(spikeTimedf_temp_sorted['trialNum'])[0]
        sns.scatterplot(spikeTimedf_temp_sorted,x='spktim',y='trialNumGlob',color='gray',ax=ax1,s=rasterdotsize) 
        ax1.plot([0,0],[0,max(spikeTimedf_temp_sorted['trialNumGlob'].values)],color='black',linestyle='--',linewidth=linewidth)
        ax1.tick_params(axis='both', which='major', labelsize=fontsizeNo-8)
        ax1.set_ylabel('Trials',**font_properties)
        ax1.set_xlabel('',**font_properties)
        ax1.set_yticks([])  
        ax1.set_xlim(timwin)  
        ax1.set_xticks([]) 

        ax2 = axes[1]#axes.twinx()
        timpnts = np.arange(timwin[0],timwin[1],1/1000)
        psth = np.array(getPSTH(list(spikeTimedf_temp_sorted.spktim.values),timpnts,bintimeLen,
                       len(spikeTimedf_temp_sorted['trialNum'].unique()),len(spikeTimedf_temp_sorted['trialNum'].unique()),
                       kernelFlag='on')[0])/bintimeLen
        ax2.plot(timpnts,psth,color='gray',linewidth=linewidth)
        ax2.set_xlabel('Time (s; relative to background onset)',**font_properties)
        ax2.set_xticks([timwin[0],0,timwin[1]])    
        ax2.set_xlim(timwin)        
        yaxmax.append(np.max(psth))

    if extraStr=='Coo':  
        for cc,labname in enumerate(sorted(spikeTimedf_temp['trialMod'].unique())):            
            groupdf_ori = spikeTimedf_temp[spikeTimedf_temp['trialMod']==labname]
            diffcatcol = 'snr'
            # group snr into 3 goups
            groupdf_ori['snr']=groupdf_ori['snr'].replace({-15:'difficult',-10:'difficult',-5:'medium',0:'medium',5:'easy',10:'easy'})
            groupdf_ori['snr']  = pd.Categorical(groupdf_ori['snr'], categories=['difficult','medium','easy'], ordered=True)

            groupdf=groupdf_ori.sort_values(by=[diffcatcol,'trialNum'],ascending=[True,True],axis=0,kind='mergesort',inplace=False).reset_index(drop=True)
            groupdf['trialNumGlob'] =pd.factorize(groupdf['trialNum'])[0]

            # Get unique snr categories
            categories = ['easy','medium','difficult']
            # # Map categories to colors
            dotcolor_dict = dict(zip(categories, dotcolors))
            linecolor_dict = dict(zip(categories, linecolors))

            ax1 = axes[0,cc]  
            # plot raster 
            sns.scatterplot(groupdf,x='spktim',y='trialNumGlob',hue='snr',hue_order=categories,palette=dotcolor_dict,ax=ax1,s=rasterdotsize)            
            #add joystick move onset   
            sns.scatterplot(groupdf.drop_duplicates(subset='trialNumGlob').reset_index(),
                            x='jsOnset',y='trialNumGlob',marker='x',color='black',ax=ax1,s=rasterdotsize)  
            ax1.plot([0,0],[0,max(groupdf['trialNumGlob'].values)],color='black',linestyle='--',linewidth=linewidth)            
            
            timpnts = np.arange(timwin[0],timwin[1],1/1000)
            
            ax2 = axes[1,cc]#axes.twinx()
            if len(groupdf['trialNum'].unique())>=1:
                for snr_temp in categories:
                    groupdf_temp = groupdf[groupdf['snr']==snr_temp]
                    psth = np.array(getPSTH(list(groupdf_temp.spktim.values),timpnts,bintimeLen,
                                len(groupdf_temp['trialNum'].unique()),len(groupdf_temp['trialNum'].unique()),
                                kernelFlag='on')[0])/bintimeLen                    
                    ax2.plot(timpnts,psth,color=linecolor_dict[snr_temp],linewidth=linewidth,label=snr_temp)
                    yaxmax.append(np.max(psth))
            if cc==1:
                legend_elements = [plt.Line2D([0], [0], color=vv, lw=2,linestyle=':', label=str(kk)) for ii,(kk,vv) in enumerate(linecolor_dict.items())]
                # ax1.legend(handles=legend_elements, frameon=False, framealpha=0,loc='upper right',bbox_to_anchor=(1.37, 1), fontsize=fontsizeNo-8,title='SNR',title_fontsize=fontsizeNo-6)                
                ax1.legend([], frameon=False, framealpha=0,loc='upper right',bbox_to_anchor=(1.37, 1))                
                ax2.legend(frameon=False, framealpha=0,loc='upper right',bbox_to_anchor=(1.38, 1), fontsize=fontsizeNo-8,title='SNR',title_fontsize=fontsizeNo-6)
            else:
                ax1.legend([], frameon=False, framealpha=0) 
                ax2.legend([], frameon=False, framealpha=0) 
            
            ax1.plot([0,0],[0,max(groupdf['trialNumGlob'].values)],color='black',linestyle='--',linewidth=linewidth)
            ax1.tick_params(axis='both', which='major', labelsize=fontsizeNo-8)
            # ax1.set_ylabel('Trials',**font_properties)
            ax1.set_ylabel('',**font_properties)
            ax1.set_xlabel('',**font_properties)
            ax1.set_yticks([])   
            ax1.set_xlim(timwin)
            ax1.set_xticks([])
            if labname=='a':
                txtstr = 'Static Visual' 
                xoffset = 0.37
            if labname=='av':
                txtstr = 'Congruent Visual' 
                xoffset = 0.43
            ax1.text(xoffset, 0.97, txtstr,  
                transform=ax1.transAxes, color='black',fontweight='bold',# Specify the position in axis coordinates
                fontsize=fontsizeNo-6,             # Font size
                fontname=fontnameStr,
                verticalalignment='top', # Align text at the top
                horizontalalignment='right') # Align text to the right            

            ax2.set_xlabel('Time (s; relative to target onset)',**font_properties)
            ax2.set_xticks([timwin[0],0,timwin[1]])    
            ax2.set_xlim(timwin)  

    return yaxmax

 

# MonkeyDate_all = {'Elay':['230420'] } #
# # MonkeyDate_all = {'Elay':['230613','230927','240306','230829','231004','231019'],'Wu':['240701']} #
figsavPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Figures/RasterSample'
Pathway='/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/data'

# MonkeyDate_all = getMonkeyDate_all()
MonkeyDate_all = {'Wu':['240625'] }#{'Wu':['240713','240625',],'Elay':['231017'] } 
# figsavPathway = '/home/huaizhen/Documents/MonkeyAVproj/data/Figures/RasterSample'
# Pathway='/home/huaizhen/Documents/MonkeyAVproj/data/preprocNeuralMatfiles/Vprobe+EEG2/'

figformat = 'svg'
fontsizeNo = 24
fontnameStr = 'Arial'#'DejaVu Sans'
font_properties = {'fontname': fontnameStr, 'fontsize': fontsizeNo-6}
linewidth = 3

manualInspectDF = pd.DataFrame()
if __name__ == '__main__':
    #start go through session-cluster 
    for Monkey,Date in MonkeyDate_all.items():    
        for Date_temp in Date:
            outputPathway = os.path.join(figsavPathway,Monkey+Date_temp)
            try:
                os.mkdir(outputPathway)
            except FileExistsError:
                pass
            spikeTimeDict,labelDictFilter, \
                timeSamp2Chorus,spikefs,behavefs= loadPreprocessMat(Monkey,Date_temp,Pathway)
            
            # get trial by trial raster in each cluster, align2chorus
            cccount = 0
            for keys in list(spikeTimeDict.keys()): #['cls35_ch5_mua'] ['cls47_ch10_good']
                if 'cls' in keys:
                    if any(substr in keys for substr in ['good','mua']):
                        print('..................'+Monkey+Date_temp+' '+keys+' in progress............')
                        # fig, axess = plt.subplots(1,3,figsize=(22,6),gridspec_kw={'width_ratios': [1,1.5,1.5]})
                        # plt.subplots_adjust(wspace=0.3)
                        # # Share x axes between the second and third plots
                        # ax2 = axess[0,1]  # Second plot
                        # ax3 = axess[0,2]  # Third plot
                        # # Make second and third plots share x axes
                        # ax3.sharex(ax2)
                        fig, axess = plt.subplots(2,3,figsize=(22,8),gridspec_kw={'width_ratios': [1,1.5,1.5],'height_ratios': [3, 1]})
                        plt.subplots_adjust(wspace=0.2,hspace=0.05)
                        # # Make the second and third columns share the x-axis
                        # axess[0, 0].sharex(axess[1, 0])  # Share x-axis between axess[0,1] and axess[0,2]
                        # axess[0, 2].sharex(axess[0, 1])  # Share x-axis between axess[0,1] and axess[0,2]
                        # axess[1, 1].sharex(axess[0, 1])  # Share x-axis between axess[0,1] and axess[1,1]
                        # axess[1, 2].sharex(axess[0, 1])  # Share x-axis between axess[0,1] and axess[1,2]

                        # # get trial by trial raster in each cluster
                        # labelDict_sub = {}
                        # ntrials = 200
                        # for key,value in labelDictFilter.items():
                        #     labelDict_sub[key] = value[-ntrials:]
                        # labelDictFilter = labelDict_sub.copy()
                        # spikeTime_temp = spikeTimeDict[keys][-ntrials:]
                                            
                        spikeTime_temp = spikeTimeDict[keys]

                        extrastr = 'Chorus'
                        timwin = [-0.3,0.3]
                        spikeTimedf_temp = getClsRasterMultiprocess(timeSamp2Chorus,spikeTime_temp,spikefs,behavefs,\
                                                            labelDictFilter['chorusOnsetInd'],\
                                                            labelDictFilter['chorusOnsetInd'] ,\
                                                            labelDictFilter['joystickOnsetInd'],\
                                                            timwin,labelDictFilter)                            
                        spikeTimedf_temp_filtered,_ = SortfilterDF(spikeTimedf_temp,filterlable = {'trialMod':['a','av','v'],'respLabel':[0,1,88,10,100]})                                               
                        ymax1 = plotraster2(spikeTimedf_temp_filtered.reset_index(drop=True),timwin,20/1000,axess[:,0],extrastr,behavefs)

                        extrastr = 'Coo'
                        timwin = [-0.6,1]
                        spikeTimedf_temp = getClsRasterMultiprocess(timeSamp2Chorus,spikeTime_temp,spikefs,behavefs,\
                                                            labelDictFilter['chorusOnsetInd'],\
                                                            labelDictFilter['cooOnsetIndwithVirtual'] ,\
                                                            labelDictFilter['joystickOnsetInd'],\
                                                            timwin,labelDictFilter)                            
                        spikeTimedf_temp_filtered,_ = SortfilterDF(spikeTimedf_temp,filterlable = {'trialMod':['a','av'],'respLabel':[0]})
                        ymax2 = plotraster2(spikeTimedf_temp_filtered,timwin,20/1000,axess[:,1:],extrastr,behavefs)

                        # set all ax2 ylim the same range
                        for axnum,allax2_temp in enumerate(axess[1,:]):
                            ymaxval = np.round(1.2*np.max(ymax1+ymax2),decimals=1)
                            allax2_temp.set_ylim([0,ymaxval])
                            allax2_temp.yaxis.set_ticks([0,np.round(ymaxval/2,decimals=1),ymaxval]) 
                            allax2_temp.tick_params(axis='both', which='major', labelsize=fontsizeNo-8)
                            if axnum==0:
                                allax2_temp.set_ylabel('Firing rate (Hz)',labelpad=5,**font_properties)
                            allax2_temp.plot([0,0],[0,ymaxval],color='black',linestyle='--',linewidth=linewidth)

                        fig.suptitle(str(cccount)+'_'+Date_temp+keys)
                        # fig.tight_layout()
                        fig.savefig(outputPathway+os.path.sep+str(cccount)+'_'+keys+'_raster'+'.'+figformat,format = figformat)
                        plt.close()  
                        cccount = cccount+1 

                        


            
              
                    




