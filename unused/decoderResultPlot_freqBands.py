import json
import numpy as np
from scipy.stats import sem,tstd
from matplotlib import pyplot as plt 
import os 

pathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/'+\
    'MonkeyEEG/Fitresults/Fitresults/decoderDimRed/'

figFolder = 'decoderDimRed/'
figpathway = os.path.join('/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/'+\
    'MonkeyEEG/Figures/',figFolder)

try:
    os.mkdir(figpathway)
except FileExistsError:
    pass

timeRange = [-0.7,0.5]
freBand = {'none':[],'delta':[1,4],'theta':[4,8],\
            'alpha':[8,12],'beta':[12,25],'lowgamma':[25,58]}
# freBand = {'none':[],'theta':[4,8]}

Monkey = 'Elay'
AlleegalignStr = ['align2js','align2coo']# 'align2coo' 'align2js'
AllchannelStr = ['align2coo_medial1']  #'align2coo_medial1' 'align2coo_frontal3' 
extraStr = '_av'+'_PCAraw'+'_hit_'+'decode_snr'

for eegalignStr in AlleegalignStr:
    for channelStr in AllchannelStr:
        figfilename = Monkey +'_'+eegalignStr+'_'+channelStr+extraStr+'_'
        fig1,axes1 = plt.subplots()
        fig2,axes2 = plt.subplots(2,3,sharex=True,sharey=True)
        axes2 = axes2.ravel()
        for indd,frekeys in enumerate(freBand):

            jsonFilename = Monkey +'_'+eegalignStr+'_'+channelStr+'_'+frekeys+extraStr+'_svmFitscore.json'
            with open(pathway+jsonFilename) as fq:
                data = json.load(fq)

            try: 
                data[list(data.keys())[0]].keys()

                fitresult = data['all_fit_result_slide']
                raw_Mean = data['all_raw_timeseries']
                catsNum = len(list(raw_Mean[list(raw_Mean.keys())[0]][0][0].keys()))-1
                totalSessions = len(list(fitresult.keys()))
                totalRpts = len(fitresult[list(fitresult.keys())[0]][0])
                totalTimWind = len(fitresult[list(fitresult.keys())[0]][0][0])
                
                aveAccDaily = np.empty((totalSessions,totalTimWind)) #sessions X timeslide
                for dd, dkey in enumerate(list(fitresult.keys())):
                    for tt in range(totalTimWind):
                        aveAccDaily[dd,tt] = np.mean([fitresult[dkey][0][i][tt]['accuracy_score'] for i in range(totalRpts)])
                timpnts1= np.linspace(timeRange[0],timeRange[1],totalTimWind)
                Sess_mean_Acc = np.mean(aveAccDaily,axis=0)
                Sess_sd_Acc = tstd(aveAccDaily,axis=0)
                axes1.plot(timpnts1,Sess_mean_Acc,label=frekeys)
                axes1.fill_between(timpnts1,Sess_mean_Acc-Sess_sd_Acc,Sess_mean_Acc+Sess_sd_Acc,alpha=0.2)
                if indd==len(freBand)-1:
                    axes1.plot(timpnts1,np.ones_like(timpnts1)/catsNum,'k',label = 'chance')

                raw_Mean = data['all_raw_timeseries']
                print('num of trials in each category in each training repetition')
                print([str(raw_Mean[list(raw_Mean.keys())[0]][0][0]['CatnumTrials']) for rr in range(totalRpts)])
                raw_Mean_temp = raw_Mean[list(raw_Mean.keys())[0]][0][0]
                totalTimpnts = len(raw_Mean_temp[list(raw_Mean_temp.keys())[0]][0])
                
                timpnts2= np.linspace(timeRange[0],timeRange[1],totalTimpnts)
                            
                for cc, conds in enumerate(list(raw_Mean_temp.keys())[0:-1]):
                    aveTimeseriesDaily = np.empty((totalSessions,totalTimpnts))
                    for dd, dkey in enumerate(list(raw_Mean.keys())):
                        aveTimeseriesDaily[dd,:] = np.mean(np.array([raw_Mean[dkey][0][rr][conds][0] for rr in range(totalRpts)]),axis=0)
                    
                    Sess_mean_Timeseries = np.mean(aveTimeseriesDaily,axis=0)
                    Sess_sd_Timeseries = tstd(aveTimeseriesDaily,axis=0)
                    axes2[indd].plot(timpnts2,Sess_mean_Timeseries,label=conds)
                    axes2[indd].fill_between(timpnts2,Sess_mean_Timeseries-Sess_sd_Timeseries,Sess_mean_Timeseries+Sess_sd_Timeseries,alpha=0.2)
                axes2[indd].set_title(frekeys)         
                axes2[indd].set_ylabel('Amp')
                axes2[indd].legend(fontsize='xx-small',framealpha=0)
            except AttributeError:
                fitresult = data['all_fit_result_slide']
                catsNum = len(list(data['all_raw_timeseries'][0].keys()))-1
                aveAcc = np.empty((0))
                sdAcc = np.empty((0))
                for tt in range(len(fitresult[0])):
                    aveAcc = np.append(aveAcc,np.mean([fitresult[i][tt]['accuracy_score'] for i in range(len(fitresult))]))
                    sdAcc = np.append(sdAcc,tstd([fitresult[i][tt]['accuracy_score'] for i in range(len(fitresult))]))
                timpnts1= np.linspace(timeRange[0],timeRange[1],len(fitresult[0]))
                axes1.plot(timpnts1,aveAcc,label=frekeys)
                axes1.fill_between(timpnts1,aveAcc-sdAcc,aveAcc+sdAcc,alpha=0.2)
                if indd==len(freBand)-1:
                    axes1.plot(timpnts1,np.ones_like(timpnts1)/catsNum,'k',label = 'chance')

                raw_Mean = data['all_raw_timeseries']
                print('num of trials in each category ' + str(raw_Mean[0]['CatnumTrials']))
                timpnts2= np.linspace(timeRange[0],timeRange[1],len(raw_Mean[0][list(raw_Mean[0].keys())[0]][0]))
                aveTimeseries = np.empty((0))
                semTimeseries = np.empty((0))
                for (ind,conds) in enumerate(list(raw_Mean[0].keys())):
                    try:
                        aveTimeseries = np.array(raw_Mean[0][conds][0])
                        semTimeseries = np.array(raw_Mean[0][conds][1])
                        axes2[indd].plot(timpnts2,aveTimeseries,label=conds)
                        # axes2[indd].fill_between(timpnts2,aveTimeseries-semTimeseries,aveTimeseries+semTimeseries,alpha=0.2)
                    except TypeError:
                        pass
                axes2[indd].set_title(frekeys)         
                axes2[indd].set_ylabel('Amp')
                axes2[indd].legend(fontsize='xx-small',framealpha=0)

            axes2[indd].set_xlabel('time (s)')
            axes1.set_xlabel('time points (s)')
            axes1.set_ylabel('Acc')
            axes1.set_ylim(1/catsNum-0.1,1/catsNum+0.15)
            axes1.legend(fontsize='xx-small',framealpha=0)
        fig1.savefig(figpathway+figfilename+'_ACC.png')    
        fig2.savefig(figpathway+figfilename+'_rawTimeseries.png')
