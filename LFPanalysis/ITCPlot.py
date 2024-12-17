import json
import numpy as np
from scipy.stats import sem
from matplotlib import pyplot as plt 

figpathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Figures/'
pathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/'
extraStr = 'Elay_align2js_align2coo_medial1_tf_phase_ITC_'
fig, axs = plt.subplots( len(np.arange(10,-20,-5)),2, figsize=(5,15), sharey=True,sharex=True)
# axs = axs.ravel()
for indd, snr_temp in enumerate(np.arange(10,-20,-5)):
    jsonFilename = extraStr+'snr'+str(snr_temp)+'_hit.json'

    with open(pathway+jsonFilename) as fq:
        data = json.load(fq)

    result = data['ITC_tf_cats']
    timeRange = data['timeRange']
    morletFreq = data['morletFreq']
    numTrials = data['trials']

    # # estimate ITC dif between conditions
    # vmax = 0.2
    # vmin = -0.2
    # result_cat_diff = np.array(result[list(result.keys())[0]])-np.array(result[list(result.keys())[1]])
    # show_temp = axs[indd].imshow(result_cat_diff, cmap=plt.cm.viridis,
    #                             extent=[timeRange[0], timeRange[1], morletFreq[0], morletFreq[-1]],
    #                             aspect='auto', origin='lower') #
    # axs[indd].set_title('snr'+str(snr_temp)+' '+str(numTrials)+' trials')
    # axs[indd].set_xlabel('time(s)')
    # axs[indd].set_ylabel('Freq(Hz)')
    # fig.colorbar(show_temp, ax=axs[indd])

    # find global max/min
    vmax = np.max(result[max(result,key=result.get)])
    print(vmax)
    # vmin = np.min(result[min(result,key=result.get)])
    vmax = 0.47
    vmin = 0
    for inddcat,catkeys in enumerate(result):
        result_cat_temp = np.array(result[catkeys])
        show_temp = axs[indd,inddcat].imshow(result_cat_temp, cmap=plt.cm.viridis,
                                    extent=[timeRange[0], timeRange[1], morletFreq[0], morletFreq[-1]],
                                    aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
        axs[indd,inddcat].set_title(catkeys+' '+str(numTrials)+' trials')
        axs[indd,inddcat].set_xlim(left=timeRange[0], right=0)
        # axs[indd,inddcat].set_xlabel('time(s)')
        # if indd ==0:
        #     axs[indd,inddcat].set_ylabel('Freq(Hz)')
        # fig.colorbar(show_temp, cax=axs[indd,1])
fig.savefig(figpathway+extraStr+'_hit.png')  
