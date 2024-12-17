import json
import numpy as np
from scipy.stats import sem
from matplotlib import pyplot as plt 

figpathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Figures/'
pathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/'
# jsonFilename = 'Elay_align2js_align2coo_frontal3_svmFitscore_5sessions.json'  
# jsonFilename = 'Elay_align2coo_align2coo_frontal3_svmFitscore_5sessions.json'
jsonFilename = 'Elay_align2js_align2coo_frontal3_Audhit_svmFitscore.json'
chanStr = 'frontal'
extraNameStr = '_Audhit'
timeRange = [-0.7,0.1]

with open(pathway+jsonFilename) as fq:
    data = json.load(fq)

fitresult = data['all_fit_result_slide']
aveAcc = np.empty((0))
semAcc = np.empty((0))
for tt in range(len(fitresult[0])):
    aveAcc = np.append(aveAcc,np.mean([fitresult[i][tt]['accuracy_score'] for i in range(len(fitresult))]))
    semAcc = np.append(semAcc,sem([fitresult[i][tt]['accuracy_score'] for i in range(len(fitresult))]))
fig = plt.figure()
timpnts= np.linspace(timeRange[0],timeRange[1],len(fitresult[0]))
plt.plot(timpnts,aveAcc,color = '#C79FEF',label='decoder')
plt.fill_between(timpnts,aveAcc-semAcc,aveAcc+semAcc,color='#E6E6FA')
plt.plot(timpnts,np.ones_like(timpnts)/5,'r',label = 'chance')
plt.xlabel('time points (s)')
plt.ylabel('Acc')
plt.ylim(0.1,0.5)
plt.legend()
plt.title(chanStr)
plt.savefig(figpathway+jsonFilename[0:-5]+'_ACC.png')
plt.show()


raw_Mean = data['all_raw_timeseries']
fig = plt.figure()
timpnts= np.linspace(timeRange[0],timeRange[1],len(list(raw_Mean[0].values())[0]))
color = {'line':['red','darkorange','gold','yellow','lawngreen','lightseagreen'],\
        'shade':['mistyrose','bisque','lemonchiffon','lightyellow','honeydew','lightcyan']}
aveTimeseries = np.empty((0))
semTimeseries = np.empty((0))
for (ind,conds) in enumerate(list(raw_Mean[0].keys())[0:-1]):
    timeseries_conds_temp = np.array([raw_Mean[i][conds] for i in range(len(raw_Mean))])
    aveTimeseries = np.mean(timeseries_conds_temp,axis=0)
    semTimeseries = sem(timeseries_conds_temp,axis=0)
    plt.plot(timpnts,aveTimeseries,color = color['line'][ind],label=conds+'dB')
    plt.fill_between(timpnts,aveTimeseries-semTimeseries,aveTimeseries+semTimeseries,color=color['shade'][ind])
plt.xlabel('time (s)')
plt.ylabel('Amp')
plt.title(chanStr)
plt.legend()
plt.savefig(figpathway+jsonFilename[0:-5]+'_rawTimeseries.png')
plt.show()