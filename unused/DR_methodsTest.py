import time
start_time = time.monotonic()
from datetime import timedelta

from eegUtilities import loadTimWindowedEEG,genLabels,BalanceSamples
import numpy as np
from dimRedMethods import umapMethod
import matplotlib.pyplot as plt


Monkey = 'Elay'
# date = ['221128'] # two channel sessions
date = ['221128','221129','221130','221205','221207','221208','221212','221213','221214','221226','221227'] # two channel sessions
# date = ['221109','221111','221114','221116','221121'] # one frontal channel sessions
eegfilePathway = '/data/by-user/Huaizhen/'
outputPathway = '/data/by-user/Huaizhen/Fitresults/'

# Monkey = 'Wu'
# date = ['1',]
# eegfilePathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/'
# outputPathway = './'

timeRange = [0.2,0.3]
binTim = 0.1 #sec, time window for svm
freBand = {'none':[],'delta':[1,4],'theta':[4,8],\
            'alpha':[8,12],'beta':[12,25],'lowgamma':[25,58]}

AlleegalignStr = ['align2js','align2coo']# 'align2coo' 'align2js'
AllchannelStr = ['align2coo_medial1' ]  #'align2coo_medial1' 'align2coo_frontal3' 
# filterCond: has multipe keys ['respLabel','trialMod','snr'], each key can only has one value
# 'respLabel' [0,1] --> [hit, miss]
filterCond = {'respLabel':[0],'snr':[-10]}
# catCond: only one cat condition key, can have multiple values               
catCondall = [{'trialMod':['a','av']} ]

for catCond in catCondall:
    catNames = catCond[list(catCond.keys())[0]]
    extraStr ='_hits_snr-10_'+list(catCond.keys())[0]+'_2Dscatter'
    for eegalignStr in AlleegalignStr:
        for channelStr in AllchannelStr:
            jsonnameStr = Monkey+'_'+eegalignStr+'_'+channelStr+'_'+extraStr 

            fig, axes = plt.subplots(1, len(list(freBand.keys())),figsize=(40, 4))
            for indd,frekeys in enumerate(freBand):
        
                print(jsonnameStr)
                for dd in date:
                    print('\ncutting eeg in session: '+dd)
                    eegSeg = loadTimWindowedEEG(eegfilePathway+Monkey+'_'+dd+'_preprocEEGtimeseries.mat',eegalignStr,channelStr,timeRange,freBand[frekeys],[]) # trials X features(time) array 
                    Behavdata = genLabels(eegfilePathway+Monkey+'_'+dd+'_labviewNsynapseBehavStruct.mat',filterCond,catCond) # trials X 1 array
                    #find valid trials index
                    BehavalInd = [i for i,v in enumerate(Behavdata) if v==v] # trials fullfill behave filterCondition
                    xx_temp = eegSeg[BehavalInd,:]
                    yy_temp = Behavdata[BehavalInd]
                    # dignity check: filtered trials stimulus not played, should always empty if filterCond['respLabel'] is 0 or 1
                    StimvalInd = [j for j,vec in enumerate(xx_temp) if (vec!=vec).any()] 
                    if len(StimvalInd) !=0:
                        print('nan remained in filtered eeg segments in ' + Monkey + ' ' + dd)
                        xx_temp = np.delete(xx_temp,StimvalInd,axis=0)
                        yy_temp = np.delete(yy_temp,StimvalInd,axis=0)

                    try:
                        xx = np.append(xx,xx_temp,axis = 0)
                        yy = np.append(yy,yy_temp,axis = 0)
                    except NameError:
                        xx = xx_temp  
                        yy = yy_temp  
                                           
                # apply differen dimension reduction methods
                print('START dimension reduction for '+frekeys +'.................')                
                umapMethod(xx,yy,n_neighbors=15,min_dist=0.1,n_components=2,metric='euclidean',n_jobs=6).\
                    applyumap(catNames,axes[indd],frekeys)
                print('dimension reduction DONE.................')
            
                # delete xx,yy after each condition
                del xx, yy

            fig.savefig(outputPathway+jsonnameStr+'.png') 
            plt.close()



  
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))








