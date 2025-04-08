import pickle
import pandas as pd
import numpy as np

def getMonkeyDate_all():
    # # getAVmodulationIndex, save to txt file '230406','230414','230714','230912',
    # MonkeyDate_all = {'Elay':['230912','230420','230509','230525','230531','230602','230606','230608','230613',
    #                         '230616','230620','230627',
    #                         '230705','230711','230717','230718','230719','230726','230728',
    #                         '230802','230808','230810','230814','230818','230822','230829',
    #                         '230906','230908','230915','230919','230922','230927',
    #                         '231003','231004','231010','231017','231019',
    #                         '231128',
    #                         '240109','240112','240116','240119',
    #                         '240202','240206','240209','240213',
    #                         '240306','240318',
    #                         '240419','240429','240501'],
    #                     'Wu':['240522','240524','240527','240530',
    #                         '240605','240607','240610','240611','240612','240614','240618','240620','240621','240624','240625','240626','240628',
    # '240701','240702','240703','240704','240705','240708','240709','240710','240711','240712','240713','240715','240716','240717']}

    MonkeyDate_all = {'Elay':['230420','230509','230525','230531','230602','230606','230608','230613',
                            '230616','230620','230627',
                            '230705','230711','230717','230718','230719','230726','230728',
                            '230802','230808','230810','230814','230818','230822','230829',
                            '230906','230908','230912','230915','230919','230922','230927',
                            '231003','231004','231010','231017','231019',
                            '231128',
                            '240109','240112','240116','240119',
                            '240202','240206','240209','240213',
                            '240306','240318',
                            '240419','240429','240501'],
                        'Wu':['240621','240624','240625','240626','240628',
    '240701','240702','240703','240704','240705','240708','240709','240710','240711','240712','240713','240715','240716','240717']}
        
    return MonkeyDate_all


def getGranularChan_all():
    # chan from 0-23,no response: '230525','230718','231004'
    # gchan_all = {'230420':7,'230509':8,'230531':6,'230602':8,'230606':8,'230608':11,'230613':8,
    #                         '230616':8,'230620':5,'230627':4,
    #                         '230705':9,'230711':5,'230717':7,'230719':10,'230726':10,'230728':10,
    #                         '230802':10,'230808':7,'230810':13,'230814':5,'230818':6,'230822':5,'230829':3,
    #                         '230906':9,'230908':6,'230912':5,'230915':7,'230919':11,'230922':5,'230927':6,
    #                         '231003':10,'231004':11,'231010':11,'231017':6,'231019':16,
    #                         '231128':7,
    #                         '240109':6,'240112':8,'240116':10,'240119':13,
    #                         '240202':6,'240206':7,'240209':3,'240213':6,
    #                         '240306':6,'240318':9,
    #                         '240419':10,'240429':9,'240501':8,
    #                     '240621':9,'240624':3,'240625':7,'240626':6,'240628':8,
    # '240701':6,'240702':6,'240703':8,'240704':6,'240705':3,'240708':6,'240709':6,'240710':7,'240711':10,'240712':10,'240713':10,'240715':7,'240716':10,'240717':9}

# chan from 0-23,no response: '230525','230718','230810','230829','231004','231019'
# good: 230822,230908,230912,231128,240109,240116,240628

    gchan_all = {'230420':14,'230509':7,'230531':6,'230602':7,'230606':7,'230608':11,'230613':7,
                            '230616':6,'230620':6,'230627':6,
                            '230705':10,'230711':4,'230717':8,'230719':6,'230726':10,'230728':4,
                            '230802':6,'230808':4,'230810':18,'230814':5,'230818':4,'230822':5,'230829':3,
                            '230906':11,'230908':6,'230912':5,'230915':7,'230919':11,'230922':11,'230927':11,
                            '231003':11,'231004':0,'231010':2,'231017':9,'231019':7,
                            '231128':5,
                            '240109':7,'240112':1,'240116':9,'240119':14,
                            '240202':12,'240206':8,'240209':6,'240213':6,
                            '240306':6,'240318':8,
                            '240419':9,'240429':8,'240501':9,
                        '240621':8,'240624':7,'240625':12,'240626':8,'240628':12,
    '240701':8,'240702':5,'240703':10,'240704':6,'240705':8,'240708':10,'240709':10,'240710':11,'240711':11,'240712':10,'240713':8,'240715':7,'240716':10,'240717':9}
         
    return gchan_all

def neuronfilterDF(Pathway,STRFexcelPath,method,Monkey,STRF='all'):
    if len(Pathway)==0:
        AVmodPathway = '/home/huaizhen/Documents/MonkeyAVproj/data/Fitresults/AVmodIndex/'
    else:
        AVmodPathway = Pathway
        
    # filter responsive neurons based on different rules
    if method=='ttest':
        df_avMod_all = pickle.load(open(AVmodPathway+'AVmodTTestDF.pkl','rb'))
        df_avMod_all_sig = df_avMod_all[(df_avMod_all['Monkey']==Monkey)&(df_avMod_all['pval']<0.05)&(df_avMod_all['fr>1']==True)] 
    if method=='all':
        df_avMod_all = pickle.load(open(AVmodPathway+'AVmodTTestDF.pkl','rb'))
        df_avMod_all_sig = df_avMod_all[(df_avMod_all['Monkey']==Monkey)&(df_avMod_all['fr>1']==True)] 
    # print('sessions in AVmodTTestDF.pkl for 2 monkeys')
    # print(df_avMod_all['session_cls'].str[:6].unique())
    clscolstr = 'session_cls'

    print(Monkey+'  '+method+' neurons: '+str(len(df_avMod_all_sig['session_cls'].unique())))      
    ##### filter out drifted neurons
    dfinspect = pd.read_excel('AllClusters4inspectionSheet.xlsx',sheet_name=Monkey)
    driftedUnites = list(dfinspect[dfinspect['driftYES/NO/MAYBE(1,0,2)'].isin([0])]['session_cls'].values)
    df_avMod_all_sig = df_avMod_all_sig[df_avMod_all_sig['session_cls'].isin(driftedUnites)]
    # print('after remove drifting (only keep 0 drift) : '+str(len(df_avMod_all_sig['session_cls'].unique())))      
    # ##### check usable neurons in each session
    # df_usableBYsess=df_avMod_all_sig.drop_duplicates(subset='session_cls',keep='first')
    # print('total usable neurons in each session in '+Monkey)
    # print(df_usableBYsess.groupby('session').size().reset_index())

    # filter out neurons with sigSTRF
    dfSTRF1 = pd.read_excel(STRFexcelPath+'STRF_stats.xlsx',sheet_name=Monkey)
    dfSTRF1['session_cls'] = dfSTRF1.apply(lambda row: f"{row['session']}_{row['sigclsname']}", axis=1)
    # label dmr blocks before and after task as 0 and 1
    dfSTRF = pd.DataFrame()
    for session in list(dfSTRF1.session.unique()):
        dfSTRF_allBlks_temp = dfSTRF1[dfSTRF1['session']==session].sort_values(['blockname'])
        dfSTRF_allBlks_temp['blocknum'] = pd.factorize(dfSTRF_allBlks_temp['blockname'].values,sort=False)[0]
        dfSTRF = pd.concat([dfSTRF,dfSTRF_allBlks_temp],axis=0)
    # # only consider the units with strf measured right after the task
    # dfSTRF = dfSTRF[dfSTRF['blocknum']==1] # comment this line if consider the units with strf measured before and after the task
    ################ track clusters have same strf significance in both the beginning and the end of the task session, addSTRFcol in df_avMod_all_sig     
    dfSTRF = dfSTRF.drop(dfSTRF[dfSTRF['pval1']==10000].index)#delete clusters those can not get strf
    dfSTRF = dfSTRF.drop(dfSTRF[dfSTRF['pval1']==-10000].index)#delete clusters those have too few spikes (<100) during strf
    ###sig
    dfSTRF_sig = dfSTRF[dfSTRF['pval1']<0.05]
    duplicates = dfSTRF_sig[dfSTRF_sig.duplicated('session_cls',keep=False)]
    sigSTRFUnites = list(duplicates['session_cls'].values)  
    dfSTRF_sig = dfSTRF_sig[dfSTRF_sig['session_cls'].isin(sigSTRFUnites)]
     
    ###notsig1
    dfSTRF_notsig = dfSTRF[(dfSTRF['pval1']>=0.05)|(dfSTRF['pval1'].isna())]   
    # dfSTRF_notsig = dfSTRF[(dfSTRF['pval1']>=0.05)|(dfSTRF['pval1'].isna())|(dfSTRF['pval1']==10000)]    
    duplicates = dfSTRF_notsig[dfSTRF_notsig.duplicated('session_cls',keep=False)]
    notsigSTRFUnites = list(duplicates['session_cls'].values)  
    dfSTRF_notsig = dfSTRF_notsig[dfSTRF_notsig['session_cls'].isin(notsigSTRFUnites)]
 
    # ###notsig2
    # dfSTRF_notsig = dfSTRF[dfSTRF['pval1']==10000]  #clusters those can not get strf  
    # duplicates = dfSTRF_notsig[dfSTRF_notsig.duplicated('session_cls',keep=False)]
    # notsigSTRFUnites = list(duplicates['session_cls'].values)  
    # dfSTRF_notsig = dfSTRF_notsig[dfSTRF_notsig['session_cls'].isin(notsigSTRFUnites)]
    # ###notsig3
    # dfSTRF_notsig = dfSTRF[dfSTRF['pval1']==-10000]  #clusters those have too few spikes (<100) during strf  
    # duplicates = dfSTRF_notsig[dfSTRF_notsig.duplicated('session_cls',keep=False)]
    # notsigSTRFUnites = list(duplicates['session_cls'].values)  
    # dfSTRF_notsig = dfSTRF_notsig[dfSTRF_notsig['session_cls'].isin(notsigSTRFUnites)]

    df_avMod_all_sig['STRFsig'] = np.nan

    df_avMod_all_sig.loc[df_avMod_all_sig['session_cls'].isin(sigSTRFUnites),'STRFsig'] = True
    df_avMod_all_sig.loc[df_avMod_all_sig['session_cls'].isin(notsigSTRFUnites),'STRFsig'] = False

    if STRF == 'all':
        df_avMod_all_sig = df_avMod_all_sig[df_avMod_all_sig['STRFsig'].isin([True,False])]       
    if STRF=='sig':         
        df_avMod_all_sig = df_avMod_all_sig[df_avMod_all_sig['session_cls'].isin(sigSTRFUnites)]
        print(Monkey+'  '+method+' neurons 0 drifting & sig strf: '+str(len(df_avMod_all_sig['session_cls'].unique())))      
    if STRF=='notsig':
        df_avMod_all_sig = df_avMod_all_sig[df_avMod_all_sig['session_cls'].isin(notsigSTRFUnites)] 
        print(Monkey+'  '+method+' neurons 0 drifting & notsig strf: '+str(len(df_avMod_all_sig['session_cls'].unique())))      
    if STRF=='addSTRFcol':
        pass
    #     dfSTRF_sig = dfSTRF[((dfSTRF['pval12']<0.05) & (dfSTRF['pval12']>-1))|((dfSTRF['pval1']<0.05) & (dfSTRF['pval1']>-1))|((dfSTRF['pval2']<0.05) & (dfSTRF['pval2']>-1))]
    #     df_avMod_all_sig.loc[:,'STRFsig'] = df_avMod_all_sig['session_cls'].isin(dfSTRF_sig['session_cls'])

    return df_avMod_all_sig,(clscolstr,dfSTRF_sig,dfSTRF_notsig)



