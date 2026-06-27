import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
import scipy.stats
from DL_functions_main import *
from pandas.tseries.offsets import MonthEnd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
pd.set_option('display.max_columns', 100)
from datetime import datetime
import matplotlib.gridspec as gridspec
from parameters import *
import time
import warnings
warnings.filterwarnings("ignore")


final_output = 'output'
if not os.path.exists(f"../{final_output}"):
    os.makedirs(f"../{final_output}")


oos_end = '2020-12-31'
T_OOS = len(pd.date_range(start=time_split, end=oos_end, freq='M'))
T_ALL = T_INS + T_OOS

bm_type = 'CAPM'

char_type = 'bond_equity_option'
if char_type == 'bond':
    list_char = list_bond
elif char_type == 'bond_equity_option':
    list_char = list_bond + list_equity + list_option
elif char_type == 'bond_equity':
    list_char = list_bond + list_equity

port_type = 'ew'


def maximum_drawdown(returns):
    cumulative_returns = (1 + returns).cumprod()  
    running_max = cumulative_returns.cummax()     
    drawdown = (cumulative_returns - running_max) / running_max  
    max_drawdown = drawdown.min()  
    return max_drawdown

def calc_MKT_alpha(input_y, input_x):

    x = np.array(input_x)
    y = np.array(input_y)
    
    X = sm.add_constant(x)
    
    try:
        model = sm.OLS(y, X).fit()
        intercept = model.params[0]
        p_value = model.pvalues[0]
        
        if p_value < 0.01:
            significance = "***"
        elif p_value < 0.05:
            significance = "**"
        elif p_value < 0.1:
            significance = "*"
        else:
            significance = ""
        
        intercept_100 = intercept * 100
        formatted_result = f"{intercept_100:.2f}{significance}"
        
        return formatted_result
        
    except Exception as e:
        return "NaN"


df_rf_all = pd.read_csv(f'../data/data_in_all_{update_date}/one_month_bill.csv',index_col=0)
df_rf_all.index = pd.to_datetime(df_rf_all.index.values)+MonthEnd(0)
df_rf_all = df_rf_all.loc[time_start:oos_end]

if port_type == 'ew':
    market_factor = 'MKTbond'
elif port_type == 'vw':
    market_factor = 'MKTbond_vw'
df_mkt_all = pd.read_csv(f'../data/data_in_all_{update_date}/Replicated_Bond_risk_factors_bbw4.csv',index_col=0)[market_factor].loc[time_start:oos_end]
df_mkt_all = np.array(df_mkt_all).reshape(-1,1)

dict_mkt = {}
dict_mkt['ew'] = pd.read_csv(f'../data/data_in_all_{update_date}/Replicated_Bond_risk_factors_bbw4.csv',index_col=0)['MKTbond'].loc[time_start:oos_end]
dict_mkt['ew'].index = pd.to_datetime(dict_mkt['ew'].index)
dict_mkt['vw'] = pd.read_csv(f'../data/data_in_all_{update_date}/Replicated_Bond_risk_factors_bbw4.csv',index_col=0)['MKTbond_vw'].loc[time_start:oos_end]
df_temp = pd.DataFrame(dict_mkt)

df_rppca_new = pd.concat([pd.read_csv(f'../data/rppca_{update_date}/rppca_ins.csv',header=None),
                          pd.read_csv(f'../data/rppca_{update_date}/rppca_oos.csv',header=None)])
df_rppca_new.columns = [f'rppca_{x+1}' for x in range(df_rppca_new.shape[1])]
df_rppca_new = df_rppca_new.reset_index(drop=True)[:T_ALL]
df_rppca = df_rppca_new.values


df_ipca_is = pd.read_csv(f'../data/ipca_{update_date}/ipca5_ins_{char_type}_{panel_num}_{update_date}.csv').set_index('Dates')
df_ipca_oos = pd.read_csv(f'../data/ipca_{update_date}/ipca5_oos_{char_type}_{panel_num}_{update_date}.csv').set_index('Unnamed: 0')
df_ipca = pd.concat([df_ipca_is,df_ipca_oos]).loc[time_start:oos_end].values

df_bbw = pd.read_csv(f'../data/data_in_all_{update_date}/Replicated_Bond_risk_factors_bbw4.csv',index_col=0)
df_bbw_all = df_bbw.loc[time_start:oos_end].values
df_bbw_all[:,0] = df_mkt_all.reshape(-1)


dict_bm = {}
dict_bm['CAPM'] = df_mkt_all.reshape(-1,1)
dict_bm['BBW'] = df_bbw_all.copy()
dict_bm['IPCA'] = df_ipca.copy()
dict_bm['ew'] = df_mkt_all.reshape(-1,1)
dict_bm['vw'] = dict_mkt['vw'].values.reshape(-1,1)



df_termdef = pd.read_csv(f'../data/data_in_all_{update_date}/term_and_default_factor.csv', index_col=0).loc[time_start:oos_end, ['Term Factor','Default Factor']]

df_ff5 = pd.read_csv(f'../data/data_in_all_{update_date}/bond_factors.csv',index_col=0).loc[time_start:oos_end, ['MKT','SMB','HML']]
df_ff5[['TERM','DEF']] = df_termdef.astype(float).values
df_ff5['MKT'] = dict_bm['ew'].copy()
df_ff5 = df_ff5.values


def TP2(df_return, lam):

    m, n = df_return.shape
    q = np.mean(df_return)
    p = np.cov(df_return.T)
    y = df_return

    w_1 = np.linalg.inv(p + np.eye(n) * lam).dot(q + np.ones(n)*lam)
    w_1 = w_1/np.sum(w_1)
    
    return w_1


# #######################################
# ##### Table 1: Summary Statistics #####
# #######################################

# start = time.time()

# df_charactecristic_impute = pd.read_feather(f'../data/data_in_all_{update_date}/characteristics_raw_selected{panel_num}_with_Equity_{update_date}_xret.feather').set_index('trd_exctn_dt').loc[:oos_end]

# df_charactecristic_impute = df_charactecristic_impute.merge(df_rf_all,left_on='trd_exctn_dt',right_index=True,how='left')

# df_charactecristic_impute['ExRet'] = df_charactecristic_impute['excess_ret']
# df_charactecristic_impute['log_size'] = np.log(df_charactecristic_impute['size'])
# df_charactecristic_impute['Public'] = df_charactecristic_impute['permno']>0
# df_charactecristic_impute['IG'] = df_charactecristic_impute['rating']<=10

# selected_stat = ['count','mean','std','50%']
# df_IG = df_charactecristic_impute[df_charactecristic_impute['IG']][['monthly_return_winsorized','ExRet','rating','duration','age','size']].describe().loc[selected_stat].T
# df_NIG = df_charactecristic_impute[~df_charactecristic_impute['IG']][['monthly_return_winsorized','ExRet','rating','duration','age','size']].describe().loc[selected_stat].T
# df_PUB = df_charactecristic_impute[df_charactecristic_impute['Public']][['monthly_return_winsorized','ExRet','rating','duration','age','size']].describe().loc[selected_stat].T
# df_PRI = df_charactecristic_impute[~df_charactecristic_impute['Public']][['monthly_return_winsorized','ExRet','rating','duration','age','size']].describe().loc[selected_stat].T
# df_ALL = df_charactecristic_impute[['monthly_return_winsorized','ExRet','rating','duration','age','size']].describe().loc[selected_stat].T

# pa_row = [x + item  for x in ['monthly_return_winsorized','ExRet','rating','duration','age','size'] for item in [' mean',' std',' median']]
# pa_row = ['Bond-month observations']+pa_row
# df_table1_pa =pd.DataFrame(index = pa_row, columns = ['ALL','IG','NIG','Public','Private']) 

# for i in range(5):
#     df_  = [df_ALL,df_IG,df_NIG,df_PUB,df_PRI][i]
#     count_value = df_.loc['monthly_return_winsorized']['count'].astype(int)
#     df_table1_pa.iloc[0,i] = f"{count_value:,}"

#     for item in df_.index:
#         if item == 'ExRet':
#             df_table1_pa.loc[item+' mean'].iloc[i] = np.round((df_.loc[item,'mean'])*100,2)
#             df_table1_pa.loc[item+' std'].iloc[i] = np.round((df_.loc[item,'std'])*100,2)
#         elif item == 'size':
#             df_table1_pa.loc[item+' mean'].iloc[i] = np.round((df_.loc[item,'mean'])/1000, 0)
#             df_table1_pa.loc[item+' std'].iloc[i] = np.round((df_.loc[item,'std'])/1000, 0)
#         else:
#             df_table1_pa.loc[item+' mean'].iloc[i] = np.round((df_.loc[item,'mean']),2)
#             df_table1_pa.loc[item+' std'].iloc[i] = np.round((df_.loc[item,'std']),2)

# df_table1_pa.dropna().to_csv(f'../{final_output}/Table_1_panelA.csv', index=True, encoding='utf_8_sig')


# df_charactecristic_impute['RRR'] = np.nan
# df_charactecristic_impute.loc[df_charactecristic_impute['rating'] == 1,'RRR'] = 'AAA'
# df_charactecristic_impute.loc[(df_charactecristic_impute['rating'] >1) & (df_charactecristic_impute['rating'] <=4),'RRR'] = 'AA'
# df_charactecristic_impute.loc[(df_charactecristic_impute['rating'] >4) & (df_charactecristic_impute['rating'] <=7),'RRR'] = 'A'
# df_charactecristic_impute.loc[(df_charactecristic_impute['rating'] >7) & (df_charactecristic_impute['rating'] <=10),'RRR'] = 'B'
# df_charactecristic_impute.loc[df_charactecristic_impute['rating'] >10,'RRR'] = 'Junk'
# df_charactecristic_impute['MMM'] = np.nan
# for i in range(10,1,-1):
    
#     df_charactecristic_impute.loc[df_charactecristic_impute['time2maturity'] <=i,'MMM'] = i
#     if i==10:
#         df_charactecristic_impute.loc[df_charactecristic_impute['time2maturity'] >i,'MMM'] = 11

# df_table1_pb = df_charactecristic_impute.groupby(['RRR','MMM']).count().reset_index().pivot(index='MMM', columns='RRR', values=['monthly_return'])
# df_table1_pb.columns = df_table1_pb.columns.droplevel(0)
# df_table1_pb = df_table1_pb/np.sum(np.sum(df_table1_pb))
# df_table1_pb['ALL_c'] = df_table1_pb.sum(axis=1)
# df_table1_pb.loc['ALL_r',:] = df_table1_pb.sum(axis=0)
# df_table1_pb = df_table1_pb[['AAA','AA','A','B','Junk','ALL_c']].round(4)*100
# df_table1_pb.to_csv(f'../{final_output}/Table_1_panelB.csv', index=True, encoding='utf_8_sig')


# df_table1_pc = df_charactecristic_impute.groupby(['RRR','Public']).count().reset_index().pivot(index='Public', columns='RRR', values=['monthly_return'])
# df_table1_pc = df_table1_pc/np.sum(np.sum(df_table1_pc))
# df_table1_pc.columns = df_table1_pc.columns.droplevel(0)
# df_table1_pc['ALL_c'] = df_table1_pc.sum(axis=1)
# df_table1_pc.loc['ALL_r',:] = df_table1_pc.sum(axis=0)
# df_table1_pc = df_table1_pc[['AAA','AA','A','B','Junk','ALL_c']].round(4)*100
# df_table1_pc.to_csv(f'../{final_output}/Table_1_panelC.csv', index=True, encoding='utf_8_sig')

# print(f"Table 1 run time: {round((time.time() - start), 2)} seconds \n")

###########################################################
##### Figure A2: Softmax Ranking: a1 = 50, and a2 = 8 #####
###########################################################

start = time.time()

ew_dir = './results_{}/ew_with_batch/{}'.format(char_type, bm_type)
df_best = pd.read_csv(f'{ew_dir}/best_summary_seed{seed}.csv')

def MVP_weight(df_):
    lam = 1e-15
    p = df_.shape[1]
    dict_weight = np.zeros([T_OOS,p])
    for i in range(T_INS,T_ALL):
        df_2 = df_[i-T_INS:i,0:]

        if p == 1:
            w = np.ones([1])
        else:
            sign_df = df_2
            
            w = np.linalg.inv(np.cov(sign_df.T)+np.eye(p) * lam) @ (np.mean(sign_df,axis=0)+np.ones(p) * lam)

        dict_weight[i-T_INS] = w/np.abs(w.sum())
    return dict_weight


dict_capm_uncond = {}
dict_capm_cond = {}
dict_capm_is_uncond = {}
dict_capm_oos_uncond = {}
dict_capm_is_cond = {}
dict_capm_oos_cond = {}

dict_loss = {}

dict_df = {}
for port_type in ['vw','ew']:

    dict_capm_uncond[port_type] = {}
    dict_capm_cond[port_type] = {}
    dict_capm_is_uncond[port_type] = {}
    dict_capm_oos_uncond[port_type] = {}
    dict_capm_is_cond[port_type] = {}
    dict_capm_oos_cond[port_type] = {}

    dict_loss[port_type] = {}

    dict_df[port_type] = {}
    output_dir = './results_{}/{}_with_batch/{}'.format(char_type, port_type, bm_type)

    for n_layer in range(1, max_layers+1):
        this_params = df_best[(df_best['n_layer'] == n_layer)]
        learning_rate = this_params['learning_rate'].values.tolist()[0]
        dropout_rate = this_params['dropout_rate'].values.tolist()[0]
        a1 = this_params['a1'].values.tolist()[0]
        a2 = this_params['a2'].values.tolist()[0]
        for n_deep_factors in range(1, deep_factors_num+1):
            
            file_name_uncond = '{}/factor_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.txt'.format(
                        output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed)

            loss_file = '{}/loss_path_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.txt'.format(
                        output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed)
        
            temp_fix = pd.read_table(file_name_uncond,sep='\\s+',header=None)
            sign = np.sign(np.mean(temp_fix[:T_INS]))
            temp_fix = sign*temp_fix
            temp_fix = temp_fix.values[:T_ALL]

            key_name = '{}_{}'.format(n_layer,n_deep_factors)
            last_key_name = '{}_{}'.format(n_layer,n_deep_factors-1)
            sr_uncond = np.mean(temp_fix[T_INS:T_ALL,])/np.std(temp_fix[T_INS:T_ALL,])*np.sqrt(12)

            if n_deep_factors == 1:
                dict_df[port_type][key_name] = np.concatenate([dict_bm[port_type],temp_fix],axis=1)
                dict_capm_uncond[port_type][key_name] = np.concatenate([dict_bm[port_type],temp_fix],axis=1)

            else:
                dict_df[port_type][key_name] = np.concatenate([dict_df[port_type][last_key_name],temp_fix],axis=1)
                dict_capm_uncond[port_type][key_name] = np.concatenate([dict_capm_uncond[port_type][last_key_name],temp_fix],axis=1)

            dict_capm_is_uncond[port_type][key_name] = dict_capm_uncond[port_type][key_name][:T_INS,:]
            dict_capm_oos_uncond[port_type][key_name] = dict_capm_uncond[port_type][key_name][T_INS:T_ALL,:]

            dict_loss[port_type][key_name] = pd.read_table(loss_file,sep='\\s+',header=None)


dict_return = {}
dict_SR = {}
dict_pvalue = {}
list_column = [bm_type,'D=1','D=2','D=3']
df_sr = {}
df_DF_mdd = {}
df_DF_alpha = {}
dict_weight = {}


for port_type in ['vw','ew']:
    dict_weight[port_type] = MVP_weight(dict_bm[port_type])
    dict_return['IS'] = {}
    dict_return['OOS'] = {}
    dict_return['IS'][port_type] = dict_bm[port_type][:T_INS]
    dict_return['OOS'][port_type] = dict_bm[port_type][T_INS:T_ALL]

    dict_pvalue[port_type] = {}
    df_sr[port_type] = {}
    df_DF_mdd[port_type] = {}
    df_DF_alpha[port_type] = {}
    dict_SR[port_type] = {}
    for time_period in ['IS','OOS']:
        df_sr[port_type][time_period] = pd.DataFrame(index =['L=1','L=2','L=3'],columns = list_column)
        df_DF_mdd[port_type][time_period] = pd.DataFrame(index =['L=1','L=2','L=3'],columns = list_column)
        df_DF_alpha[port_type][time_period] = pd.DataFrame(index =['L=1','L=2','L=3'],columns = list_column)
        
        port_rtn = dict_return[time_period][port_type]@dict_weight[port_type][0]
        dict_SR[port_type][time_period] = np.mean(port_rtn)/np.std(port_rtn)*np.sqrt(12)

        df_DF_mdd[port_type][time_period].iloc[:,0] = maximum_drawdown(pd.Series(port_rtn.reshape(-1)))*(-100)
        df_DF_alpha[port_type][time_period].iloc[:,0] = np.nan

        dict_pvalue[port_type][time_period] = {}
        df_sr[port_type][time_period].iloc[:,0] = dict_SR[port_type][time_period]


this_dict_SR = {}
dict_capm_norm = {}
dict_timeseries = {}
dict_timeseries_norm = {}
for port_type in ['vw','ew']:
    this_dict_df = dict_df[port_type]
    this_df_sr = df_sr[port_type]
    this_df_DF_mdd = df_DF_mdd[port_type]
    this_df_DF_alpha = df_DF_alpha[port_type]
    market_SR_IS = dict_SR[port_type]['IS']
    market_SR_OOS = dict_SR[port_type]['OOS']

    this_dict_SR[port_type] = {}
    dict_capm_norm[port_type] = {}
    dict_timeseries[port_type] = {}
    dict_timeseries_norm[port_type] = {}
    dict_weight[port_type] = {}

    for time_period in ['IS','OOS']:
        this_dict_SR[port_type][time_period] = {}
        dict_pvalue[port_type][time_period] = pd.DataFrame(index =['L=1','L=2','L=3'],columns = list_column)
        
        for ii in range(1, max_layers+1):
            for jj in range(1, deep_factors_num+1):

                key = '{}_{}'.format(ii,jj)
                w_temp = np.linalg.inv(np.cov(this_dict_df[key][:T_INS].T))@np.mean(this_dict_df[key][:T_INS],axis=0)
                w_temp = w_temp / np.abs(w_temp).sum()

                dict_weight[port_type][key] = w_temp
                df_ = this_dict_df[key]
                dict_capm_norm[port_type][key] = df_.copy()
                
                if time_period == 'IS':

                    T = T_INS
                    df_ = df_[:T_INS]
                    p = df_.shape[1]
                    dict_return[time_period][key] = df_ @ dict_weight[port_type][key]
                    
                    dict_timeseries_norm[port_type][key] = dict_return[time_period][key]

                    this_dict_SR[port_type][time_period][key] = np.mean(dict_return[time_period][key])/np.std(dict_return[time_period][key])*np.sqrt(12)
                    this_df_sr[time_period].iloc[ii-1,jj] = this_dict_SR[port_type][time_period][key]
                    this_df_DF_mdd[time_period].iloc[ii-1,jj] = maximum_drawdown(pd.Series(dict_return[time_period][key]))*(-100)
                    this_df_DF_alpha[time_period].iloc[ii-1,jj] = calc_MKT_alpha(dict_return[time_period][key], df_[:,0])

                    if jj>=2:
                        D = 1 
                        F= T/D *(T-p-D)/(T-p-1)*(this_dict_SR[port_type][time_period][key]**2-this_dict_SR[port_type][time_period]['{}_{}'.format(ii,jj-1)]**2)/(1+this_dict_SR[port_type][time_period]['{}_{}'.format(ii,jj-1)]**2)
                        dict_pvalue[port_type][time_period].iloc[ii-1,jj] = 1-scipy.stats.f.cdf(F,D,T-p-D)
                    else:
                        D = 1
                        F= T/D *(T-p-D)/(T-p-1)*(this_dict_SR[port_type][time_period][key]**2-market_SR_IS**2)/(1+market_SR_IS**2)
                        dict_pvalue[port_type][time_period].iloc[ii-1,jj] = 1-scipy.stats.f.cdf(F,D,T-p-D)
                        
                        
                    if jj>=2:
                        D = jj 
                        F= T/p *(T-p-D)/(T-p-1)*(this_dict_SR[port_type][time_period][key]**2-this_dict_SR[port_type][time_period]['{}_{}'.format(ii,jj-1)]**2)/(1+this_dict_SR[port_type][time_period]['{}_{}'.format(ii,jj-1)]**2)
                        dict_pvalue[port_type][time_period].iloc[ii-1,jj] = 1-scipy.stats.f.cdf(F,p,T-p-D)
                    else:
                        D = 1
                        F= T/p *(T-p-D)/(T-p-1)*(this_dict_SR[port_type][time_period][key]**2-market_SR_IS**2)/(1+market_SR_IS**2)
                        dict_pvalue[port_type][time_period].iloc[ii-1,jj] = 1-scipy.stats.f.cdf(F,p,T-p-D)
                        
                        
                
                elif time_period == 'OOS':

                    T = T_OOS
                    df_2 = df_[T_INS:T_ALL,]
                    p = df_2.shape[1]
                    sign_df = np.sign(np.mean(df_2,axis=0))*df_2
                    dict_return[time_period][key] =  sign_df @ dict_weight[port_type][key]
                    
                    
                    dict_timeseries_norm[port_type][key] = np.concatenate([dict_timeseries_norm[port_type][key],dict_return[time_period][key]])
                    
                    dict_timeseries[port_type][key] = dict_timeseries_norm[port_type][key]
                    this_dict_SR[port_type][time_period][key] = np.mean(dict_return[time_period][key])/np.std(dict_return[time_period][key])*np.sqrt(12)
                    this_df_sr[time_period].iloc[ii-1,jj] = this_dict_SR[port_type][time_period][key]
                    this_df_DF_mdd[time_period].iloc[ii-1,jj] = maximum_drawdown(pd.Series(dict_return[time_period][key]))*(-100)
                    this_df_DF_alpha[time_period].iloc[ii-1,jj] = calc_MKT_alpha(dict_return[time_period][key], sign_df[:,0])
                    
                    if jj>=2:
                        D = 1
                        F= T/D *(T-p-D)/(T-p-1)*(this_dict_SR[port_type][time_period][key]**2-this_dict_SR[port_type][time_period]['{}_{}'.format(ii,jj-1)]**2)/(1+this_dict_SR[port_type][time_period]['{}_{}'.format(ii,jj-1)]**2)
                        dict_pvalue[port_type][time_period].iloc[ii-1,jj] = 1-scipy.stats.f.cdf(F,D,T-p-D)
                    else:
                        D = 1
                        F= T/D *(T-p-D)/(T-p-1)*(this_dict_SR[port_type][time_period][key]**2-market_SR_OOS**2)/(1+market_SR_OOS**2)
                        dict_pvalue[port_type][time_period].iloc[ii-1,jj] = 1-scipy.stats.f.cdf(F,D,T-p-D)
                    if jj>=2:
                        D = jj 
                        F= T/p *(T-p-D)/(T-p-1)*(this_dict_SR[port_type][time_period][key]**2-this_dict_SR[port_type][time_period]['{}_{}'.format(ii,jj-1)]**2)/(1+this_dict_SR[port_type][time_period]['{}_{}'.format(ii,jj-1)]**2)
                        dict_pvalue[port_type][time_period].iloc[ii-1,jj] = 1-scipy.stats.f.cdf(F,p,T-p-D)
                    else:
                        D = 1
                        F= T/p *(T-p-D)/(T-p-1)*(this_dict_SR[port_type][time_period][key]**2-market_SR_OOS**2)/(1+market_SR_OOS**2)
                        dict_pvalue[port_type][time_period].iloc[ii-1,jj] = 1-scipy.stats.f.cdf(F,p,T-p-D)
                    
count = 0
for df_ in [df_bbw_all,df_ipca,df_ff5,df_rppca]:
    # df_rppca
    if count == 0:
        key = 'BBW'
    elif count == 1:
        key = 'IPCA'
    elif count == 2:
        key = 'FF5'
    else:
        key = 'RPPCA'
    count += 1
    
    weight = MVP_weight(df_)

    df_is = df_[:T_INS,:]

    dict_timeseries[key] = df_is @ weight[0]
    
    df_2 = df_[T_INS:T_ALL,0:]

    dict_timeseries[key] =  np.concatenate([dict_timeseries[key],df_2 @ weight[0]])
    
    dict_timeseries_norm[key] = dict_timeseries[key].copy()


dict_temp = {}
for n_layer in range(1, max_layers+1):
    this_params = df_best[(df_best['n_layer'] == n_layer)]
    learning_rate = this_params['learning_rate'].values.tolist()[0]
    dropout_rate = this_params['dropout_rate'].values.tolist()[0]
    a1 = this_params['a1'].values.tolist()[0]
    a2 = this_params['a2'].values.tolist()[0]

    for n_deep_factors in range(1, 2):

        file_name_uncond = '{}/weights_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.txt'.format(
                    output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed)

        temp_fix = pd.read_table(file_name_uncond,sep='\\s+',header=None).values

        key_name = '{}_{}'.format(n_layer,n_deep_factors)

        dict_temp['{}_layer'.format(n_layer)] = temp_fix

for key in dict_temp.keys():
    plt.plot(np.sort(dict_temp[key][0]),label=key)
    if np.max(np.abs(dict_temp[key][0])) >= 0.005:
        plt.ylim(-0.005, 0.005)
plt.legend()

# plt.savefig(f'../{final_output}/Figure_A2.pdf', dpi=100, bbox_inches='tight')
plt.show()
plt.close()

# print(f"Figure A2 run time: {round((time.time() - start), 2)} seconds \n")

################################################
##### Table 2: Deep Corporate Bond Factors #####
################################################

start = time.time()

list_uncond_best_capm = ['1_1','2_1','3_1']
best_factor_choose = int(np.where(df_sr['ew']['OOS']['D=1'] == df_sr['ew']['OOS']['D=1'].max())[0].tolist()[0]+1)
df_sel_ft = dict_df['ew'][f'{best_factor_choose}_1']

lam = 1e-15
iter_index = (['MKT$_{EW}$','R$_{s}^1$','R$_{s}^2$','R$_{s}^3$']+
              ['MKT$_{VW}$','R$_{d}^1$','R$_{d}^2$','R$_{d}^3$']+
              ['FF5$^{Opt}$','IPCA$^{Opt}$','RP-PCA$^{Opt}$']+
              ['FF5(+DF$_{EW}$)$^{Opt}$','IPCA(+DF$_{EW})$^{Opt}$','RP-PCA(+DF$_{EW})$^{Opt}$']+
              ['DTP$_{EW1}$', 'DTP$_{EW2}$', 'DTP$_{EW3}$']+
              ['$DTP_{VW1}$', '$DTP_{VW2}$', '$DTP_{VW3}$'])
df_stat = pd.DataFrame(index = iter_index, columns = ['Mean','Min','Max','tstat','Std','SR','Max DD'])
df_pval = df_stat.copy()

def maximum_drawdown(returns):
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    return max_drawdown

def output_tp(df):
    df_ = df.copy()
    p = df_.shape[1]
    df_ins = df_[:T_INS]
    w = (np.mean(df_ins,axis=0)+np.ones(p) * lam)@(np.linalg.inv(np.cov(df_ins.T))+np.eye(p) * lam)
    
    w = w/np.sum(np.abs(w))
    tp = df_@w
    return tp

count2 = 0
for count1,df_ in enumerate([dict_bm['ew']]+
                            [dict_df['ew'][x][:,-1].reshape(-1,1) for x in np.array(list_uncond_best_capm)]+
                            [dict_bm['vw']]+
                            [dict_df['vw'][x][:,-1].reshape(-1,1) for x in np.array(list_uncond_best_capm)]+
                            [output_tp(df_ff5)]+
                            [output_tp(df_ipca)]+
                            [output_tp(df_rppca)]+
                            [output_tp(np.concatenate([df_ff5, df_sel_ft[:,[1]]], axis=1))]+
                            [output_tp(np.concatenate([df_ipca, df_sel_ft[:,[1]]], axis=1))]+
                            [output_tp(np.concatenate([df_rppca, df_sel_ft[:,[1]]], axis=1))]+
                            [output_tp(np.concatenate([dict_bm['ew'],dict_df['ew']['1_1'][:,[-1]]],axis=1))]+
                            [output_tp(np.concatenate([dict_bm['ew'],dict_df['ew']['2_1'][:,[-1]]],axis=1))]+
                            [output_tp(np.concatenate([dict_bm['ew'],dict_df['ew']['3_1'][:,[-1]]],axis=1))]+
                            [output_tp(np.concatenate([dict_bm['vw'],dict_df['vw']['1_1'][:,[-1]]],axis=1))]+
                            [output_tp(np.concatenate([dict_bm['vw'],dict_df['vw']['2_1'][:,[-1]]],axis=1))]+
                            [output_tp(np.concatenate([dict_bm['vw'],dict_df['vw']['3_1'][:,[-1]]],axis=1))]):
    count2 = 0
    whether_calc_p_value = False


    df_for_mdd = df_.copy()

    var_ = iter_index[count1]
    for y in [df_[T_INS:T_ALL]]:
        Mean = np.mean(y*100)
        Std = np.std(y*100)
        
        cum_ret = (1+y).cumprod()
        rolling_max = np.maximum.accumulate(cum_ret)
        rolling_max_12window = pd.DataFrame(cum_ret).rolling(12,min_periods=1).max().to_numpy().reshape(-1)
        max_dd = maximum_drawdown(pd.Series(y.reshape(-1)))*(-100)
        
        newey_west_L = int(4*(len(y)/100)**(2/9))
        
        result = sm.OLS(y*100,np.ones(len(y))*y.mean()*100).fit(cov_type='HAC',cov_kwds={'maxlags':newey_west_L})
        tstats = result.tvalues[0]
        pval = result.pvalues[0]
        
        
        df_stat.loc[var_,'Mean'] = Mean
        df_stat.loc[var_,'Min'] = np.min(y*100)
        df_stat.loc[var_,'Max'] = np.max(y*100)
        df_stat.loc[var_,'tstat'] = tstats
        df_stat.loc[var_,'Std'] = Std
        tmp_sr = Mean/Std*np.sqrt(12)
        df_stat.loc[var_,'SR'] = np.round(tmp_sr,5)
        df_stat.loc[var_,'Max DD'] = np.round(max_dd,5)

        # Test significance of the Sharpe ratio improvement following Barillas and Shanken (2017)
        D = 1
        T = len(y)
        p = 2

        if var_ == 'FF5(+DF$_{EW}$)$^{Opt}$':
            whether_calc_p_value = True
            compared_sr = (df_stat.loc['FF5$^{Opt}$','Mean'])/(df_stat.loc['FF5$^{Opt}$','Std'])*np.sqrt(12)
        elif var_ == 'IPCA(+DF$_{EW})$^{Opt}$':
            whether_calc_p_value = True
            compared_sr = (df_stat.loc['IPCA$^{Opt}$','Mean'])/(df_stat.loc['IPCA$^{Opt}$','Std'])*np.sqrt(12)
        elif var_ == 'RP-PCA(+DF$_{EW})$^{Opt}$':
            whether_calc_p_value = True
            compared_sr = (df_stat.loc['RP-PCA$^{Opt}$','Mean'])/(df_stat.loc['RP-PCA$^{Opt}$','Std'])*np.sqrt(12)
        elif var_ in ['DTP$_{EW1}$','DTP$_{EW2}$','DTP$_{EW3}$']:
            whether_calc_p_value = True
            compared_sr = (df_stat.loc['MKT$_{EW}$','Mean'])/(df_stat.loc['MKT$_{EW}$','Std'])*np.sqrt(12)
        elif var_ in ['$DTP_{VW1}$', '$DTP_{VW2}$', '$DTP_{VW3}$']:
            whether_calc_p_value = True
            compared_sr = (df_stat.loc['MKT$_{VW}$','Mean'])/(df_stat.loc['MKT$_{VW}$','Std'])*np.sqrt(12)

        if whether_calc_p_value == True:
            F= T/D *(T-p-D)/(T-p-1)*(tmp_sr**2-compared_sr**2)/(1+compared_sr**2)
            df_stat.loc[var_,'p_value'] = 1-scipy.stats.f.cdf(F,D,T-p-D)
        else:
            df_stat.loc[var_,'p_value'] = np.nan

    count1 =count1+1


df_table2_B = df_stat.applymap(lambda x: ('%.2f' % x))
df_table2_B = '$' +df_table2_B +'$'
df_table2_B['Mean'] = df_table2_B['Mean'].astype(str)+'('+ df_table2_B['tstat'].astype(str) +')'
df_table2_B = df_table2_B[['Mean','SR','Min','Max','Max DD']]
df_table2_B_ew = df_table2_B.loc[['R$_{s}^1$','R$_{s}^2$','R$_{s}^3$']]
df_table2_B_ew.index = ['R$_{s}^1$','R$_{s}^2$','R$_{s}^3$']
df_table2_B_ew[''] = ''
df_table2_B_vw = df_table2_B.loc[['R$_{d}^1$','R$_{d}^2$','R$_{d}^3$']]
df_table2_B_vw.index = df_table2_B_ew.index
df_table2_B = pd.concat([df_table2_B_ew, df_table2_B_vw], axis=1)

df_table2_B.to_csv(f'../{final_output}/Table_2_panelB.csv', index=False, encoding='utf_8_sig')

print(f"Table 2 Panel B run time: {round((time.time() - start), 2)} seconds \n")

df_stat.reset_index()

######################################################################
##### Table 3: Deep Tangency Portfolios and Competing Portfolios #####
######################################################################

df_table3 = df_stat.applymap(lambda x: ('%.2f' % x))
df_table3['p_value'] = df_stat['p_value'].copy()
df_table3_A_ew = df_table3.loc[['MKT$_{EW}$','DTP$_{EW1}$','DTP$_{EW2}$','DTP$_{EW3}$']]
df_table3_A_ew['SR'][df_table3_A_ew['p_value']<=0.1] = df_table3_A_ew['SR'][df_table3_A_ew['p_value']<=0.1]+'*'
df_table3_A_ew['SR'][df_table3_A_ew['p_value']<=0.05] = df_table3_A_ew['SR'][df_table3_A_ew['p_value']<=0.05]+'*'
df_table3_A_ew['SR'][df_table3_A_ew['p_value']<=0.01] = df_table3_A_ew['SR'][df_table3_A_ew['p_value']<=0.01]+'*'
df_table3_A_ew = df_table3_A_ew[['SR','Max DD']].T
df_table3_A_ew.columns = ['MKT','$L_{1}$','$L_{2}$','$L_{3}$']
df_table3_A_ew.index = ['EW_SR','EW_MDD']

df_table3_A_vw = df_table3.loc[['MKT$_{VW}$','$DTP_{VW1}$', '$DTP_{VW2}$', '$DTP_{VW3}$']]
df_table3_A_vw['SR'][df_table3_A_vw['p_value']<=0.1] = df_table3_A_vw['SR'][df_table3_A_vw['p_value']<=0.1]+'*'
df_table3_A_vw['SR'][df_table3_A_vw['p_value']<=0.05] = df_table3_A_vw['SR'][df_table3_A_vw['p_value']<=0.05]+'*'
df_table3_A_vw['SR'][df_table3_A_vw['p_value']<=0.01] = df_table3_A_vw['SR'][df_table3_A_vw['p_value']<=0.01]+'*'
df_table3_A_vw = df_table3_A_vw[['SR','Max DD']].T
df_table3_A_vw.columns = ['MKT','$L_{1}$','$L_{2}$','$L_{3}$']
df_table3_A_vw.index = ['VW_SR','VW_MDD']
df_table3_A = pd.concat([df_table3_A_ew, df_table3_A_vw], axis=0)

df_table3_B_SR = df_table3.loc[['FF5$^{Opt}$','IPCA$^{Opt}$','RP-PCA$^{Opt}$']]
df_table3_B_SR = df_table3_B_SR[['SR','Max DD']].T
df_table3_B_SR.columns = ['FF5','IPCA5','RP-PCA5']
df_table3_B_SR.index = ['SR','MDD']

df_table3_B_SR_DF = df_table3.loc[['FF5(+DF$_{EW}$)$^{Opt}$','IPCA(+DF$_{EW})$^{Opt}$','RP-PCA(+DF$_{EW})$^{Opt}$']]
df_table3_B_SR_DF['SR'][df_table3_B_SR_DF['p_value']<=0.1] = df_table3_B_SR_DF['SR'][df_table3_B_SR_DF['p_value']<=0.1]+'*'
df_table3_B_SR_DF['SR'][df_table3_B_SR_DF['p_value']<=0.05] = df_table3_B_SR_DF['SR'][df_table3_B_SR_DF['p_value']<=0.05]+'*'
df_table3_B_SR_DF['SR'][df_table3_B_SR_DF['p_value']<=0.01] = df_table3_B_SR_DF['SR'][df_table3_B_SR_DF['p_value']<=0.01]+'*'
df_table3_B_DF = df_table3_B_SR_DF[['SR','Max DD']].T
df_table3_B_DF.columns = ['FF5','IPCA5','RP-PCA5']
df_table3_B_DF.index = ['SR (+DF)','MDD (+DF)']
df_table3_B = pd.concat([df_table3_B_SR, df_table3_B_DF], axis=0)

df_table3_A[''] = ''
df_table3 = pd.concat([df_table3_A.reset_index(), df_table3_B.reset_index()], axis=1)
df_table3.to_csv(f'../{final_output}/Table_3.csv', index=False, encoding='utf_8_sig')

print(f"Table 3 run time: {round((time.time() - start), 2)} seconds \n")

######################################################################
##### Figure 2: Correlations between the Market and Deep Factors #####
######################################################################

start = time.time()

port_choose = 'ew'

df_cumret = pd.DataFrame([(1+df_mkt_all).cumprod(),
                          (1+dict_capm_uncond[port_choose][list_uncond_best_capm[0]][:,1]).cumprod(),
                          (1+dict_capm_uncond[port_choose][list_uncond_best_capm[1]][:,1]).cumprod(),
                          (1+dict_capm_uncond[port_choose][list_uncond_best_capm[-1]][:,1]).cumprod()]+
                         [(1+dict_timeseries[port_choose][x]).cumprod() for x in list_uncond_best_capm]+
                         [(1+dict_timeseries[x]).cumprod() for x in ['BBW','IPCA','FF5','RPPCA']])
df_cumret = df_cumret.T
df_cumret.index = pd.to_datetime(df_bbw.loc[time_start:oos_end].index)
df_cumret.columns = ['MKT','$R_d^1$','$R_d^2$','$R_d^3$','$R_1^{opt}$','$R_2^{opt}$','$R_3^{opt}$','BBW','IPCA','FF5','RPPCA']

df_cumret_oos = pd.DataFrame([(1+df_mkt_all[T_INS:T_ALL]).cumprod(),
                              (1+dict_capm_uncond[port_choose][list_uncond_best_capm[0]][T_INS:T_ALL,1]).cumprod(),
                              (1+dict_capm_uncond[port_choose][list_uncond_best_capm[1]][T_INS:T_ALL,1]).cumprod(),
                              (1+dict_capm_uncond[port_choose][list_uncond_best_capm[-1]][T_INS:T_ALL,1]).cumprod()]+
                             [(1+dict_timeseries[port_choose][x][T_INS:T_ALL]).cumprod() for x in list_uncond_best_capm]+
                             [(1+dict_timeseries[x][T_INS:T_ALL]).cumprod() for x in ['BBW','IPCA','FF5','RPPCA']]).T
df_cumret_oos.index = pd.to_datetime(df_bbw.loc[time_split:oos_end].index)
df_cumret_oos.columns = ['MKT','$R_d^1$','$R_d^2$','$R_d^3$','$R_1^{opt}$','$R_2^{opt}$','$R_3^{opt}$','BBW','IPCA','FF5','RPPCA']

df_ret = pd.DataFrame([df_mkt_all.reshape(-1),
                       dict_capm_uncond[port_choose][list_uncond_best_capm[0]][:,1],
                       dict_capm_uncond[port_choose][list_uncond_best_capm[1]][:,1],
                       dict_capm_uncond[port_choose][list_uncond_best_capm[-1]][:,1]]+
                      [dict_timeseries[port_choose][x] for x in list_uncond_best_capm]+
                      [dict_timeseries[x] for x in ['BBW','IPCA','FF5','RPPCA']])
df_ret = df_ret.T
df_ret.index = pd.to_datetime(df_bbw.loc[time_start:oos_end].index)
df_ret.columns = ['MKT','$R_d^1$','$R_d^2$','$R_d^3$','$R_1^{opt}$','$R_2^{opt}$','$R_3^{opt}$','BBW','IPCA','FF5','RPPCA']

df_ret_oos = pd.DataFrame([df_mkt_all[T_INS:T_ALL].reshape(-1),
                           dict_capm_uncond[port_choose][list_uncond_best_capm[0]][T_INS:T_ALL,1],
                           dict_capm_uncond[port_choose][list_uncond_best_capm[1]][T_INS:T_ALL,1],
                           dict_capm_uncond[port_choose][list_uncond_best_capm[-1]][T_INS:T_ALL,1]]+
                          [dict_timeseries[port_choose][x][T_INS:T_ALL] for x in list_uncond_best_capm]+
                          [dict_timeseries[x][T_INS:T_ALL] for x in ['BBW','IPCA','FF5','RPPCA']]).T
df_ret_oos.index = pd.to_datetime(df_bbw.loc[time_split:oos_end].index)
df_ret_oos.columns = ['MKT','$R_d^1$','$R_d^2$','$R_d^3$','$R_1^{opt}$','$R_2^{opt}$','$R_3^{opt}$','BBW','IPCA','FF5','RPPCA']

df_ret_norm = df_ret/(df_ret.std()*np.sqrt(12)) * 0.1
df_ret_oos_norm = df_ret_oos/(df_ret_oos.std()*np.sqrt(12))*0.1

df_cumret_norm = (df_ret_norm+1).cumprod()
df_cumret_oos_norm = (df_ret_oos_norm+1).cumprod()
dict_capm_uncond[port_choose].keys()

fig_size = 9
fig, axes = plt.subplots(1, 2, figsize=(fig_size*2.3*0.8,fig_size*0.8))

plt.subplot(1,2,1)
x_mkt = df_mkt_all[:T_INS]
y_df = dict_capm_uncond[port_choose][list_uncond_best_capm[best_factor_choose-1]][:T_INS,1]
plt.scatter(x_mkt/x_mkt.std()*0.1/np.sqrt(12),y_df/y_df.std()*0.1/np.sqrt(12), s=fig_size*3, color='dodgerblue')
plt.axhline(y=0, linewidth=1.5, label='horizontal-line', alpha=0.8, color='black')
plt.axvline(x=0, linewidth=1.5, label='horizontal-line', alpha=0.8, color='black')
plt.xlabel('MKT', size=fig_size*2)
plt.ylabel('Deep Factor', size=fig_size*2) 
plt.tick_params(axis='both', which='major', labelsize=fig_size*2)
corr_in = np.corrcoef((x_mkt/x_mkt.std()*0.1/np.sqrt(12)).ravel(), (y_df/y_df.std()*0.1/np.sqrt(12)).ravel())[0,1]
plt.title('A. In Sample', size=fig_size*2, fontweight="bold")

plt.subplot(1,2,2)
x_mkt = df_mkt_all[T_INS:T_ALL]
y_df = dict_capm_uncond[port_choose][list_uncond_best_capm[best_factor_choose-1]][T_INS:T_ALL,1]
plt.scatter(x_mkt/x_mkt.std()*0.1/np.sqrt(12), y_df/y_df.std()*0.1/np.sqrt(12), s=fig_size*3, color='dodgerblue')
plt.axhline(y=0, linewidth=1.5, label='horizontal-line', alpha=0.8, color='black')
plt.axvline(x=0, linewidth=1.5, label='horizontal-line', alpha=0.8, color='black')
plt.xlabel('MKT', size=fig_size*2)
plt.ylabel('Deep Factor', size=fig_size*2) 
plt.tick_params(axis='both', which='major', labelsize=fig_size*2)
corr_oos = np.corrcoef((x_mkt/x_mkt.std()*0.1/np.sqrt(12)).ravel(), (y_df/y_df.std()*0.1/np.sqrt(12)).ravel())[0,1]
plt.title('B. Out of Sample', size=fig_size*2, fontweight="bold")

plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.3, 
                    hspace=0.2)
plt.savefig(f'../{final_output}/Figure_2.pdf', dpi=100, bbox_inches = 'tight')
plt.show()
plt.close()

print(f"Figure 2 run time: {round((time.time() - start), 2)} seconds \n")


########################################################
##### Figure 3: Time Series and Cumulative Returns #####
########################################################

start = time.time()

fig, axes = plt.subplots(2, 1, figsize=(13, 10))

ax1 = axes[0]

x = df_rf_all.index[T_INS:T_ALL]
y1 = df_sel_ft[T_INS:T_ALL,1] 
y2 = df_sel_ft[T_INS:T_ALL,0] 
y4 = y2/y2.std()*0.1/np.sqrt(12)
y3 = y1/y1.std()*0.1/np.sqrt(12) 

ax1.bar(x, y3, alpha=0.8, width=20, label='Deep Factor')
ax1.plot(x, y4, 'r-', linewidth=4, label='Market Factor')

ax1.axvspan('2020-02-28', '2020-04-30', alpha=0.3, color='grey')
ax1.tick_params(axis='both', labelsize=15)
ax1.grid(True)
ax1.legend(fontsize=15, loc='lower left')
ax1.set_title('A. Time Series Returns', size=15, fontweight="bold")  # 添加标题

ax1.set_xlim(x[0] - pd.DateOffset(months=1), x[-1] + pd.DateOffset(months=1))

ax2 = axes[1]

linewidth = 4
port_list = ['MKT', f'$R_d^{best_factor_choose}$', f'$R_{best_factor_choose}^{{opt}}$']
port_name = ['Market Factor', 'Deep Factor', 'Deep TP']
line_types = ['-','--','-.']
color_list = ['steelblue','red','magenta']

for i, port in enumerate(port_list):
    ax2.plot(df_cumret_oos_norm[port], ls=line_types[i], linewidth=linewidth, 
             label=port_name[i], color=color_list[i])

ax2.set_title('B. Cumulative Returns', size=15, fontweight="bold")
ax2.legend(loc='upper left', fontsize=15)
ax2.grid(color='grey', linestyle='-.', linewidth=.5)
ax2.axvspan('2020-02-28', '2020-04-30', alpha=0.3, color='grey')
ax2.tick_params(axis='both', which='major', labelsize=15)

ax2.set_xlim(df_cumret_oos_norm.index[0] - pd.DateOffset(months=1), 
             df_cumret_oos_norm.index[-1] + pd.DateOffset(months=1))

plt.tight_layout()
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.3)
plt.savefig(f'../{final_output}/Figure_3.pdf', dpi=100, bbox_inches = 'tight')
plt.show()
plt.close()

print(f"Figure 3 run time: {round((time.time() - start), 2)} seconds \n")



################################################
##### Table 4: Factor-Spanning Regressions #####
################################################

start = time.time()

df_sel_ft = dict_df['ew'][f'{best_factor_choose}_1']


lam = 1e-15
list_sample = [f'$R_{{d}}^{best_factor_choose}$','',f'$R_{best_factor_choose}^{{opt}}$','']
gap = len(list_sample)

iter_index = ['AlphaFF5','$MKT_{EQ}$','SMB','HML','TRM','DEF']
iter_index = iter_index + ['AlphaIPCA5']+['$IPCA_{}$'.format(i) for i in range(1,6)]
iter_index = iter_index + ['$RPPCA_{}$'.format(i) for i in range(1,6)]

df_beta = {}
df_pvalue = {}

count1 = 0
count2 = 0

for x,count1,model_name in [(df_ff5,5,'FF5'), (df_ipca,5,'IPCA5'), (df_rppca,5,'RPPCA5')]:
    df_beta[model_name] = pd.DataFrame(index =list_sample,columns = range(7))
    df_pvalue[model_name] = df_beta[model_name].copy()
    count2 = 0
    for y in [df_sel_ft]:
        combined_y = y.copy()
                
        y2 = y[:,-1].reshape(-1,1)[T_INS:T_ALL,:]*100
        x2 = x[T_INS:T_ALL,:]*100
        
        newey_west_L = int(4*(len(y2)/100)**(2/9))
        mod_ts = sm.OLS(y2,sm.add_constant(x2))
        res= mod_ts.fit(cov_type='HAC',cov_kwds={'maxlags':newey_west_L})
        a_val2 = res.params[:]
        p_val2 = res.pvalues[:]
        t_val2 = res.tvalues[:]
        R_square = res.rsquared_adj
        
        df_beta[model_name].iloc[0,:count1+2] = np.append(a_val2,R_square*100)
        df_beta[model_name].iloc[1,:count1+1]= t_val2
        
        df_pvalue[model_name].iloc[0,:count1+1] = p_val2        

        y1 = combined_y[:T_INS,:]*100
        x1 = x[:T_INS,:]*100
        w1 = TP2(y1, lam)
        ret_1 = y1@w1

        y2 = combined_y[T_INS:T_ALL,:]*100
        x2 = x[T_INS:T_ALL,:]*100
        ret_2 = y2@w1
        
        newey_west_L = int(4*(len(ret_2)/100)**(2/9))
        mod_ts4 = sm.OLS(ret_2,sm.add_constant(x2))
        res4= mod_ts4.fit(cov_type='HAC',cov_kwds={'maxlags':newey_west_L})
        a_val4 = res4.params[:]
        p_val4 = res4.pvalues[:]
        t_val4 = res4.tvalues[:]
        R_square = res4.rsquared_adj
        df_beta[model_name].iloc[2,:count1+2] = np.append(a_val4,R_square*100)
        df_beta[model_name].iloc[3,:count1+1]= t_val4
        
        df_pvalue[model_name].iloc[2,:count1+1] = p_val4


df_table4 = pd.concat([df_beta['FF5'],df_beta['IPCA5'],df_beta['RPPCA5']]).applymap(lambda x: ('%.2f' % x))
col_even = [1,3,5,7,9,11]
df_table4.iloc[col_even] = '('+df_table4.iloc[col_even]+')'
df_table4 = '$'+df_table4+'^{'
df_table4 = df_table4+'}$'
df_table4 = df_table4.replace('$(nan)^{}$','{}')
df_table4[''] = ['FF5']*4+['IPCA5']*4+['RPPCA5']*4
df_table4 = df_table4.set_index('',append=True)
df_table4.columns = ['alpha','beta_MKT','beta_SMB','beta_HML','beta_TRM','beta_DEF','Adj R^2']
df_table4.to_csv(f'../{final_output}/Table_4.csv', index=True, encoding='utf_8_sig')

print(f"Table 4 run time: {round((time.time() - start), 2)} seconds \n")



# #####################################################################
# ##### Figure 4: Variable Importance: Average Absolute Gradients #####
# #####################################################################

# start = time.time()

# dict_char = {}
# for n_layer in range(1, max_layers+1):
#     this_params = df_best[(df_best['n_layer'] == n_layer)]
#     learning_rate = this_params['learning_rate'].values.tolist()[0]
#     dropout_rate = this_params['dropout_rate'].values.tolist()[0]
#     a1 = this_params['a1'].values.tolist()[0]
#     a2 = this_params['a2'].values.tolist()[0]
#     for n_deep_factors in range(1, 2):

#         file_name_uncond = '{}/dchars_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.txt'.format(
#                 output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed)

#         temp_fix = pd.read_table(file_name_uncond,sep='\\s+',header=None).values

#         key_name = '{}_{}'.format(n_layer,n_deep_factors)

#         dict_char[key_name] = temp_fix[:T_ALL,:]

# df_char_all_2 = pd.read_feather(f'../data/data_in_all_{update_date}/characteristics_impute_selected{panel_num}_with_Equity_{update_date}_xret.feather').set_index('trd_exctn_dt')
# df_frame = df_char_all_2.resample('M').mean()[list_char]*0

# df_char_all_2 = df_char_all_2.loc[:oos_end,list_char]
# df_char_all_2 = df_char_all_2.values.reshape((T_ALL, panel_num, -1))

# y = dict_char[list_uncond_best_capm[best_factor_choose-1]]
# x = df_char_all_2

# beta_array = np.zeros([x.shape[2],T_ALL])
# beta_char = np.zeros([T_ALL,panel_num,len(list_char)])

# for i in range(0,T_ALL):
#     y_part = y[i]
#     x_part = x[i]
    
#     mod_ts = sm.OLS(y_part,x_part)
    
#     res= mod_ts.fit()
#     beta_array[:,i] = res.params[:]
#     beta_char[i] = (x_part[i]*beta_array[:,i])


# var_betaxchar = np.zeros([T_ALL,len(list_char)])
# for i in range(0,T_ALL):
#     var_betaxchar[i] = np.var(beta_char[i],axis=0)

# df_rename = pd.read_csv(f'../data/data_in_all_{update_date}/bond_char_list.csv',header=None)
# df_rename = df_rename.rename(columns={0:'before',1:'after'})

# df_char_sum = pd.DataFrame(index= df_frame.columns)

# df_char_sum['mean'] = np.mean(beta_array,axis=1)
# df_char_sum['std'] = np.std(beta_array,axis=1)

# for i,row in enumerate(df_char_sum.index):
#     newey_west_L = int(4*(T_ALL/100)**(2/9))
#     mod_ts = sm.OLS(beta_array[i], np.ones(T_ALL))
#     res= mod_ts.fit(cov_type='HAC',cov_kwds={'maxlags':newey_west_L})
#     df_char_sum.loc[row,'VI2'] = res.params[0]
#     df_char_sum.loc[row,'t_stat'] = res.tvalues[0]
#     df_char_sum.loc[row,'pvalue'] = res.pvalues[0]
# df_char_sum['percentage'] = np.abs(df_char_sum['mean'])/np.sum(np.abs(df_char_sum['mean']))*100

    
# for row in df_char_sum.index:
#     if row in df_rename['before'].values:
#         df_char_sum.loc[row,'newchar'] = df_rename[df_rename['before']==row]['after'].values[0]
#     else:
#         df_char_sum.loc[row,'newchar'] = row.upper()

#     if df_char_sum.loc[row,'pvalue']<0.01:
#         df_char_sum.loc[row,'newchar'] += '***'
#     elif df_char_sum.loc[row,'pvalue']<0.05:
#         df_char_sum.loc[row,'newchar'] += '**'
#     elif df_char_sum.loc[row,'pvalue']<0.1:
#         df_char_sum.loc[row,'newchar'] += '*'
        
#     if row in list_bond:
#         df_char_sum.loc[row,'category'] = 'Bond'
#     elif row in list_equity:
#         df_char_sum.loc[row,'category'] = 'Equity'
#     else:
#         df_char_sum.loc[row,'category'] = 'Option'

# df_char_sum['VI'] = np.mean((var_betaxchar/np.var(y,axis=1).reshape(-1,1)),axis=0)
# df_char_sum['VI'] = df_char_sum['VI']/df_char_sum['VI'].sum()
# df_char_sum[['Bond','Equity','Option']]=np.nan
# for row in df_char_sum.index:
#     if df_char_sum.loc[row,'category'] == 'Bond':
#         df_char_sum.loc[row,'Bond'] = df_char_sum.loc[row,'VI2']
#     elif df_char_sum.loc[row,'category'] == 'Equity':
#         df_char_sum.loc[row,'Equity'] = df_char_sum.loc[row,'VI2']
#     else:
#         df_char_sum.loc[row,'Option'] = df_char_sum.loc[row,'VI2']
        
# i=1
# num_char = len(list_char)
# list_layer = [num_char//2,num_char//4,num_char//8,num_char//16]

# this_params = df_best[(df_best['n_layer'] == n_layer)]
# learning_rate = this_params['learning_rate'].values.tolist()[0]
# dropout_rate = this_params['dropout_rate'].values.tolist()[0]
# a1 = this_params['a1'].values.tolist()[0]
# a2 = this_params['a2'].values.tolist()[0]

# input_layer = list_layer[:1].copy()

# df_temp = pd.concat([pd.read_csv('{}/grad_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.txt'.format(
#             output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed)).iloc[[-1],1:] for j,(index_date,row) in enumerate(df_rf_all.iloc[T_INS:T_ALL].iterrows())]).abs().mean()
# df_char = df_temp.div(df_temp.sum()).sort_values(ascending=False).to_frame('VI')

# df_char_sum['newchar'] =  df_char_sum['newchar'].str.replace(r'\*+', '', regex=True)

# df_char = df_char.merge(df_char_sum[['category','Bond',	'Equity','Option','newchar']],left_index=True,right_index=True)

# import seaborn as sns

# top_display = len(df_char)
# custom_palette = sns.color_palette('Set3',10)

# df_char_top50 = df_char.iloc[:top_display]

# bar_colors = {'Bond': custom_palette[4], 'Equity': custom_palette[3],'Option':custom_palette[1]}

# fig,ax = plt.subplots(figsize=(80,180))
# sns.barplot(data =df_char_top50, y='newchar',x='VI',hue='category',hue_order=['Bond','Equity','Option'],dodge=False,palette=[custom_palette[4],custom_palette[1],custom_palette[3]])

# ax.tick_params(labelsize=72,length=0)
# plt.box(False)
# ax.xaxis.grid(linewidth=1,color='black')
# ax.xaxis.tick_top()
# ax.set_xlabel('')

# plt.legend(fontsize='96', title_fontsize='96',loc=4)

# plt.savefig(f'../{final_output}/Figure_4.pdf', dpi=100, bbox_inches = 'tight')
# plt.show()
# plt.close()

# print(f"Figure 4 run time: {round((time.time() - start), 2)} seconds \n")





##################################################
##### Table 5: Importance of Characteristics #####
##################################################

start = time.time()

frame_table_5 = []

for char_type in chars_use_list:

    ew_dir = './results_{}/ew_with_batch/{}'.format(char_type, bm_type)
    df_best = pd.read_csv(f'{ew_dir}/best_summary_seed{seed}.csv')

    dict_capm_uncond = {}
    dict_capm_cond = {}
    dict_capm_is_uncond = {}
    dict_capm_oos_uncond = {}
    dict_capm_is_cond = {}
    dict_capm_oos_cond = {}

    dict_loss = {}

    dict_df = {}
    for port_type in ['ew']:
        dict_df[port_type] = {}
        output_dir = './results_{}/{}_with_batch/{}'.format(char_type, port_type, bm_type)

        for n_layer in range(1, max_layers+1):
            this_params = df_best[(df_best['n_layer'] == n_layer)]
            learning_rate = this_params['learning_rate'].values.tolist()[0]
            dropout_rate = this_params['dropout_rate'].values.tolist()[0]
            a1 = this_params['a1'].values.tolist()[0]
            a2 = this_params['a2'].values.tolist()[0]
            for n_deep_factors in range(1, 2):
                
                file_name_uncond = '{}/factor_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.txt'.format(
                            output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed)

                loss_file = '{}/loss_path_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.txt'.format(
                            output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed)
            
                temp_fix = pd.read_table(file_name_uncond,sep='\\s+',header=None)
                sign = np.sign(np.mean(temp_fix[:T_INS]))
                temp_fix = sign*temp_fix
                temp_fix = temp_fix.values[:T_ALL]

                key_name = '{}_{}'.format(n_layer,n_deep_factors)
                last_key_name = '{}_{}'.format(n_layer,n_deep_factors-1)
                sr_uncond = np.mean(temp_fix[T_INS:T_ALL,])/np.std(temp_fix[T_INS:T_ALL,])*np.sqrt(12)

                if n_deep_factors == 1:
                    dict_df[port_type][key_name] = np.concatenate([dict_bm[port_type],temp_fix],axis=1)
                    dict_capm_uncond[key_name] = np.concatenate([dict_bm[bm_type],temp_fix],axis=1)

                else:
                    dict_df[port_type][key_name] = np.concatenate([dict_df[port_type][last_key_name],temp_fix],axis=1)
                    dict_capm_uncond[key_name] = np.concatenate([dict_capm_uncond[last_key_name],temp_fix],axis=1)

                dict_capm_is_uncond[key_name] = dict_capm_uncond[key_name][:T_INS,:]
                dict_capm_oos_uncond[key_name] = dict_capm_uncond[key_name][T_INS:T_ALL,:]

                dict_loss[key_name] = pd.read_table(loss_file,sep='\\s+',header=None)

    dict_return = {}
    dict_SR = {}
    dict_pvalue = {}
    list_column = [bm_type,'D=1']
    df_sr = {}
    dict_weight = {}

    for port_type in ['ew']:
        dict_weight[port_type] = MVP_weight(dict_bm[port_type])
        dict_return['IS'] = {}
        dict_return['OOS'] = {}
        dict_return['IS'][port_type] = dict_bm[port_type][:T_INS]
        dict_return['OOS'][port_type] = dict_bm[port_type][T_INS:T_ALL]

        df_sr[port_type] = {}
        dict_SR[port_type] = {}
        for time_period in ['IS','OOS']:
            df_sr[port_type][time_period] = pd.DataFrame(index =['L=1','L=2','L=3'],columns = list_column)
            
            dict_SR[port_type][time_period] = {}
            dict_SR[port_type][time_period] = np.mean(dict_return[time_period][port_type]@dict_weight[port_type][0])/np.std(dict_return[time_period][port_type]@dict_weight[port_type][0])*np.sqrt(12)
            
            dict_pvalue[time_period] = {}
            df_sr[port_type][time_period].iloc[:,0] = dict_SR[port_type][time_period]

    for port_type in ['ew']:
        this_dict_df = dict_df[port_type]
        this_df_sr = df_sr[port_type]
        market_SR_IS = dict_SR[port_type]['IS']
        market_SR_OOS = dict_SR[port_type]['OOS']

        this_dict_SR = {}
        dict_capm_norm = {}
        dict_timeseries = {}
        dict_timeseries_norm = {}
        for time_period in ['IS','OOS']:
            this_dict_SR[time_period] = {}
            dict_pvalue[time_period] = pd.DataFrame(index =['L=1','L=2','L=3'],columns = list_column)
            
            for ii in range(1, max_layers+1):
                for jj in range(1, 2):

                    key = '{}_{}'.format(ii,jj)
                    w_temp = np.linalg.inv(np.cov(this_dict_df[key][:T_INS].T))@np.mean(this_dict_df[key][:T_INS],axis=0)
                    w_temp = w_temp / np.abs(w_temp).sum()

                    dict_weight[key] = w_temp
                    df_ = this_dict_df[key]
                    dict_capm_norm[key] = df_.copy()
                    
                    if time_period == 'IS':

                        T = T_INS
                        df_ = df_[:T_INS]
                        p = df_.shape[1]
                        dict_return[time_period][key] = df_ @ dict_weight[key]
                        
                        dict_timeseries_norm[key] = dict_return[time_period][key]
                    
                        this_dict_SR[time_period][key] = np.mean(dict_return[time_period][key])/np.std(dict_return[time_period][key])*np.sqrt(12)
                        this_df_sr[time_period].iloc[ii-1,jj] = this_dict_SR[time_period][key]
                        if jj>=2:
                            D = 1 
                            F= T/D *(T-p-D)/(T-p-1)*(this_dict_SR[time_period][key]**2-this_dict_SR[time_period]['{}_{}'.format(ii,jj-1)]**2)/(1+this_dict_SR[time_period]['{}_{}'.format(ii,jj-1)]**2)
                            dict_pvalue[time_period].iloc[ii-1,jj] = 1-scipy.stats.f.cdf(F,D,T-p-D)
                        else:
                            D = 1
                            F= T/D *(T-p-D)/(T-p-1)*(this_dict_SR[time_period][key]**2-market_SR_IS**2)/(1+market_SR_IS**2)
                            dict_pvalue[time_period].iloc[ii-1,jj] = 1-scipy.stats.f.cdf(F,D,T-p-D)
                            
                            
                        if jj>=2:
                            D = jj 
                            F= T/p *(T-p-D)/(T-p-1)*(this_dict_SR[time_period][key]**2-this_dict_SR[time_period]['{}_{}'.format(ii,jj-1)]**2)/(1+this_dict_SR[time_period]['{}_{}'.format(ii,jj-1)]**2)
                            dict_pvalue[time_period].iloc[ii-1,jj] = 1-scipy.stats.f.cdf(F,p,T-p-D)
                        else:
                            D = 1
                            F= T/p *(T-p-D)/(T-p-1)*(this_dict_SR[time_period][key]**2-market_SR_IS**2)/(1+market_SR_IS**2)
                            dict_pvalue[time_period].iloc[ii-1,jj] = 1-scipy.stats.f.cdf(F,p,T-p-D)

                    
                    elif time_period == 'OOS':

                        T = T_OOS
                        df_2 = df_[T_INS:T_ALL,]
                        p = df_2.shape[1]
                        sign_df = np.sign(np.mean(df_2,axis=0))*df_2
                        dict_return[time_period][key] =  sign_df @ dict_weight[key]
                        
                        dict_timeseries_norm[key] = np.concatenate([dict_timeseries_norm[key],dict_return[time_period][key]])
                        
                        dict_timeseries[key] = dict_timeseries_norm[key]
                        this_dict_SR[time_period][key] = np.mean(dict_return[time_period][key])/np.std(dict_return[time_period][key])*np.sqrt(12)
                        this_df_sr[time_period].iloc[ii-1,jj] = this_dict_SR[time_period][key]
                        if jj>=2:
                            D = 1 
                            F= T/D *(T-p-D)/(T-p-1)*(this_dict_SR[time_period][key]**2-this_dict_SR[time_period]['{}_{}'.format(ii,jj-1)]**2)/(1+this_dict_SR[time_period]['{}_{}'.format(ii,jj-1)]**2)
                            dict_pvalue[time_period].iloc[ii-1,jj] = 1-scipy.stats.f.cdf(F,D,T-p-D)
                        else:
                            D = 1
                            F= T/D *(T-p-D)/(T-p-1)*(this_dict_SR[time_period][key]**2-market_SR_OOS**2)/(1+market_SR_OOS**2)
                            dict_pvalue[time_period].iloc[ii-1,jj] = 1-scipy.stats.f.cdf(F,D,T-p-D)
                        if jj>=2:
                            D = jj 
                            F= T/p *(T-p-D)/(T-p-1)*(this_dict_SR[time_period][key]**2-this_dict_SR[time_period]['{}_{}'.format(ii,jj-1)]**2)/(1+this_dict_SR[time_period]['{}_{}'.format(ii,jj-1)]**2)
                            dict_pvalue[time_period].iloc[ii-1,jj] = 1-scipy.stats.f.cdf(F,p,T-p-D)
                        else:
                            D = 1
                            F= T/p *(T-p-D)/(T-p-1)*(this_dict_SR[time_period][key]**2-market_SR_OOS**2)/(1+market_SR_OOS**2)
                            dict_pvalue[time_period].iloc[ii-1,jj] = 1-scipy.stats.f.cdf(F,p,T-p-D)
                    
    count = 0
    for df_ in [df_bbw_all,df_ipca,df_ff5,df_rppca]:
        # df_rppca
        if count ==0:
            key = 'BBW'
        elif count ==1:
            key = 'IPCA'
        elif count ==2:
            key='FF5'
        else:
            key = 'RPPCA'
        count += 1
        
        weight = MVP_weight(df_)

        df_is = df_[:T_INS,:]

        dict_timeseries[key] = df_is @ weight[0]
        
        df_2 = df_[T_INS:T_ALL,0:]

        dict_timeseries[key] =  np.concatenate([dict_timeseries[key],df_2 @ weight[0]])
        
        dict_timeseries_norm[key] = dict_timeseries[key].copy()

    for port in ['ew']:
        for time_period in ['OOS']:
            df_sr[port][time_period] = np.round(df_sr[port][time_period].astype(float), 2).astype(str)

            df_sr[port][time_period][(dict_pvalue[time_period]<=0.1)] = df_sr[port][time_period][(dict_pvalue[time_period]<=0.1)]+'*'
            df_sr[port][time_period][(dict_pvalue[time_period]<=0.05)] = df_sr[port][time_period][(dict_pvalue[time_period]<=0.05)]+'*'
            df_sr[port][time_period][(dict_pvalue[time_period]<=0.01)] = df_sr[port][time_period][(dict_pvalue[time_period]<=0.01)]+'*'

    output = df_sr['ew']['OOS'].reset_index().set_index(['index', 'CAPM'])
    output.columns = [char_type]
    frame_table_5.append(output)
df_table_5 = pd.concat(frame_table_5, axis=1)
df_table_5.to_csv(f'../{final_output}/Table_5.csv', index=True, encoding='utf_8_sig')

print(f"Table 5 run time: {round((time.time() - start), 2)} seconds \n")

