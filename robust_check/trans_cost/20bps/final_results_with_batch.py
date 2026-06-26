import numpy as np
import pandas as pd
import statsmodels.api as sm
import imp
import os
import scipy.stats
from DL_functions_trans_cost_20bps import *
from pandas.tseries.offsets import MonthEnd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
pd.set_option('display.max_columns', 100)
from datetime import datetime
import matplotlib.gridspec as gridspec
from parameters import *
from itertools import product
import warnings
warnings.filterwarnings("ignore")

final_output = 'output'
if not os.path.exists(f"{final_output}"):
    os.makedirs(f"{final_output}")

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
        print(f"Calculation Error: {e}")
        return "NaN"

df_rf_all = pd.read_csv(f'../../../data/data_in_all_{update_date}/one_month_bill.csv',index_col=0)
df_rf_all.index = pd.to_datetime(df_rf_all.index.values)+MonthEnd(0)
df_rf_all = df_rf_all.loc[time_start:oos_end]

df_char_all = pd.read_feather(f'../../../data/data_in_all_{update_date}/characteristics_impute_selected{panel_num}_with_Equity_{update_date}_xret.feather').set_index('trd_exctn_dt').loc[time_start:oos_end]

if port_type == 'ew':
    market_factor = 'MKTbond'
elif port_type == 'vw':
    market_factor = 'MKTbond_vw'
df_mkt_all = pd.read_csv(f'../../../data/data_in_all_{update_date}/Replicated_Bond_risk_factors_bbw4.csv',index_col=0)[market_factor].loc[time_start:oos_end]
df_mkt_all = np.array(df_mkt_all).reshape(-1,1)

df_return_all = np.array(df_char_all['excess_ret']).reshape((T_ALL, -1))



bd_spread = pd.read_csv(f'../../../data/data_in_all_{update_date}/bonds_ba_spread.csv')
bd_spread['DATE'] = pd.to_datetime(bd_spread['DATE'])
bd_spread['T_Spread_val'] = bd_spread['T_Spread'].str.rstrip('%').astype('float') / 100
bd_spread['complete_cusip'] = bd_spread['CUSIP'].apply(lambda x: f"b'{x}'")
print(bd_spread['T_Spread_val'].describe())

df_char_all = pd.read_feather(f'../../../data/data_in_all_{update_date}/characteristics_impute_selected{panel_num}_with_Equity_{update_date}_xret.feather')
df_char_all['CUSIP'] = df_char_all['complete_cusip'].apply(lambda x: x.decode('utf-8'))
df_char_all = df_char_all.merge(bd_spread[['DATE', 'CUSIP', 'T_Spread_val']],left_on = ['trd_exctn_dt','CUSIP'], right_on = ['DATE','CUSIP'], how='left')
df_char_all['T_Spread_val'] = df_char_all.groupby('trd_exctn_dt')['T_Spread_val'].apply(lambda x: x.fillna(x.median()))
df_char_all = df_char_all.set_index('trd_exctn_dt')
df_char_all_2 = df_char_all.copy()
df_char_all = df_char_all.loc[time_start:oos_end]

df_return_all = np.array(df_char_all['excess_ret']).reshape((T_ALL, -1))
df_bdspread_all = np.array(df_char_all['T_Spread_val']).reshape((T_ALL, -1))


dict_mkt = {}
dict_mkt['ew'] = pd.read_csv(f'../../../data/data_in_all_{update_date}/Replicated_Bond_risk_factors_bbw4.csv',index_col=0)['MKTbond'].loc[time_start:oos_end]
dict_mkt['ew'].index = pd.to_datetime(dict_mkt['ew'].index)
dict_mkt['vw'] = pd.read_csv(f'../../../data/data_in_all_{update_date}/Replicated_Bond_risk_factors_bbw4.csv',index_col=0)['MKTbond_vw'].loc[time_start:oos_end]
df_temp = pd.DataFrame(dict_mkt)


df_rppca_new = pd.concat([pd.read_csv(f'../../../data/rppca_{update_date}/rppca_ins.csv',header=None),
                          pd.read_csv(f'../../../data/rppca_{update_date}/rppca_oos.csv',header=None)])
df_rppca_new.columns = [f'rppca_{x+1}' for x in range(df_rppca_new.shape[1])]
df_rppca_new = df_rppca_new.reset_index(drop=True)[:T_ALL]
df_rppca = df_rppca_new.values


df_ipca_is = pd.read_csv(f'../../../data/ipca_{update_date}/ipca5_ins_{char_type}_{panel_num}_{update_date}.csv').set_index('Dates')
df_ipca_oos = pd.read_csv(f'../../../data/ipca_{update_date}/ipca5_oos_{char_type}_{panel_num}_{update_date}.csv').set_index('Unnamed: 0')
df_ipca = pd.concat([df_ipca_is,df_ipca_oos]).loc[time_start:oos_end].values

df_bbw = pd.read_csv(f'../../../data/data_in_all_{update_date}/Replicated_Bond_risk_factors_bbw4.csv',index_col=0)
df_bbw_all = df_bbw.loc[time_start:oos_end].values
df_bbw_all[:,0] = df_mkt_all.reshape(-1) 


dict_bm = {}
dict_bm['CAPM'] = df_mkt_all.reshape(-1,1)
dict_bm['BBW'] = df_bbw_all.copy()
dict_bm['IPCA'] = df_ipca.copy()
dict_bm['ew'] = df_mkt_all.reshape(-1,1)
dict_bm['vw'] = dict_mkt['vw'].values.reshape(-1,1)

df_termdef = pd.read_csv(f'../../../data/data_in_all_{update_date}/term_and_default_factor.csv', index_col=0).loc[time_start:oos_end, ['Term Factor','Default Factor']]

df_ff5 = pd.read_csv(f'../../../data/data_in_all_{update_date}/bond_factors.csv',index_col=0).loc[time_start:oos_end, ['MKT','SMB','HML']]
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




# Calculate results

for gamma in list_gamma:

    if (gamma == 0.7) or (gamma == 0.5):
        batches = 15
    else:
        batches = 30

    ew_dir = './results_{}/ew_with_batch_{}/{}'.format(char_type, batches, bm_type)
    df_best = pd.read_csv(f'{ew_dir}/best_summary_gamma_{gamma}_seed{seed}.csv')
    df_best


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

            dict_weight[i-T_INS] = w
        return dict_weight


    dict_capm_uncond = {}
    dict_capm_cond = {}
    dict_capm_is_uncond = {}
    dict_capm_oos_uncond = {}
    dict_capm_is_cond = {}
    dict_capm_oos_cond = {}

    dict_loss = {}

    dict_df, dict_df_w_df = {}, {}
    for port_type in ['vw','ew']:

        dict_capm_uncond[port_type] = {}
        dict_capm_cond[port_type] = {}
        dict_capm_is_uncond[port_type] = {}
        dict_capm_oos_uncond[port_type] = {}
        dict_capm_is_cond[port_type] = {}
        dict_capm_oos_cond[port_type] = {}

        dict_loss[port_type] = {}

        dict_df[port_type] = {}
        dict_df_w_df[port_type] = {}
        output_dir = './results_{}/{}_with_batch_{}/{}'.format(char_type, port_type, batches, bm_type)

        for n_layer in range(1, max_layers+1):
            this_params = df_best[(df_best['n_layer'] == n_layer)]
            learning_rate = this_params['learning_rate'].values.tolist()[0]
            dropout_rate = this_params['dropout_rate'].values.tolist()[0]
            a1 = this_params['a1'].values.tolist()[0]
            a2 = this_params['a2'].values.tolist()[0]
            for n_deep_factors in range(1, deep_factors_num+1):
                
                file_name_uncond = '{}/factor_lr{}_dr{}_a1{}_a2{}_gamma{}_L{}_D{}_seed{}.txt'.format(
                            output_dir, learning_rate, dropout_rate, a1, a2, gamma, n_layer, n_deep_factors, seed)

                weight_file = '{}/weights_lr{}_dr{}_a1{}_a2{}_gamma{}_L{}_D{}_seed{}.txt'.format(
                            output_dir, learning_rate, dropout_rate, a1, a2, gamma, n_layer, n_deep_factors, seed)

                loss_file = '{}/loss_path_lr{}_dr{}_a1{}_a2{}_gamma{}_L{}_D{}_seed{}.txt'.format(
                            output_dir, learning_rate, dropout_rate, a1, a2, gamma, n_layer, n_deep_factors, seed)

                w_df_file = '{}/w_df_lr{}_dr{}_a1{}_a2{}_gamma{}_L{}_D{}_seed{}.txt'.format(
                            output_dir, learning_rate, dropout_rate, a1, a2, gamma, n_layer, n_deep_factors, seed)

                temp_fix = pd.read_table(file_name_uncond,sep='\s+',header=None)
                temp_fix = temp_fix.values[:T_ALL]

                w_df = pd.read_table(w_df_file,sep='\s+',header=None).values[0]

                key_name = '{}_{}'.format(n_layer,n_deep_factors)
                last_key_name = '{}_{}'.format(n_layer,n_deep_factors-1)

                if n_deep_factors == 1:
                    dict_df[port_type][key_name] = np.concatenate([dict_bm[port_type],temp_fix],axis=1)
                    dict_df_w_df[port_type][key_name] = np.concatenate([[1-w_df], [w_df]], axis=1)
                    dict_capm_uncond[port_type][key_name] = np.concatenate([dict_bm[port_type],temp_fix],axis=1)

                else:
                    dict_df[port_type][key_name] = np.concatenate([dict_df[port_type][last_key_name],temp_fix],axis=1)
                    dict_df_w_df[port_type][key_name] = np.concatenate([dict_df_w_df[port_type][last_key_name], [w_df]], axis=1)
                    dict_capm_uncond[port_type][key_name] = np.concatenate([dict_capm_uncond[port_type][last_key_name],temp_fix],axis=1)

                dict_capm_is_uncond[port_type][key_name] = dict_capm_uncond[port_type][key_name][:T_INS,:]
                dict_capm_oos_uncond[port_type][key_name] = dict_capm_uncond[port_type][key_name][T_INS:T_ALL,:]

                dict_loss[port_type][key_name] = pd.read_table(loss_file,sep='\s+',header=None)


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
        this_dict_df_w_df = dict_df_w_df[port_type]
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
                    w_temp = this_dict_df_w_df[key][0]
                    dict_weight[port_type][key] = w_temp
                    print(f'weight_insample: {w_temp}')

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
                        sign_df = df_2
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


    for port in ['vw','ew']:
        for time_period in ['IS','OOS']:
            df_sr[port][time_period] = np.round(df_sr[port][time_period].astype(float), 2).astype(str)

            df_sr[port][time_period][(dict_pvalue[port][time_period]<=0.1)] = df_sr[port][time_period][(dict_pvalue[port][time_period]<=0.1)]+'*'
            df_sr[port][time_period][(dict_pvalue[port][time_period]<=0.05)] = df_sr[port][time_period][(dict_pvalue[port][time_period]<=0.05)]+'*'
            df_sr[port][time_period][(dict_pvalue[port][time_period]<=0.01)] = df_sr[port][time_period][(dict_pvalue[port][time_period]<=0.01)]+'*'



    df_tableA6 = pd.concat([df_sr['ew']['OOS'], df_sr['vw']['OOS']])
    df_tableA6[''] = ['EW_OOS']*3 + ['VW_OOS']*3
    df_tableA6 = df_tableA6.set_index(['','CAPM'], append=True)
    df_tableA6 = df_tableA6[['D=1']].rename(columns={'D=1':'SR'})

    df_tableA6_mdd = pd.concat([df_DF_mdd['ew']['OOS'], df_DF_mdd['vw']['OOS']])
    df_tableA6_mdd[''] = ['EW_OOS']*3 + ['VW_OOS']*3
    df_tableA6_mdd = df_tableA6_mdd.set_index(['','CAPM'], append=True)
    df_tableA6_mdd = df_tableA6_mdd[['D=1']].rename(columns={'D=1':'MDD'})
    df_tableA6['MDD'] = df_tableA6_mdd['MDD'].tolist()

    df_tableA6_alpha = pd.concat([df_DF_alpha['ew']['OOS'], df_DF_alpha['vw']['OOS']])
    df_tableA6_alpha[''] = ['EW_OOS']*3 + ['VW_OOS']*3
    df_tableA6_alpha = df_tableA6_alpha.set_index(['','CAPM'], append=True)
    df_tableA6_alpha = df_tableA6_alpha[['D=1']].rename(columns={'D=1':'Alpha'})
    df_tableA6['Alpha'] = df_tableA6_alpha['Alpha']

    df_tableA6 = np.round(df_tableA6, 2)
    df_tableA6.to_csv(f'{final_output}/Table_A6_SR_MDD_Alpha_trans_cost_{int(trans_cost*10000)}bps_gamma_{gamma}.csv', index=True, encoding='utf_8_sig')


