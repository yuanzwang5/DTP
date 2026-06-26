
import numpy as np
import pandas as pd
import statsmodels.api as sm
import imp
import os
from DL_functions_sample_randomly_split import *
from pandas.tseries.offsets import MonthEnd
from datetime import datetime
from parameters import *

final_output = 'output'
if not os.path.exists(f"../../{final_output}"):
    os.makedirs(f"../../{final_output}")

bm_type = "CAPM"
char_type = "bond_equity_option"
time_end = '2020-12-31'

df_rf_all = pd.read_csv(f'../../data/data_in_all_{update_date}/one_month_bill.csv',index_col=0)
df_rf_all.index = pd.to_datetime(df_rf_all.index)+MonthEnd(0)
df_rf_all = df_rf_all.loc[time_start:time_end]


df_set_time = pd.read_csv('saved_random_seq_666.csv',parse_dates=['caldt']).iloc[:,1:] #.set_index('caldt')
time_list = list(df_set_time['set'].values)

df_time_random = df_rf_all.copy()
df_time_random.columns = ['set']
df_time_random['set'] = time_list
df_time_random

dict_sub_time = {}
dict_sub_time['time_1'] = df_time_random[df_time_random['set'] == 1]
dict_sub_time['time_2'] = df_time_random[df_time_random['set'] == 2]
dict_sub_time['time_3'] = df_time_random[df_time_random['set'] == 3]


df_mkt_all = pd.read_csv(f'../../data/data_in_all_{update_date}/Replicated_Bond_risk_factors_bbw4.csv',index_col=0)['MKTbond'].loc[time_start:time_end]
df_mkt_all.index = pd.to_datetime(df_mkt_all.index)


df_split = df_mkt_all.to_frame('MKT_ret').merge(df_time_random,left_index=True,right_index=True)
import matplotlib.pyplot as plt

fig,ax = plt.subplots(figsize = (13.5,4.5))
for i in [1,2,3]:
    subset = df_split.query(f'set == {i}')
    
    ax.scatter(subset.index, subset['MKT_ret'], label=f'$time_{i}$')
plt.legend()
plt.savefig(f'../../{final_output}/Figure_A3.pdf', dpi=100, bbox_inches = 'tight')
plt.show()
plt.close()

dict_para = {}
dict_mkt_sub = {}
df_SR_3folds = pd.DataFrame(index=pd.MultiIndex.from_product([['ew','vw'], ['time_1','time_2','time_3','whole_oos']]), columns=['MKT','1_layer','2_layer','3_layer'])
df_char_all = pd.read_feather(
    f'../../data/data_in_all_{update_date}/characteristics_impute_selected{panel_num}_with_Equity_{update_date}_xret.feather').set_index('trd_exctn_dt').loc[time_start:time_end]

for port_type in ['ew','vw']:
    
    if port_type == 'ew':
        market_factor = 'MKTbond'
    elif port_type == 'vw':
        market_factor = 'MKTbond_vw'
    df_mkt_all = pd.read_csv(f'../../data/data_in_all_{update_date}/Replicated_Bond_risk_factors_bbw4.csv',index_col=0)[market_factor].loc[time_start:time_end]
    df_mkt_all.index = pd.to_datetime(df_mkt_all.index)
    df_split = df_mkt_all.to_frame('MKT_ret').merge(df_time_random,left_index=True,right_index=True)

    for n_layer in range(1, max_layers+1):
        
        for n_deep_factors in range(1,2):
            time_series = None
            df_tp = pd.DataFrame()
            for oos_target in oos_target_list:
                dict_mkt_sub[oos_target] = df_split[df_split['set'] == int(oos_target[-1])]['MKT_ret']
        
                output_dir = f'./results_{char_type}/{port_type}_3folds_period_{oos_target}_with_batch/{bm_type}'
                os.makedirs(output_dir, exist_ok=True)

                # Load in best parameters combination determined by Equal-Weighted case``
                ew_dir = f'./results_{char_type}/ew_3folds_period_{oos_target}_with_batch/{bm_type}'
                df_best = pd.read_csv(f'{ew_dir}/best_summary_seed{seed}.csv')
                                        
                df_SR_3folds.loc[(port_type,'{}'.format(oos_target)),'MKT'] = np.mean(dict_mkt_sub[oos_target])/np.std(dict_mkt_sub[oos_target])*np.sqrt(12)

                this_params = df_best[(df_best['n_layer'] == n_layer)]
                learning_rate = this_params['learning_rate'].values[0]
                dropout_rate = this_params['dropout_rate'].values[0]
                a1 = this_params['a1'].values[0]
                a2 = this_params['a2'].values[0]
                
                tp_oos = pd.read_csv('{}/tp_oos_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.txt'.format(
                        output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed),header=None)
                df_tp = pd.concat([df_tp,tp_oos],axis=0)
                df_SR_3folds.loc[(port_type,'{}'.format(oos_target)),'{}_layer'.format(n_layer)] = (tp_oos.mean()/tp_oos.std()*np.sqrt(12)).values[0]

            df_SR_3folds.loc[(port_type,'{}'.format('whole_oos')),'MKT'] = (df_mkt_all.mean()/df_mkt_all.std()*np.sqrt(12))

            df_SR_3folds.loc[(port_type,'{}'.format('whole_oos')),'{}_layer'.format(n_layer)] = (df_tp.mean()/df_tp.std()*np.sqrt(12)).values[0]


df_SR_3folds.applymap(lambda x: ('%.2f' % x)).T.to_csv(f'../../{final_output}/Table_A5_panelB.csv')





