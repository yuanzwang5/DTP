import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
from DL_functions_sample_randomly_split import *
from pandas.tseries.offsets import MonthEnd
from datetime import datetime
from parameters import *










T_ALL = T_INS + T_OOS

df_rf_all = pd.read_csv(f'../../data/data_in_all_{update_date}/one_month_bill.csv',index_col=0)
df_rf_all.index = pd.to_datetime(df_rf_all.index)+MonthEnd(0)
df_rf_all = df_rf_all.loc[time_start:time_end]
df_rf_all_use = np.array(df_rf_all).reshape((T_ALL, -1))
df_time = df_rf_all.index


df_char_all = pd.read_feather(
    f'../../data/data_in_all_{update_date}/characteristics_impute_selected{panel_num}_with_Equity_{update_date}_xret.feather').set_index('trd_exctn_dt').loc[time_start:time_end]

df_xret_all = np.array(df_char_all['monthly_return']).reshape((T_ALL, -1))

if char_type == 'bond':
    list_char = list_bond
    df_char_all = df_char_all.loc[:, list_char]
elif char_type == 'bond_equity_option':
    list_char = list_bond + list_equity+list_option
    df_char_all = df_char_all.loc[:, list_char]
elif char_type == 'bond_equity':
    list_char = list_bond + list_equity
    df_char_all = df_char_all.loc[:, list_char]


df_char_all = np.array(df_char_all).reshape((T_ALL, panel_num, -1))

df_set_time = pd.read_csv('saved_random_seq_666.csv',parse_dates=['caldt']).iloc[:,1:] #.set_index('caldt')
time_list = list(df_set_time['set'].values)

num_char = len(list_char)
list_layer = [num_char//2,num_char//4,num_char//8,num_char//16]


port_type_list = ['ew', 'vw']
for port_type in port_type_list:
    # Decide which market factor to use
    if port_type == 'ew':
        market_factor = 'MKTbond'
    elif port_type == 'vw':
        market_factor = 'MKTbond_vw'
    df_mkt_all = pd.read_csv(f'../../data/data_in_all_{update_date}/Replicated_Bond_risk_factors_bbw4.csv',index_col=0)[market_factor].loc[time_start:time_end]

    df_mkt_all = np.array(df_mkt_all)

    dict_bm = {}
    dict_bm['CAPM'] = df_mkt_all.reshape(-1,1)

    set_seed(seed)

    if oos_target == 'time_1':
        time_test = [i for i in range(len(time_list)) if time_list[i] == 1]
        time_train = [i for i in range(len(time_list)) if time_list[i] != 1]

    elif oos_target == 'time_2':
        time_test = [i for i in range(len(time_list)) if time_list[i] == 2]
        time_train = [i for i in range(len(time_list)) if time_list[i] != 2]

    elif oos_target == 'time_3':
        time_test = [i for i in range(len(time_list)) if time_list[i] == 3]
        time_train = [i for i in range(len(time_list)) if time_list[i] != 3]

    ins_return = np.nan_to_num(winsorize_by_row(df_xret_all[time_train,:]) - df_rf_all_use[time_train,:], nan=0.0)
    oos_return = np.nan_to_num(df_xret_all[time_test,:] - df_rf_all_use[time_test,:], nan=0.0)

    data_input = dict(bond_return_ins = torch.from_numpy(ins_return).float(),
                      bond_return_oos = torch.from_numpy(oos_return).float(),
                      num_firm = df_char_all.shape[1],
                      num_feature = num_char)

    df_benchmark_ins = dict_bm[bm_type][time_train]
    df_benchmark_oos = dict_bm[bm_type][time_test]

    tensor_input_ins = torch.from_numpy(df_char_all[time_train,:,:]).float()
    tensor_input_oos = torch.from_numpy(df_char_all[time_test,:,:]).float()


    output_dir = f'./results_{char_type}/{port_type}_3folds_period_{oos_target}_with_batch/{bm_type}'
    os.makedirs(output_dir, exist_ok=True)

    # Load in best parameters combination determined by Equal-Weighted case``
    ew_dir = f'./results_{char_type}/ew_3folds_period_{oos_target}_with_batch/{bm_type}'
    df_best = pd.read_csv(f'{ew_dir}/best_summary_seed{seed}.csv')

    set_seed(seed)

    n_layer = i
    input_layer = list_layer[:n_layer].copy()
    data_input['layers'] = input_layer
    this_params = df_best[(df_best['n_layer'] == n_layer)]
    learning_rate = this_params['learning_rate'].values[0]
    dropout_rate = this_params['dropout_rate'].values[0]
    a1 = this_params['a1'].values[0]
    a2 = this_params['a2'].values[0]
    data_input['dropout_rate'] = dropout_rate
    data_input['a1'] = a1
    data_input['a2'] = a2
    print(f'Layer {n_layer} Deep_Factor 1: \n Learning rate: {learning_rate}, dropout_rate: {dropout_rate}, a1: {a1}, a2: {a2}')

    for n_deep_factors in range(1, deep_factors_num+1):

        if n_deep_factors >= 2:
            temp_ins = torch.cat([data_input['factor_ins'], F_ins],axis=1)
            data_input['factor_ins'] = torch.from_numpy(temp_ins.data.numpy()).float()
            temp_oos = torch.cat([data_input['factor_oos'], F_oos],axis=1)
            data_input['factor_oos'] = torch.from_numpy(temp_oos.data.numpy()).float()
        else:
            data_input['factor_ins'] =  torch.from_numpy(df_benchmark_ins).float()
            data_input['factor_oos'] =  torch.from_numpy(df_benchmark_oos).float()

        list_loss_ins,F_ins,F_oos,weight_ins,weight_oos,x_ins,x_oos,list_grad,net = run_ins_oos_with_batch(data_input, tensor_input_ins, tensor_input_oos, learning_rate, batch_size=batch_size)

        F_whole_np = torch.cat([F_ins, F_oos]).data.numpy()
        x_all = torch.cat([x_ins, x_oos]).data.numpy()
        w_all = torch.cat([weight_ins, weight_oos]).data.numpy()

        tp_oos = (torch.cat([data_input['factor_oos'],F_oos],axis=1)@net.w_allocate).data.numpy()


        if n_deep_factors >= 2:
            FFF_oos = np.concatenate([FFF_oos,F_oos.reshape(-1,1).data.numpy()],axis=1)
        else:
            FFF_oos = np.concatenate([df_benchmark_oos,F_oos.reshape(-1,1).data.numpy()],axis=1)

        loss_oos = - np.mean(FFF_oos,axis=0)@np.linalg.inv(np.cov(FFF_oos.T))@np.mean(FFF_oos,axis=0)


        np.savetxt('{}/factor_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.txt'.format(
                output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed), F_whole_np)

        np.savetxt('{}/tp_oos_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.txt'.format(
                output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed), tp_oos)
           
        np.savetxt('{}/loss_path_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.txt'.format(
                output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed), list_loss_ins)

        np.savetxt('{}/loss_oos_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.txt'.format(
                output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed), [loss_oos])

        print(f'Save_success for n_layer = {n_layer}, n_deep_factors = {n_deep_factors},\n learning_rate = {learning_rate},\n dropout_rate = {dropout_rate}, \n a1 = {a1}, \n a2 = {a2}, \n loss_ins = {list_loss_ins[-1]}, \n seed = {seed} \n')
