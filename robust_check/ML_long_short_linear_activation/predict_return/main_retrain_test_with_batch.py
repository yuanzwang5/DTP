import numpy as np
import pandas as pd
import statsmodels.api as sm
import imp
import os
from DL_functions_predict_return import *
from pandas.tseries.offsets import MonthEnd
from datetime import datetime
current_time = datetime.now().strftime('%m%d')
from parameters import *







T_ALL = T_INS + T_OOS

df_rf_all = pd.read_csv(f'../../../data/data_in_all_{update_date}/one_month_bill.csv',index_col=0)
df_rf_all.index = pd.to_datetime(df_rf_all.index)+MonthEnd(0)
df_rf_all = df_rf_all.loc[time_start:time_end]

df_char_all = pd.read_feather(
    f'../../../data/data_in_all_{update_date}/characteristics_impute_selected{panel_num}_with_Equity_{update_date}_xret.feather').set_index('trd_exctn_dt').loc[time_start:time_end]

df_return_all = np.array(df_char_all['excess_ret']).reshape((T_ALL, -1))
df_xret_all = df_return_all

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

num_char = len(list_char)
list_layer = [num_char//2,num_char//4,num_char//8,num_char//16]


port_type_list = ['ew', 'vw']
for port_type in port_type_list:
    if port_type == 'ew':
        market_factor = 'MKTbond'
    elif port_type == 'vw':
        market_factor = 'MKTbond_vw'
    df_mkt_all = pd.read_csv(f'../../../data/data_in_all_{update_date}/Replicated_Bond_risk_factors_bbw4.csv',index_col=0)[market_factor].loc[time_start:time_end]

    df_mkt_all = np.array(df_mkt_all)

    dict_bm = {}
    dict_bm['CAPM'] = df_mkt_all.reshape(-1,1)

    data_input = dict(bond_return_ins = torch.from_numpy(df_xret_all[:T_INS,:]).float(),
                    bond_return_oos = torch.from_numpy(df_xret_all[T_INS:,:]).float(),
                    num_firm = df_char_all.shape[1],
                    num_feature = num_char)

    tensor_input_ins = torch.from_numpy(df_char_all[:T_INS,:,:]).float()
    tensor_input_oos = torch.from_numpy(df_char_all[T_INS:,:,:]).float()

    df_benchmark_ins = dict_bm[bm_type][:T_INS,:]
    df_benchmark_oos = dict_bm[bm_type][T_INS:,:]
        

    output_dir = './results_{}/{}_with_batch/{}'.format(char_type, port_type, bm_type)
    os.makedirs(output_dir, exist_ok=True)

    # Load in best parameters combination determined by Equal-Weighted case
    ew_dir = './results_{}/ew_with_batch/{}'.format(char_type, bm_type)
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

    n_deep_factors = 1

    list_loss_ins,ret_pred_ins,ret_pred_oos,net = run_ins_oos_with_batch(data_input, tensor_input_ins, tensor_input_oos, learning_rate, batch_size=batch_size)

    np.savetxt('{}/loss_path_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.txt'.format(
            output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed), list_loss_ins)

    np.savetxt('{}/ret_pred_ins_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.txt'.format(
            output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed), ret_pred_ins)

    np.savetxt('{}/ret_pred_oos_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.txt'.format(
            output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed), ret_pred_oos)

    np.save('{}/ret_pred_ins_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.npy'.format(
            output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed), ret_pred_ins)

    np.save('{}/ret_pred_oos_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.npy'.format(
            output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed), ret_pred_oos)

    torch.save(net, '{}/torchnn_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.pt'.format(
            output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed))

    print(f'Save_success for n_layer = {n_layer}, n_deep_factors = {n_deep_factors},\n learning_rate = {learning_rate},\n dropout_rate = {dropout_rate}, \n a1 = {a1}, \n a2 = {a2}, \n loss_ins = {list_loss_ins[-1]}, \n seed = {seed} \n')
