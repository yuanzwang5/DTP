import numpy as np
import pandas as pd
import statsmodels.api as sm
import imp
import os
from DL_functions_main import *
from pandas.tseries.offsets import MonthEnd
from parameters import *









T_ALL = T_INS + T_OOS
odd_months = np.arange(0, T_VALID, 1)
even_months = np.arange(T_VALID, T_INS, 1)

df_rf_all = pd.read_csv(f'../data/data_in_all_{update_date}/one_month_bill.csv',index_col=0)
df_rf_all.index = pd.to_datetime(df_rf_all.index)+MonthEnd(0)
df_rf_all = df_rf_all.loc[time_start:time_end]

df_char_all = pd.read_feather(
    f'../data/data_in_all_{update_date}/characteristics_impute_selected{panel_num}_with_Equity_{update_date}_xret.feather').set_index('trd_exctn_dt').loc[time_start:time_end]
r_i = df_char_all[['complete_cusip','excess_ret']].reset_index().pivot(index='trd_exctn_dt', columns='complete_cusip', values='excess_ret')
r_i = r_i.fillna(0)
r_i_in = r_i[r_i.index < time_split]
r_i_out = r_i[r_i.index >= time_split]

if port_type == 'ew':
    market_factor = 'MKTbond'
elif port_type == 'vw':
    market_factor = 'MKTbond_vw'
df_mkt_all = pd.read_csv(f'../data/data_in_all_{update_date}/Replicated_Bond_risk_factors_bbw4.csv',index_col=0)[market_factor].loc[time_start:time_end]

df_mkt_all = np.array(df_mkt_all)
df_return_all = np.array(df_char_all['excess_ret']).reshape((T_ALL, -1))
df_xret_all = df_return_all 


if char_type == 'bond':
    list_char = list_bond
    df_char_all = df_char_all.loc[:, list_char]
elif char_type == 'bond_equity_option':
    list_char = list_bond + list_equity + list_option
    df_char_all = df_char_all.loc[:, list_char]
elif char_type == 'bond_equity':
    list_char = list_bond + list_equity
    df_char_all = df_char_all.loc[:, list_char]
elif char_type == 'equity':
    list_char = list_equity
    df_char_all = df_char_all.loc[:, list_char]

df_char_all = np.array(df_char_all).reshape((T_ALL, panel_num, -1))

num_char = df_char_all.shape[2]
list_layer = [num_char//2,num_char//4,num_char//8,num_char//16]

dict_bm = {}
dict_bm['CAPM_eqctrl'] = df_mkt_all.reshape(-1,1)
dict_bm['CAPM_group'] = df_mkt_all.reshape(-1,1)
dict_bm['CAPM'] = df_mkt_all.reshape(-1,1)

output_dir = './results_{}/{}_validation_with_batch_{}_{}/{}'.format(char_type, port_type, batch_size, cross_validate, bm_type)
os.makedirs(output_dir, exist_ok=True)

set_seed(seed)

if cross_validate == 'part_1':
    data_input = dict(bond_return_ins = torch.from_numpy(df_xret_all[odd_months,:]).float(),
                      bond_return_oos = torch.from_numpy(df_xret_all[even_months,:]).float(),
                      num_firm = df_char_all.shape[1],
                      num_feature = num_char)
    df_benchmark_train = dict_bm[bm_type][odd_months]
    df_benchmark_valid = dict_bm[bm_type][even_months]
        
    char_train = df_char_all[odd_months,:,:]
    char_valid = df_char_all[even_months,:,:]

elif cross_validate == 'part_2':
    data_input = dict(bond_return_ins = torch.from_numpy(df_xret_all[even_months,:]).float(),
                      bond_return_oos = torch.from_numpy(df_xret_all[odd_months,:]).float(),
                      num_firm = df_char_all.shape[1],
                      num_feature = num_char)
    df_benchmark_train = dict_bm[bm_type][even_months]
    df_benchmark_valid = dict_bm[bm_type][odd_months]
        
    char_train = df_char_all[even_months,:,:]
    char_valid = df_char_all[odd_months,:,:]

tensor_input_train = torch.from_numpy(char_train).float()
tensor_input_valid = torch.from_numpy(char_valid).float()

for dropout_rate in list_dropout:
    set_seed(seed)
    data_input['dropout_rate'] = dropout_rate

    for a1_a2 in list_a1_a2:
        set_seed(seed)
        a1, a2 = a1_a2
        data_input['a1'] = a1
        data_input['a2'] = a2

        for n_layer in range(i, i + 1):
            set_seed(seed)
            input_layer = list_layer[:n_layer].copy()
            data_input['layers'] = input_layer
            for n_deep_factors in range(1, 2):
                set_seed(seed)
                if n_deep_factors >= 2:
                    temp_ins = torch.cat([data_input['factor_ins'], F_train],axis=1)
                    data_input['factor_ins'] = torch.from_numpy(temp_ins.data.numpy()).float()
                    temp_oos = torch.cat([data_input['factor_oos'], F_valid],axis=1)
                    data_input['factor_oos'] = torch.from_numpy(temp_oos.data.numpy()).float()
                else:
                    data_input['factor_ins'] = torch.from_numpy(df_benchmark_train).float()
                    data_input['factor_oos'] = torch.from_numpy(df_benchmark_valid).float()
                
                ##############################################
                ##### Start to train and valid the model #####
                ##############################################
                
                list_loss_train,list_loss_valid,F_train,F_valid,best_loss = run_train_valid_with_batch(data_input, tensor_input_train, tensor_input_valid, learning_rate, epochs=epochs, batch_size=batch_size)
                

                F_whole_np = torch.cat([F_train, F_valid]).data.numpy()

                if n_deep_factors >= 2:
                    FFF_valid = np.concatenate([FFF_valid, F_valid.reshape(-1,1).data.numpy()],axis=1)
                else:
                    FFF_valid = np.concatenate([df_benchmark_valid, F_valid.reshape(-1,1).data.numpy()],axis=1)

                loss_valid = - np.mean(FFF_valid,axis=0)@np.linalg.inv(np.cov(FFF_valid.T))@np.mean(FFF_valid,axis=0)

                np.savetxt(
                    '{}/factor_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.txt'.format(
                        output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed), F_whole_np)

                np.savetxt(
                    '{}/list_loss_valid_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.txt'.format(
                        output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed), list_loss_valid)

                np.savetxt(
                    '{}/list_loss_train_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.txt'.format(
                        output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed), list_loss_train)

                np.savetxt(
                    '{}/loss_valid_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.txt'.format(
                        output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed), [loss_valid])

                np.savetxt(
                    '{}/loss_best_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.txt'.format(
                        output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed), [best_loss])

                print(f'Save_success for n_layer = {n_layer}, n_deep_factors = {n_deep_factors}, learning_rate = {learning_rate}, dropout_rate = {dropout_rate}, a1 = {a1}, a2 = {a2}, seed = {seed}')
