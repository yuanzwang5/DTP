import numpy as np
import pandas as pd
import statsmodels.api as sm
import imp
import os
from DL_functions_main import *
from pandas.tseries.offsets import MonthEnd
from datetime import datetime
current_time = datetime.now().strftime('%m%d')
from parameters import *







T_ALL = T_INS + T_OOS

df_rf_all = pd.read_csv(f'../data/data_in_all_{update_date}/one_month_bill.csv',index_col=0)
df_rf_all.index = pd.to_datetime(df_rf_all.index)+MonthEnd(0)
df_rf_all = df_rf_all.loc[time_start:time_end]

df_char_all = pd.read_feather(
    f'../data/data_in_all_{update_date}/characteristics_impute_selected{panel_num}_with_Equity_{update_date}_xret.feather').set_index('trd_exctn_dt').loc[time_start:time_end]
r_i = df_char_all[['complete_cusip','excess_ret']].reset_index().pivot(index='trd_exctn_dt', columns='complete_cusip', values='excess_ret')
r_i = r_i.fillna(0)
r_i_in = r_i[r_i.index < time_split]
r_i_out = r_i[r_i.index >= time_split]
df_xret_all = np.array(df_char_all['excess_ret']).reshape((T_ALL, -1))

if char_type == 'bond':
    list_char = list_bond
    df_char_all = df_char_all.loc[:, list_char]
elif char_type == 'bond_equity_option':
    list_char = list_bond + list_equity+list_option
    df_char_all = df_char_all.loc[:, list_char]
elif char_type == 'bond_equity':
    list_char = list_bond + list_equity
    df_char_all = df_char_all.loc[:, list_char]
elif char_type == 'equity':
    list_char = list_equity
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
    df_mkt_all = pd.read_csv(f'../data/data_in_all_{update_date}/Replicated_Bond_risk_factors_bbw4.csv',index_col=0)[market_factor].loc[time_start:time_end]

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
        print(w_all.shape)
            

        df_grad = pd.DataFrame(columns = list_char, index=['Multilayer_grad'])      
        df_grad.loc['Multilayer_grad',list_char] = list_grad[0][0]
        
        df_grad.to_csv('{}/grad_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.txt'.format(
                output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed))

        np.savetxt('{}/factor_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.txt'.format(
                output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed), F_whole_np)

        np.savetxt('{}/loss_path_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.txt'.format(
                output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed), list_loss_ins)

        np.savetxt('{}/weights_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.txt'.format(
                output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed), w_all)

        np.savetxt('{}/dchars_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.txt'.format(
                output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed), x_all)

        torch.save(net, '{}/torchnn_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.pt'.format(
                output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed))

        # Output hidden weights
        if len(input_layer)>=1:
            hidden_weight_1 = net.hidden.weight.data.numpy()
            print(f'hidden_weight_1 shape: {hidden_weight_1.shape}\n')
            print(f'hidden_weight_1 summary statistics:\n {numpy_describe(hidden_weight_1)}\n')
            np.savetxt('{}/hidden_weights_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.txt'.format(
                output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed), hidden_weight_1)
        if len(input_layer)>=2:
            hidden_weight_2 = net.hidden_2.weight.data.numpy()
            print(f'hidden_weight_2 shape: {hidden_weight_2.shape}\n')
            print(f'hidden_weight_2 summary statistics:\n {numpy_describe(hidden_weight_2)}\n')
            np.savetxt('{}/hidden_weights_layer_2_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.txt'.format(
                output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed), hidden_weight_2)
        if len(input_layer)>=3:
            hidden_weight_3 = net.hidden_3.weight.data.numpy()
            print(f'hidden_weight_3 shape: {hidden_weight_3.shape}\n')
            print(f'hidden_weight_3 summary statistics:\n {numpy_describe(hidden_weight_3)}\n')
            np.savetxt('{}/hidden_weights_layer_3_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.txt'.format(
                output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed), hidden_weight_3)
        if len(input_layer)>=4:
            hidden_weight_4 = net.hidden_4.weight.data.numpy()
            print(f'hidden_weight_4 shape: {hidden_weight_4.shape}\n')
            print(f'hidden_weight_4 summary statistics:\n {numpy_describe(hidden_weight_4)}\n')
            np.savetxt('{}/hidden_weights_layer_4_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.txt'.format(
                output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed), hidden_weight_4)

        print(f'Save_success for n_layer = {n_layer}, n_deep_factors = {n_deep_factors},\n learning_rate = {learning_rate},\n dropout_rate = {dropout_rate}, \n a1 = {a1}, \n a2 = {a2}, \n loss_ins = {list_loss_ins[-1]}, \n seed = {seed} \n')
