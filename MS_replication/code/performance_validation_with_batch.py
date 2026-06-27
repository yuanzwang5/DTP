import numpy as np
import pandas as pd
import os
import scipy.stats
from DL_functions_main import *
from pandas.tseries.offsets import MonthEnd
import statsmodels.api as sm
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 100)
from parameters import *
from itertools import product


sort_case = 'best_loss' 
list_char = list_bond+list_equity+list_option

T_ALL = T_INS + T_OOS
odd_months = np.arange(0, T_VALID, 1)
even_months = np.arange(T_VALID, T_INS, 1)



for bm_type in benchmark_list:
    for char_type in chars_use_list:
        
        if char_type == 'bond_equity_option':
            list_learning = [0.001, 0.005, 0.01, 0.05, 0.1]
            list_dropout = [0.05, 0.1, 0.2, 0.3]
            list_a1_a2 = [(50, 8)]
            batches = 10
        elif char_type == 'bond_equity':
            list_learning = [0.001, 0.005, 0.01, 0.05, 0.1]
            list_dropout = [0.05, 0.1, 0.2, 0.3]
            list_a1_a2 = [(50, 5)]
            batches = 10
        elif char_type == 'bond':
            list_learning = [0.001, 0.005, 0.01, 0.05, 0.1]
            list_dropout = [0.05, 0.1, 0.2, 0.3]
            list_a1_a2 = [(100, 10)]
            batches = 10
        elif char_type == 'equity':
            list_learning = [0.005, 0.01, 0.05, 0.1, 0.5]
            list_dropout = [0.05, 0.1, 0.2, 0.3]
            list_a1_a2 = [(200, 10)]
            batches = 30

        for port_type in port_type_list:

            save_dir = './results_{}/{}_with_batch/{}'.format(char_type, port_type, bm_type)
            os.makedirs(save_dir, exist_ok=True)

            if port_type == 'ew':
                market_factor = 'MKTbond'
            elif port_type == 'vw':
                market_factor = 'MKTbond_vw'
            df_mkt_all = pd.read_csv(f'../data/data_in_all_{update_date}/Replicated_Bond_risk_factors_bbw4.csv',index_col=0)[market_factor].loc[time_start:time_end]
            df_mkt_all = np.array(df_mkt_all).reshape(-1,1)

            df_bbw = pd.read_csv(f'../data/data_in_all_{update_date}/Replicated_Bond_risk_factors_bbw4.csv',index_col=0)
            df_bbw_all = df_bbw.loc[time_start:time_end].values
            df_bbw_all[:,0] = df_mkt_all.reshape(-1)

            dict_bm = {}
            dict_bm['CAPM'] = df_mkt_all.reshape(-1,1)
            dict_bm['BBW'] = df_bbw_all.copy()


            def TP2(df_return, lam):

                m, n = df_return.shape
                q = np.mean(df_return)
                p = np.cov(df_return.T)
                y = df_return

                w_1 = np.linalg.inv(p + np.eye(n) * lam).dot(q + np.ones(n)*lam)
                w_1 = w_1/np.sum(w_1)
                
                return w_1

            # 1. Summarize the loss of each parameter combination
            oos_SR = {}
            for n_layer in range(1, max_layers+1):

                loss_frame = []
                for n_deep_factors in range(1, 2):
                    key = '{}_{}'.format(n_layer,n_deep_factors)
                    oos_SR[key] = pd.DataFrame(index = list_learning, columns = pd.MultiIndex.from_product([list_dropout, list_a1_a2], names=['dropout_rate', 'a1_a2']))
                    for learning_rate in list_learning:
                        for dropout_rate in list_dropout:
                            for a1_a2 in list_a1_a2:
                                a1, a2 = a1_a2

                                cross_valid_loss, cross_best_loss_check = [], []
                                for cross_validate in cross_validation_list:
                                    output_dir = './results_{}/{}_validation_with_batch_{}_{}/{}'.format(char_type, port_type, batches, cross_validate, bm_type)

                                    file_name = '{}/factor_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.txt'.format(
                                            output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed)

                                    loss_name = '{}/loss_valid_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.txt'.format(
                                            output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed)

                                    best_loss = '{}/loss_best_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.txt'.format(
                                            output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed)

                                    if os.path.exists(loss_name):
                                        if cross_validate == 'part_1':
                                            df_benchmark_oos = dict_bm[bm_type][even_months,:]
                                            F_is = pd.read_table(file_name,sep='\s+',header=None).values[odd_months]
                                            F_oos = pd.read_table(file_name,sep='\s+',header=None).values[even_months]
                                            FFF_oos = np.concatenate([df_benchmark_oos,F_oos.reshape(-1,1)],axis=1)
                                            FFF_is = np.concatenate([dict_bm[bm_type][odd_months,:],F_is.reshape(-1,1)],axis=1)
                                        elif cross_validate == 'part_2':
                                            df_benchmark_oos = dict_bm[bm_type][odd_months,:]
                                            F_oos = pd.read_table(file_name,sep='\s+',header=None).values[odd_months]
                                            F_is = pd.read_table(file_name,sep='\s+',header=None).values[even_months]
                                            FFF_oos = np.concatenate([df_benchmark_oos,F_oos.reshape(-1,1)],axis=1)
                                            FFF_is = np.concatenate([dict_bm[bm_type][even_months,:],F_is.reshape(-1,1)],axis=1)

                                        loss_best = pd.read_table(best_loss,sep='\s+',header=None).values[0]
                                        cross_valid_loss.append(loss_best)
                                        cross_best_loss_check.append(loss_best)
                                    else:
                                        cross_valid_loss.append(np.nan)
                                        cross_best_loss_check.append(np.nan)
                                print(n_layer,learning_rate,dropout_rate,a1,a2,cross_valid_loss,np.round(np.mean(cross_valid_loss),4),cross_best_loss_check,np.round(np.mean(cross_best_loss_check),4))
                                oos_SR[key].loc[learning_rate,(dropout_rate,a1_a2)] = np.round(np.mean(cross_valid_loss),4)
                                cross_valid_loss.append(np.mean(cross_valid_loss))
                                cross_best_loss_check.append(np.mean(cross_best_loss_check))

                                this_summary = pd.DataFrame([[learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors] + cross_valid_loss + cross_best_loss_check], columns=['learning_rate', 'dropout_rate', 'a1', 'a2', 'n_layer', 'n_deep_factors'] + [f'cross_valid_part_{i}' for i in range(1, len(cross_validation_list)+1)] + ['mean_loss'] + [f'best_loss_part_{i}' for i in range(1, len(cross_validation_list)+1)] + ['mean_best_loss'])
                                loss_frame.append(this_summary)

                df_loss = pd.concat(loss_frame, ignore_index=True)
                if sort_case == 'mean_loss':
                    df_loss = df_loss.sort_values(by=['mean_loss'], ascending=[True]).reset_index(drop=True)
                elif sort_case == 'best_loss':
                    df_loss = df_loss.sort_values(by=['mean_best_loss'], ascending=[True]).reset_index(drop=True)
                df_loss.to_csv(f'{save_dir}/validation_loss_summary_{n_layer}_layer_seed{seed}.csv', index=False, encoding='utf_8_sig')

            def return_index_columns(df):
                min_value = df.values.min()
                min_index = df.values.argmin()
                min_row, min_col = divmod(min_index, df.shape[1])
                min_row_name = df.index[min_row]
                min_col_name = df.columns[min_col]
                return min_value,min_row_name,min_col_name

            for n_layer in range(1, max_layers+1):
                for n_deep_factors in range(1, 2):
                    min_value, min_row_name, min_col_name = return_index_columns(oos_SR[f'{n_layer}_{n_deep_factors}'])
                    print(f'Layer: {n_layer}, Deep_Factor: {n_deep_factors}, the lr:{min_row_name}, dropout_rate:{min_col_name}, min value:{min_value}')


            # 2. Best parameter combination
            best_frame = []
            for n_layer in range(1, max_layers+1):
                df_loss = pd.read_csv(f'{save_dir}/validation_loss_summary_{n_layer}_layer_seed{seed}.csv')
                this_sub_loss = df_loss[df_loss['n_deep_factors'] == 1]
                if sort_case == 'mean_loss':
                    this_sub_loss = this_sub_loss.sort_values(by=['mean_loss'], ascending=[True]).reset_index(drop=True)
                elif sort_case == 'best_loss':
                    this_sub_loss = this_sub_loss.sort_values(by=['mean_best_loss'], ascending=[True]).reset_index(drop=True)
                best_loss = pd.DataFrame(this_sub_loss.iloc[0]).T
                best_frame.append(best_loss)
            df_best = pd.concat(best_frame, ignore_index=True)
            df_best = df_best[['n_layer', 'n_deep_factors', 'learning_rate', 'dropout_rate', 'a1', 'a2', 'mean_best_loss'] + [f'best_loss_part_{i}' for i in range(1, len(cross_validation_list)+1)] + ['mean_loss'] + [f'cross_valid_part_{i}' for i in range(1, len(cross_validation_list)+1)]]
            df_best.to_csv(f'{save_dir}/best_summary_seed{seed}.csv', index=False, encoding='utf_8_sig')
            print(df_best)
