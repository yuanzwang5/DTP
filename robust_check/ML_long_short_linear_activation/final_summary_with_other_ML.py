import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

import os
import statsmodels.api as sm
import sys
from parameters import *
from datetime import datetime
from pandas.tseries.offsets import MonthEnd

time_end='2020-12-31'

port_type = 'ew'
char_type = 'bond_equity_option'
bm_type = 'CAPM'

T_INS = 120
T_OOS = len(pd.date_range(start=time_split, end=time_end, freq='M'))
T_ALL = T_INS + T_OOS

port_percent = [10, 20, 30]

def create_long_short_portfolio(df, percent=0.3):
    portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)
    
    for date in df.index:
        performance = df.loc[date]
        
        top_10_percent = performance.nlargest(int(len(performance) * percent))
        bottom_10_percent = performance.nsmallest(int(len(performance) * percent))
        
        weights = pd.Series(0, index=performance.index)
        
        weights[top_10_percent.index] = 1 / len(top_10_percent)
        
        weights[bottom_10_percent.index] = -1 / len(bottom_10_percent)
        
        portfolio_weights.loc[date] = weights
    return portfolio_weights

if port_type == 'ew':
    market_factor = 'MKTbond'
elif port_type == 'vw':
    market_factor = 'MKTbond_vw'
df_mkt_all = pd.read_csv(f'../../data/data_in_all_{update_date}/Replicated_Bond_risk_factors_bbw4.csv',index_col=0)[market_factor].loc[time_start:time_end]

df_rf_all = pd.read_csv(f'../../data/data_in_all_{update_date}/one_month_bill.csv',index_col=0)
df_rf_all.index = pd.to_datetime(df_rf_all.index)+MonthEnd(0)
df_rf_all = df_rf_all.loc[time_start:time_end]

df_char_all = pd.read_feather(
    f'../../data/data_in_all_{update_date}/characteristics_impute_selected{panel_num}_with_Equity_{update_date}_xret.feather').set_index('trd_exctn_dt').loc[time_start:time_end]
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


################################################################

data_input = dict(bond_return_ins = (df_xret_all[:T_INS,:]),
                  bond_return_oos = (df_xret_all[T_INS:,:]),
                  num_firm = df_char_all.shape[1],
                  num_feature = num_char)

tensor_input_ins = (df_char_all[:T_INS,:,:])
tensor_input_oos = (df_char_all[T_INS:,:,:])
    

df_linear_table = pd.DataFrame(index=['OLS','Lasso','Ridge','PCA','PLS'], columns=[f'SR_INS_{p}' for p in port_percent]+[f'SR_OOS_{p}' for p in port_percent])

train_y = data_input['bond_return_ins'].reshape(-1)
train_x = tensor_input_ins.reshape(-1,num_char)

test_y = data_input['bond_return_oos'].reshape(-1)
test_x = tensor_input_oos.reshape(-1,num_char)

########## OLS ###########

print("ols regression All")

regr = LinearRegression().fit(train_x, train_y)
train_y_pred = regr.predict(train_x)
df_train_y_pred = pd.DataFrame(train_y_pred.reshape(T_INS, panel_num))
df_train_y_pred.index = df_rf_all[:T_INS].index

test_y_pred = regr.predict(test_x)
df_test_y_pred = pd.DataFrame(test_y_pred.reshape(T_OOS, panel_num))
df_test_y_pred.index = df_rf_all[T_INS:].index

sr_ins, sr_oos = [], []
for p in port_percent:
    portfolio_weights_ins = create_long_short_portfolio(df_train_y_pred, percent=p/100)
    portfolio_weights_oos = create_long_short_portfolio(df_test_y_pred, percent=p/100)
    temp_ins = (portfolio_weights_ins*data_input['bond_return_ins']).sum(axis=1).loc[:time_split]
    temp_oos = (portfolio_weights_oos*data_input['bond_return_oos']).sum(axis=1).loc[:time_end]
    
    sr_ins.append(np.sqrt(12)*temp_ins.mean()/temp_ins.std())
    sr_oos.append(np.sqrt(12)*temp_oos.mean()/temp_oos.std())

df_linear_table.loc['OLS'] = sr_ins + sr_oos
df_linear_table

df_mkt_oos = df_mkt_all.loc[time_split:time_end]
df_mkt_oos.mean()/df_mkt_oos.std()*np.sqrt(12)

########## Lasso ###########

# 2. LassoCV for pool
print("LASSO")

param_alpha = np.exp(np.arange(-4, 4, step=.5))

reg = LassoCV(cv=3,verbose=0, n_jobs=-1, selection='random', max_iter=10000).fit(train_x, train_y)

train_y_pred = reg.predict(train_x)
df_train_y_pred = pd.DataFrame(train_y_pred.reshape(T_INS, panel_num))
df_train_y_pred.index = df_rf_all[:T_INS].index

test_y_pred = reg.predict(test_x)
df_test_y_pred = pd.DataFrame(test_y_pred.reshape(T_OOS, panel_num))
df_test_y_pred.index = df_rf_all[T_INS:].index

sr_ins, sr_oos = [], []
for p in port_percent:
    portfolio_weights_ins = create_long_short_portfolio(df_train_y_pred, percent=p/100)
    portfolio_weights_oos = create_long_short_portfolio(df_test_y_pred, percent=p/100)
    temp_ins = (portfolio_weights_ins*data_input['bond_return_ins']).sum(axis=1).loc[:time_split]
    temp_oos = (portfolio_weights_oos*data_input['bond_return_oos']).sum(axis=1).loc[:time_end]
    
    sr_ins.append(np.sqrt(12)*temp_ins.mean()/temp_ins.std())
    sr_oos.append(np.sqrt(12)*temp_oos.mean()/temp_oos.std())

df_linear_table.loc['Lasso'] = sr_ins + sr_oos

########## Ridge ###########

alphas = np.logspace(-7, 5, 30)

reg = RidgeCV(alphas=alphas, cv=3, scoring='neg_mean_squared_error').fit(train_x, train_y)

train_y_pred = reg.predict(train_x)
df_train_y_pred = pd.DataFrame(train_y_pred.reshape(T_INS, panel_num))
df_train_y_pred.index = df_rf_all[:T_INS].index

test_y_pred = reg.predict(test_x)
df_test_y_pred = pd.DataFrame(test_y_pred.reshape(T_OOS, panel_num))
df_test_y_pred.index = df_rf_all.loc[time_split:].index

sr_ins, sr_oos = [], []
for p in port_percent:
    portfolio_weights_ins = create_long_short_portfolio(df_train_y_pred, percent=p/100)
    portfolio_weights_oos = create_long_short_portfolio(df_test_y_pred, percent=p/100)
    temp_ins = (portfolio_weights_ins*data_input['bond_return_ins']).sum(axis=1).loc[:time_split]
    temp_oos = (portfolio_weights_oos*data_input['bond_return_oos']).sum(axis=1).loc[:time_end]
    
    sr_ins.append(np.sqrt(12)*temp_ins.mean()/temp_ins.std())
    sr_oos.append(np.sqrt(12)*temp_oos.mean()/temp_oos.std())

df_linear_table.loc['Ridge'] = sr_ins + sr_oos


########### PCA ##########

print("PCA 5")

param_grid_pipe_pca5 = {'pca__n_components': [5]}

pipe = Pipeline(steps=[('pca', PCA()), ('reg', LinearRegression())])
search = GridSearchCV(estimator=pipe, cv=3, param_grid=param_grid_pipe_pca5, n_jobs=-1).fit(train_x,train_y)

train_y_pred = search.predict(train_x)
df_train_y_pred = pd.DataFrame(train_y_pred.reshape(T_INS, panel_num))
df_train_y_pred.index = df_rf_all[:T_INS].index

test_y_pred = search.predict(test_x)
df_test_y_pred = pd.DataFrame(test_y_pred.reshape(T_OOS, panel_num))
df_test_y_pred.index = df_rf_all[T_INS:].index

sr_ins, sr_oos = [], []
for p in port_percent:
    portfolio_weights_ins = create_long_short_portfolio(df_train_y_pred, percent=p/100)
    portfolio_weights_oos = create_long_short_portfolio(df_test_y_pred, percent=p/100)
    temp_ins = (portfolio_weights_ins*data_input['bond_return_ins']).sum(axis=1).loc[:time_split]
    temp_oos = (portfolio_weights_oos*data_input['bond_return_oos']).sum(axis=1).loc[:time_end]
    
    sr_ins.append(np.sqrt(12)*temp_ins.mean()/temp_ins.std())
    sr_oos.append(np.sqrt(12)*temp_oos.mean()/temp_oos.std())

df_linear_table.loc['PCA'] = sr_ins + sr_oos

df_linear_table = df_linear_table.drop(index='PLS')
df_linear_table.loc['MKT'] = [df_mkt_all.loc[:time_split].mean()/df_mkt_all.loc[:time_split].std()*np.sqrt(12)]*len(port_percent) + [df_mkt_oos.mean()/df_mkt_oos.std()*np.sqrt(12)]*len(port_percent)


########### NN ##########

save_folder = 'predict_return'

for n_layer in range(1, max_layers+1):
    print(f"n_layer: {n_layer}")

    n_deep_factors = 1

    output_dir = './{}/results_{}/{}_with_batch/{}'.format(save_folder, char_type, port_type, bm_type)

    # Load in best parameters combination determined by Equal-Weighted case
    ew_dir = './{}/results_{}/ew_with_batch/{}'.format(save_folder, char_type, bm_type)
    df_best = pd.read_csv(f'{ew_dir}/best_summary_seed{seed}.csv')

    this_params = df_best[(df_best['n_layer'] == n_layer)]
    learning_rate = this_params['learning_rate'].values[0]
    dropout_rate = this_params['dropout_rate'].values[0]
    a1 = this_params['a1'].values[0]
    a2 = this_params['a2'].values[0]

    file_ret_pred_ins = '{}/ret_pred_ins_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.npy'.format(
            output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed)

    file_ret_pred_oos = '{}/ret_pred_oos_lr{}_dr{}_a1{}_a2{}_L{}_D{}_seed{}.npy'.format(
            output_dir, learning_rate, dropout_rate, a1, a2, n_layer, n_deep_factors, seed)

    df_train_y_pred = pd.DataFrame(np.load(file_ret_pred_ins))
    df_train_y_pred.index = df_rf_all[:T_INS].index

    df_test_y_pred = pd.DataFrame(np.load(file_ret_pred_oos)[:T_OOS])
    df_test_y_pred.index = df_rf_all[T_INS:].index

    sr_ins, sr_oos = [], []
    for p in port_percent:
        portfolio_weights_ins = create_long_short_portfolio(df_train_y_pred, percent=p/100)
        portfolio_weights_oos = create_long_short_portfolio(df_test_y_pred, percent=p/100)
        temp_ins = (portfolio_weights_ins*data_input['bond_return_ins']).sum(axis=1).loc[:time_split]
        temp_oos = (portfolio_weights_oos*data_input['bond_return_oos']).sum(axis=1).loc[:time_end]
        
        sr_ins.append(np.sqrt(12)*temp_ins.mean()/temp_ins.std())
        sr_oos.append(np.sqrt(12)*temp_oos.mean()/temp_oos.std())
    
    df_linear_table.loc[f'NN{n_layer}'] = sr_ins + sr_oos

table_6_Panel_A = df_linear_table.T
table_6_Panel_A = table_6_Panel_A[['NN1','NN2','NN3','Lasso','Ridge','PCA']].astype(float).round(2)
table_6_Panel_A.to_csv('Table_6_Panel_A_ML_LS_Portfolios.csv', index=True, encoding='utf_8_sig')

