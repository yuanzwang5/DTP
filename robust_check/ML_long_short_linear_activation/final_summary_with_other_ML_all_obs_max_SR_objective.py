import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV, RidgeCV, Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

import os
import statsmodels.api as sm
import sys
from parameters import *
from datetime import datetime
from pandas.tseries.offsets import MonthEnd
current_time = datetime.now().strftime('%m%d')
current_time

time_end='2020-12-31'

port_type = 'ew'
char_type = 'bond_equity_option'
bm_type = 'CAPM'

T_INS = 120
T_OOS = len(pd.date_range(start=time_split, end=time_end, freq='M'))
T_ALL = T_INS + T_OOS

def create_weights(df_pred):
    df_pred = df_pred.astype(float)
    weights = pd.DataFrame(0.0, index=df_pred.index, columns=df_pred.columns)
    for idx in df_pred.index:
        row = df_pred.loc[idx]
        pos = row[row > 0]
        neg = row[row < 0]
        if len(pos) > 0:
            s_pos = pos.sum()
            if s_pos != 0:
                weights.loc[idx, pos.index] = pos / s_pos
        if len(neg) > 0:
            s_neg = neg.sum()  
            denom = -s_neg if s_neg < 0 else s_neg
            if denom != 0:
                weights.loc[idx, neg.index] = neg / denom  # sums to -1
    return weights

def cv_select_by_sr(estimator_constructor, param_list, tensor_input_ins, data_input, df_rf_ins_index, panel_num, time_split_index=None):

    T_ins = tensor_input_ins.shape[0]
    # split indices into 3 folds (contiguous by months)
    sizes = [T_ins//3, T_ins//3, T_ins - 2*(T_ins//3)]
    folds = []
    start = 0
    for s in sizes:
        folds.append(list(range(start, start+s)))
        start += s

    def eval_params(params):
        sr_list = []
        for k in range(3):
            val_idx = folds[k]
            train_idx = [i for j in range(3) if j!=k for i in folds[j]]

            X_train = tensor_input_ins[train_idx,:,:].reshape(-1, tensor_input_ins.shape[2])
            y_train = data_input['bond_return_ins'][train_idx,:].reshape(-1)

            X_val = tensor_input_ins[val_idx,:,:].reshape(-1, tensor_input_ins.shape[2])
            if params is None:
                est = estimator_constructor()
            else:
                est = estimator_constructor(**params)
            est.fit(X_train, y_train)

            y_val_pred = est.predict(X_val)
            df_val_pred = pd.DataFrame(y_val_pred.reshape(len(val_idx), panel_num))
            weights_val = create_weights(df_val_pred)
            returns_val = (weights_val.values * data_input['bond_return_ins'][val_idx,:]).sum(axis=1)
            if returns_val.std() == 0:
                sr = 0.0
            else:
                sr = np.sqrt(12) * returns_val.mean() / returns_val.std()
            sr_list.append(sr)
        return np.mean(sr_list)

    best_score = -np.inf
    best_params = None
    for p in param_list:
        score = eval_params(p)
        if score > best_score:
            best_score = score
            best_params = p

    # fit on full training set with best params
    X_full = tensor_input_ins.reshape(-1, tensor_input_ins.shape[2])
    y_full = data_input['bond_return_ins'].reshape(-1)
    if best_params is None:
        best_est = estimator_constructor()
    else:
        best_est = estimator_constructor(**best_params)
    best_est.fit(X_full, y_full)
    y_pred_full = best_est.predict(X_full)
    df_train_pred = pd.DataFrame(y_pred_full.reshape(T_ins, panel_num))
    df_train_pred.index = df_rf_ins_index
    return best_params, df_train_pred, best_est

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


df_char_all_2 = df_char_all.copy()
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
    

df_linear_table = pd.DataFrame(index=['OLS','Lasso','Ridge','PCA','PLS'], columns=['SR_INS', 'SR_OOS'])
df_linear_table

train_y = data_input['bond_return_ins'].reshape(-1)
train_x = tensor_input_ins.reshape(-1,num_char)

test_y = data_input['bond_return_oos'].reshape(-1)
test_x = tensor_input_oos.reshape(-1,num_char)

####### OLS #######

print("ols regression All (SR-driven 3-fold CV)")

# OLS: no hyperparameters, use cv_select_by_sr with [None]
best_params, df_train_y_pred, best_est = cv_select_by_sr(LinearRegression, [None], tensor_input_ins, data_input, df_rf_all[:T_INS].index, panel_num)

# predict on test (OOS)
df_test_y_pred = pd.DataFrame(best_est.predict(test_x).reshape(T_OOS, panel_num))
df_test_y_pred.index = df_rf_all[T_INS:].index

portfolio_weights_ins = create_weights(df_train_y_pred)
portfolio_weights_oos = create_weights(df_test_y_pred)
temp_ins = (portfolio_weights_ins*data_input['bond_return_ins']).sum(axis=1).loc[:time_split]
temp_oos = (portfolio_weights_oos*data_input['bond_return_oos']).sum(axis=1).loc[:time_end]
indicator = 1 if temp_ins.mean() >= 0 else -1
temp_oos = temp_oos * indicator

sr_ins = np.sqrt(12)*temp_ins.mean()/temp_ins.std() if temp_ins.std() != 0 else 0.0
sr_oos = np.sqrt(12)*temp_oos.mean()/temp_oos.std() if temp_oos.std() != 0 else 0.0
df_linear_table.loc['OLS'] = [sr_ins, sr_oos]

df_mkt_oos = df_mkt_all.loc[time_split:time_end]

####### Lasso #########

# 2. LassoCV for pool
print("LASSO (SR-driven 3-fold CV)")

param_alpha = np.exp(np.arange(-4, 4, step=.5))
param_list = [{'alpha': float(a)} for a in param_alpha]

best_params, df_train_y_pred, best_est = cv_select_by_sr(Lasso, param_list, tensor_input_ins, data_input, df_rf_all[:T_INS].index, panel_num)

df_test_y_pred = pd.DataFrame(best_est.predict(test_x).reshape(T_OOS, panel_num))
df_test_y_pred.index = df_rf_all[T_INS:].index

portfolio_weights_ins = create_weights(df_train_y_pred)
portfolio_weights_oos = create_weights(df_test_y_pred)
temp_ins = (portfolio_weights_ins*data_input['bond_return_ins']).sum(axis=1).loc[:time_split]
temp_oos = (portfolio_weights_oos*data_input['bond_return_oos']).sum(axis=1).loc[:time_end]
indicator = 1 if temp_ins.mean() >= 0 else -1
temp_oos = temp_oos * indicator

sr_ins = np.sqrt(12)*temp_ins.mean()/temp_ins.std() if temp_ins.std() != 0 else 0.0
sr_oos = np.sqrt(12)*temp_oos.mean()/temp_oos.std() if temp_oos.std() != 0 else 0.0
df_linear_table.loc['Lasso'] = [sr_ins, sr_oos]

######### Ridge ########

alphas = np.logspace(-7, 5, 30)

param_list = [{'alpha': float(a)} for a in alphas]
best_params, df_train_y_pred, best_est = cv_select_by_sr(Ridge, param_list, tensor_input_ins, data_input, df_rf_all[:T_INS].index, panel_num)

df_test_y_pred = pd.DataFrame(best_est.predict(test_x).reshape(T_OOS, panel_num))
df_test_y_pred.index = df_rf_all[T_INS:].index

portfolio_weights_ins = create_weights(df_train_y_pred)
portfolio_weights_oos = create_weights(df_test_y_pred)
temp_ins = (portfolio_weights_ins*data_input['bond_return_ins']).sum(axis=1).loc[:time_split]
temp_oos = (portfolio_weights_oos*data_input['bond_return_oos']).sum(axis=1).loc[:time_end]
indicator = 1 if temp_ins.mean() >= 0 else -1
temp_oos = temp_oos * indicator

sr_ins = np.sqrt(12)*temp_ins.mean()/temp_ins.std() if temp_ins.std() != 0 else 0.0
sr_oos = np.sqrt(12)*temp_oos.mean()/temp_oos.std() if temp_oos.std() != 0 else 0.0
df_linear_table.loc['Ridge'] = [sr_ins, sr_oos]


######## PCA ########

print("PCA (SR-driven 3-fold CV)")

param_list = [{'n_components': 5}]
def pca_constructor(n_components=5):
    return Pipeline(steps=[('pca', PCA(n_components=n_components)), ('reg', LinearRegression())])

best_params, df_train_y_pred, best_est = cv_select_by_sr(lambda **kw: pca_constructor(**kw), param_list, tensor_input_ins, data_input, df_rf_all[:T_INS].index, panel_num)

df_test_y_pred = pd.DataFrame(best_est.predict(test_x).reshape(T_OOS, panel_num))
df_test_y_pred.index = df_rf_all[T_INS:].index

portfolio_weights_ins = create_weights(df_train_y_pred)
portfolio_weights_oos = create_weights(df_test_y_pred)
temp_ins = (portfolio_weights_ins*data_input['bond_return_ins']).sum(axis=1).loc[:time_split]
temp_oos = (portfolio_weights_oos*data_input['bond_return_oos']).sum(axis=1).loc[:time_end]
indicator = 1 if temp_ins.mean() >= 0 else -1
temp_oos = temp_oos * indicator

sr_ins = np.sqrt(12)*temp_ins.mean()/temp_ins.std() if temp_ins.std() != 0 else 0.0
sr_oos = np.sqrt(12)*temp_oos.mean()/temp_oos.std() if temp_oos.std() != 0 else 0.0
df_linear_table.loc['PCA'] = [sr_ins, sr_oos]
df_linear_table

df_linear_table = df_linear_table.drop(index='PLS')
df_linear_table.loc['MKT'] = [df_mkt_all.loc[:time_split].mean()/df_mkt_all.loc[:time_split].std()*np.sqrt(12), df_mkt_oos.mean()/df_mkt_oos.std()*np.sqrt(12)]


######### NN ########

save_folder = 'max_SR'

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
    portfolio_weights_ins = create_weights(df_train_y_pred)
    portfolio_weights_oos = create_weights(df_test_y_pred)
    temp_ins = (portfolio_weights_ins*data_input['bond_return_ins']).sum(axis=1).loc[:time_split]
    temp_oos = (portfolio_weights_oos*data_input['bond_return_oos']).sum(axis=1).loc[:time_end]
    indicator = 1 if temp_ins.mean() >= 0 else -1
    temp_oos = temp_oos * indicator

    sr_ins.append(np.sqrt(12)*temp_ins.mean()/temp_ins.std())
    sr_oos.append(np.sqrt(12)*temp_oos.mean()/temp_oos.std())
    
    df_linear_table.loc[f'NN{n_layer}'] = sr_ins + sr_oos

table_6_Panel_A = df_linear_table.T
table_6_Panel_A = table_6_Panel_A[['NN1','NN2','NN3','Lasso','Ridge','PCA']].astype(float).round(2)
table_6_Panel_A.to_csv('Table_6_Panel_A_ML_Portfolios_ALL_OBS_max_SR.csv', index=True, encoding='utf_8_sig')

