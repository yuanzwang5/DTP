import pandas as pd

benchmark_list = ['CAPM']
chars_use_list = ['bond_equity_option']
port_type_list = ['ew']
cross_validation_list = ['part_1', 'part_2']

max_layers = 3
deep_factors_num = 1

list_learning = [0.001, 0.005, 0.01, 0.05, 0.1]
list_dropout = [0.05, 0.1, 0.2, 0.3]
list_a1_a2 = [(100, 8)] 

time_end = '2023-08-31'
time_split = '2014-07-31'
time_start = '2004-07-31'

T_INS = 120
T_OOS = len(pd.date_range(start=time_split, end=time_end, freq='M'))
T_VALID = 60

seed = 666

epochs = 400
batches = 10

update_date = "final"

panel_num = 3000
oos_perform_list = ['2020-12-31', '2021-12-31', '2022-12-31', '2023-08-31']


list_bond = ['rating','duration', 'VaR_5%','Amihud','1-month_mom',
       'ytm',  'size', 'age', 'time2maturity',  'turnover',  
       'VaR_10%',
       'std_Amihud', 'Roll', 'BPW', 'P_HL',
       'P_FHT',  'TC_IQR', 'Range_daily', 'trades', 
       'variance', 'skewness', 'kurtosis', 
       'COSKEW', 'ISKEW', 
       'market_beta', 'market_residual_variance', 
       'term_beta', 'default_beta', 'term_default_residual_variance', 
       'drf_beta', 'crf_beta', 'lrf_beta',
       'liq_beta', 'vix_beta', 'unc_beta', 
       '6-month_mom', '12-month_mom', 'LTR_mom',
       'barQ','std_barQ_1mom','range_monthly']

list_equity = ['abr', 'acc', 'adm',
       'agr', 'alm', 'ato', 'baspread', 'beta', 'bm', 'bm_ia', 'cash',
       'cashdebt', 'cfp', 'chcsho', 'chpmia', 'chtx', 'cinvest', 'depr',
       'dolvol', 'dy', 'ep', 'gma', 'grltnoa', 'herf', 'hire', 'ill',
       'lev', 'lgr', 'maxret', 'me', 'me_ia', 'mom12m', 'mom1m', 'mom36m',
       'mom60m', 'mom6m', 'ni', 'nincr', 'noa', 'op', 'pctacc', 'pm',
       'pscore', 'rd_sale', 'rdm', 're', 'rna', 'roa', 'roe', 'rsup',
       'rvar_capm', 'rvar_ff3', 'rvar_mean', 'seas1a', 'sgr', 'sp',
       'std_dolvol', 'std_turn', 'sue', 'turn', 'zerotrade']

list_option = ['ivslope', 'ivvol', 'ivrv', 'ivrv_ratio', 'atm_civpiv',
       'skewiv', 'ivd', 'dciv', 'dpiv', 'atm_dcivpiv', 'nopt', 'so', 'dso',
       'vol',  'pba', 'pcratio', 'toi', 
        'mfvu', 'mfvd', 'rns1m', 'rnk1m', 'ivarud30', 'rns3m',
       'rnk3m', 'rns6m', 'rnk6m', 'rns9m', 'rnk9m', 'rns12m', 'rnk12m']

