import pandas as pd

benchmark_list = ['CAPM']
chars_use_list = ['bond_equity_option'] 
port_type_list = ['ew']
cross_validation_list = ['part_1', 'part_2']

max_layers = 3
deep_factors_num = 1

list_learning = [0.001, 0.005, 0.01, 0.05, 0.1]
list_gamma = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]

time_end = '2023-08-31'
time_split = '2014-07-31'
time_start = '2004-07-31'

T_INS = 120
T_OOS = len(pd.date_range(start=time_split, end=time_end, freq='M'))
T_VALID = 60

seed = 666

epochs = 400
batches = 30

update_date = "final"

trans_cost = 0.0015 # 15 bps

panel_num = 3000
oos_perform_list = ['2020-12-31', '2021-12-31', '2022-12-31', '2023-08-31']

