import pandas as pd

list_gamma = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
list_bps = [15, 20]
frame_final = []
for bps in list_bps:
    frame_list = []
    for gamma in list_gamma:
        folder = f'{int(bps)}bps'
        df_raw = pd.read_csv(f'{folder}/output/Table_A6_SR_MDD_Alpha_trans_cost_{int(bps)}bps_gamma_{gamma}.csv')
        frame_list.append(pd.DataFrame([[f'{gamma}', df_raw.iloc[0,2]] + df_raw.iloc[0:3,3].values.tolist()+
                                       [f'{gamma}', df_raw.iloc[-1,2]] + df_raw.iloc[3:6,3].values.tolist()],
                                       columns=['ew_gamma','ew_CAPM','L=1','L=2','L=3','vw_gamma','vw_CAPM','L=1','L=2','L=3']))
    df_frame = pd.concat(frame_list, axis=0)
    blank_row = pd.DataFrame([[''] * df_frame.shape[1]], columns=df_frame.columns)
    df_frame = pd.concat([df_frame, blank_row], axis=0)
   
    frame_final.append(df_frame)

df_final = pd.concat(frame_final, axis=0)
df_final.to_csv(f'../../output/Table_A6.csv', index=False, encoding='utf_8_sig')

