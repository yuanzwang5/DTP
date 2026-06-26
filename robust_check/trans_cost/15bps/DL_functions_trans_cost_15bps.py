import numpy as np
import pandas as pd
import random
import copy
import torch
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import DataLoader, TensorDataset
import pyarrow.feather as feather

def cal_portfolio_vw_return(df):
    temp = df[['excess_ret', 'size_value']].dropna()
    return (temp['excess_ret'] * temp['size_value']).sum() / temp['size_value'].sum()


def market_return(df_,market_type = 'ew'):

    def cal_portfolio_vw_return(df):
        temp = df[['monthly_return_winsorized', 'size_value']].dropna()
        return (temp['monthly_return_winsorized'] * temp['size_value']).sum() / temp['size_value'].sum()

    def cal_portfolio_ew_return(df):
        temp = df[['monthly_return_winsorized', 'size_value']].dropna()
        return (temp['monthly_return_winsorized'] ).sum() /temp['monthly_return_winsorized'].count()

    df = df_.copy()
    if market_type =='vw':

        return (cal_portfolio_vw_return(df))
    
    else:
        return (cal_portfolio_ew_return(df))

class Net_SR(torch.nn.Module):  
    def __init__(self, data_input):
        super(Net_SR, self).__init__()     
        self.bond_ret_ins = data_input['bond_return_ins']
        self.bond_ret_oos = data_input['bond_return_oos']
        self.bd_spread_ins = data_input['bd_spread_ins']
        self.bd_spread_oos = data_input['bd_spread_oos']
        self.g_bench_ins = data_input['factor_ins']
        self.g_bench_oos = data_input['factor_oos']
        self.layers = data_input['layers']
        self.sign = None
        n_firm = data_input['num_firm']
        n_feature = data_input['num_feature']
        dropout_rate = data_input['dropout_rate']
        self.a1 = data_input['a1']
        self.a2 = data_input['a2']
        self.tc = data_input['trans_cost']
        self.gamma = data_input['gamma']
        self.raw_w_df = torch.nn.Parameter(torch.tensor(0.0))

        # Hidden layers
        if len(self.layers)>= 1:
            self.hidden = torch.nn.Linear(n_feature, self.layers[0]) 
        if len(self.layers)>= 2:
            self.hidden_2 = torch.nn.Linear(self.layers[0], self.layers[1])
        if len(self.layers)>= 3:
            self.hidden_3 = torch.nn.Linear(self.layers[1], self.layers[2])
        if len(self.layers)>= 4:
            self.hidden_4 = torch.nn.Linear(self.layers[2], self.layers[3])
        self.output = torch.nn.Linear(self.layers[-1], 1)
        
        self.batchnorm = torch.nn.BatchNorm1d(n_firm, affine=False)
        self.activate = torch.nn.Tanh()
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        # Initialize weights
        self.initialization_(self.layers)
    
    def forward(self, x, 
                bond_ret_ins=None, bond_ret_oos=None, 
                bd_spread_ins=None, bd_spread_oos=None,
                g_bench_ins=None, g_bench_oos=None, 
                sample_period=False, 
                print_state =False):
        
        if bond_ret_ins == None:
            bond_ret_ins = self.bond_ret_ins
        if bond_ret_oos == None:
            bond_ret_oos = self.bond_ret_oos
        if bd_spread_ins == None:
            bd_spread_ins = self.bd_spread_ins
        if bd_spread_oos == None:
            bd_spread_oos = self.bd_spread_oos
        if g_bench_ins == None:
            g_bench_ins = self.g_bench_ins
        if g_bench_oos == None:
            g_bench_oos = self.g_bench_oos

        if len(self.layers)>= 1:
            x = (1*self.activate(self.hidden(x)))
            x = self.dropout(x)
        if len(self.layers)>= 2:
            x = (1*self.activate(self.hidden_2(x)))
            x = self.dropout(x)
        if len(self.layers)>= 3:
            x = (1*self.activate(self.hidden_3(x)))
            x = self.dropout(x)
        if len(self.layers)>= 4:
            x = (1*self.activate(self.hidden_4(x)))
            x = self.dropout(x)
        
        x = self.output(x)
        x = self.batchnorm(x)
        
        transformed_x_a = -self.a1*torch.exp(-self.a2*x)
        transformed_x_b = -self.a1*torch.exp(self.a2*x)
        
        w_ = F.softmax(transformed_x_a, dim=1) - F.softmax(transformed_x_b, dim=1)
        w_ = w_.reshape([w_.shape[0], w_.shape[1]])

        w_tilde_ = torch.empty_like(w_)
        w_tilde_[0] = w_[0].clone()
        for t in range(1, w_.shape[0]):
            w_tilde_[t] = (1.0 - self.gamma) * w_[t] + self.gamma * w_tilde_[t - 1]
        del w_

        # Construct lagged weights w_tilde_lm_
        w_tilde_lm_ = torch.cat([w_tilde_[0:1], w_tilde_[:-1]], dim=0)

        # Calculate transaction costs
        w_diff_ = torch.abs(w_tilde_ - w_tilde_lm_)

        tc_ = torch.sum(w_diff_, dim=1).reshape(-1,1) * self.tc
        print(f'Transaction Costs Mean: {tc_.mean().item()}, Std: {tc_.std().item()}')

        raw_turnover = torch.sum(w_diff_, dim=1).reshape(-1,1)
        try:
            arr = raw_turnover.detach().cpu().numpy().flatten()
            print("raw_turnover stats - mean: {:.6g}, std: {:.6g}, min: {:.6g}, max: {:.6g} \n".format(
                arr.mean(), arr.std(ddof=1), arr.min(), arr.max()
            ))
        except Exception as e:
            print("Print raw_turnover stats failed:", e)

        # ----------------------------------------------------------------------------------- #

        # map raw_w_df -> w_df in [-2, 3] via sigmoid mapping (smooth alternative to tanh)
        w_df = torch.sigmoid(self.raw_w_df) * 5.0 - 2.0
        w_df_col = w_df.view(1,1)

        # ----------------------------------------------------------------------------------- #

        if sample_period =='oos':
            # Alternate weights
            w_ = w_tilde_ * self.sign
            # Calculate out-of-sample factors after sign adjustment
            f_notc_ = ( w_ * bond_ret_oos ).sum(axis=1).reshape(-1,1)

            # Then adjust the factors by transaction costs
            f_ = f_notc_ - tc_

            # Use only the first column of g_bench_oos as the market factor (if multiple exist)
            market = g_bench_oos
            if market.ndim > 1 and market.shape[1] > 1:
                market = market[:, 0:1]

            # Construct tangency portfolio return R_tp = (1 - w_df) * market + w_df * f_
            R_tp = (1.0 - w_df_col) * market + w_df_col * f_

            # Compute Sharpe ratio (mean / std) and set loss = -SR (minimize -SR -> maximize SR)
            loss = - R_tp.mean(dim=0) / R_tp.std(dim=0)

            return loss, f_notc_, tc_, f_, w_, x.reshape(x.shape[0],x.shape[1])
        
        else:
            # First calculate in-sample factors without transaction cost
            f_raw = ( w_tilde_ * bond_ret_ins ).sum(axis=1).reshape(-1,1)
            # Judge sign based on in-sample raw factors average
            self.sign = torch.sign(f_raw.mean())
            # Alternate weights
            w_ = w_tilde_ * self.sign
            # Calculate factors after sign adjustment
            f_notc_ = ( w_ * bond_ret_ins ).sum(axis=1).reshape(-1,1)

            # Then adjust the factors by transaction costs
            f_ = f_notc_ - tc_

            # Use only the first column of g_bench_ins as the market factor (if multiple exist)
            market = g_bench_ins
            if market.ndim > 1 and market.shape[1] > 1:
                market = market[:, 0:1]

            # Construct tangency portfolio return R_tp = (1 - w_df) * market + w_df * f_
            R_tp = (1.0 - w_df_col) * market + w_df_col * f_

            print(f"w_df_col value: {w_df_col.item()} \n")

            # For logging: TP asset weights (market, deep)
            w_tp_ins = torch.cat([ (1.0 - w_df_col).reshape(-1), w_df_col.reshape(-1) ], dim=0)
            if print_state == True:
                print('w_tp_ins (market, deep):', w_tp_ins.detach().cpu().numpy())

            # Compute Sharpe ratio (mean / std) and set loss = -SR
            loss = - R_tp.mean(dim=0) / R_tp.std(dim=0)

            return loss, w_df_col, f_notc_, tc_, f_, w_, x.reshape(x.shape[0],x.shape[1])

    def initialization_(self, list_layer):

        if len(list_layer)>=1:
            init.normal_(self.hidden.weight)
            self.hidden.weight = torch.nn.Parameter(self.hidden.weight*0.01)
        if len(list_layer)>=2:
            init.normal_(self.hidden_2.weight)
            self.hidden_2.weight = torch.nn.Parameter(self.hidden_2.weight*0.01)
        if len(list_layer)>=3:
            init.normal_(self.hidden_3.weight)
            self.hidden_3.weight = torch.nn.Parameter(self.hidden_3.weight*0.01)
        if len(list_layer)>=4:
            init.normal_(self.hidden_4.weight)
            self.hidden_4.weight = torch.nn.Parameter(self.hidden_4.weight*0.01)
        init.normal_(self.output.weight)
        self.output.weight = torch.nn.Parameter(self.output.weight*0.01)


def penalty_l1_adding(net,list_layer):
    penalty = 0.0
    if len(list_layer)>=1:
        penalty = torch.norm(net.hidden.weight,p=1) 
    if len(list_layer)>=2:
        penalty = penalty + torch.norm(net.hidden_2.weight,p=1) 
    if len(list_layer)>=3:
        penalty = penalty + torch.norm(net.hidden_3.weight,p=1) 
    if len(list_layer)>=4:
        penalty = penalty + torch.norm(net.hidden_4.weight,p=1) 
    return penalty

def penalty_l2_adding(net,list_layer):
    penalty = 0.0
    if len(list_layer)>=1:
        penalty = torch.norm(net.hidden.weight,p=2) ** 2
    if len(list_layer)>=2:
        penalty = penalty + torch.norm(net.hidden_2.weight,p=2) ** 2
    if len(list_layer)>=3:
        penalty = penalty + torch.norm(net.hidden_3.weight,p=2) ** 2
    if len(list_layer)>=4:
        penalty = penalty + torch.norm(net.hidden_4.weight,p=2) ** 2
    return penalty



def run_train_valid_with_batch(data_input,
                               tensor_input_train, tensor_input_valid,
                               learning_rate, 
                               state=False, patience=20, epochs=400, min_delta=0.01, batch_size=30):
    
    net = Net_SR(data_input)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.0)

    best_loss = float('inf')
    best_model_state = None
    no_improvement_count = 0

    list_loss_train, list_loss_valid = [], []

    # Extract useful data from data_input
    bond_ret_ins = data_input['bond_return_ins']
    bond_ret_oos = data_input['bond_return_oos']
    bd_spread_ins = data_input['bd_spread_ins']
    bd_spread_oos = data_input['bd_spread_oos']
    factor_ins = data_input['factor_ins']
    factor_oos = data_input['factor_oos']

    # Create DataLoaders for batching
    train_dataset = TensorDataset(tensor_input_train, bond_ret_ins, bd_spread_ins, factor_ins)
    valid_dataset = TensorDataset(tensor_input_valid, bond_ret_oos, bd_spread_oos, factor_oos)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(1, epochs + 1):

        ##### Training #####
        net.train()
        epoch_loss_train = 0.0

        for batch in train_loader:
            batch_input_train, batch_bond_ret_train, batch_bd_spread_train, batch_factor_train = batch 
            optimizer.zero_grad()

            loss_train, _, _, _, _, _, _ = net(batch_input_train, 
                                                bond_ret_ins=batch_bond_ret_train, 
                                                bd_spread_ins=batch_bd_spread_train, 
                                                g_bench_ins=batch_factor_train, 
                                                print_state=state)

            epoch_loss_train += loss_train.item()
            loss_train.backward()
            optimizer.step()

        list_loss_train.append(epoch_loss_train / len(train_loader))

        ##### Validation #####
        net.eval()
        epoch_loss_valid = 0.0

        with torch.no_grad():
            for batch in valid_loader:
                batch_input_valid, batch_bond_ret_valid, batch_bd_spread_valid, batch_factor_valid = batch 
                loss_valid, _, _, _, _, _ = net(batch_input_valid, 
                                                bond_ret_oos=batch_bond_ret_valid, 
                                                bd_spread_oos=batch_bd_spread_valid, 
                                                g_bench_oos=batch_factor_valid, 
                                                sample_period="oos", 
                                                print_state=state)
                state = False

                epoch_loss_valid += loss_valid.item()

        list_loss_valid.append(epoch_loss_valid / len(valid_loader))

        if epoch % 50 == 0:
            print()
            print(f'Epoch {epoch} - Training Loss: {epoch_loss_train / len(train_loader)} - Validation Loss: {epoch_loss_valid / len(valid_loader)}')
            state = True

        if epoch_loss_valid / len(valid_loader) < best_loss + min_delta:
            if epoch_loss_valid / len(valid_loader) <= best_loss:
                best_loss = epoch_loss_valid / len(valid_loader)
                best_model_state = copy.deepcopy(net.state_dict()) 
            else:
                continue
        else:
            no_improvement_count += 1

        # Check if the model has not improved for the specified number of epochs
        if no_improvement_count >= patience:
            print(f'Early stopping after epoch {epoch}')
            break

    # After training, evaluate the model on the entire training and validation datasets
    net.load_state_dict(best_model_state)
    net.eval()
    with torch.no_grad():
        _, w_df_train, F_train_notc, tc_train, F_train, _, _ = net(tensor_input_train, 
                                                                    bond_ret_ins=bond_ret_ins, 
                                                                    bd_spread_ins=bd_spread_ins, 
                                                                    g_bench_ins=factor_ins, 
                                                                    print_state=False)
        _, F_valid_notc, tc_valid, F_valid, _, _ = net(tensor_input_valid, 
                                                                    bond_ret_oos=bond_ret_oos, 
                                                                    bd_spread_oos=bd_spread_oos, 
                                                                    g_bench_oos=factor_oos, 
                                                                    sample_period="oos", 
                                                                    print_state=False)


    print(f'Epoch {epoch}/{epochs} - Training Loss: {epoch_loss_train / len(train_loader)} - Validation Loss: {epoch_loss_valid / len(valid_loader)}')

    return list_loss_train, list_loss_valid, w_df_train, F_train_notc, F_valid_notc, tc_train, tc_valid, F_train, F_valid, best_loss


def run_ins_oos_with_batch(data_input,
                           tensor_input_ins, tensor_input_oos,
                           learning_rate, 
                           state=False, patience=20, epochs=400, min_delta=0.01, batch_size=30):
    
    net = Net_SR(data_input)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.0)

    best_loss = float('inf')
    best_model_state = None
    no_improvement_count = 0

    list_loss_ins = []
    list_grad = []

    # Extract useful data from data_input
    bond_ret_ins = data_input['bond_return_ins']
    bd_spread_ins = data_input['bd_spread_ins']
    factor_ins = data_input['factor_ins']

    # Create DataLoader for batching
    ins_dataset = TensorDataset(tensor_input_ins, bond_ret_ins, bd_spread_ins, factor_ins)
    ins_loader = DataLoader(ins_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):

        ##### INS #####
        net.train()
        epoch_loss_ins = 0.0

        for batch in ins_loader:
            batch_input_ins, batch_bond_ret_ins, batch_bd_spread_ins, batch_factor_ins = batch
            optimizer.zero_grad()

            loss_ins, w_df_ins, _, _, _, _, _ = net(batch_input_ins, 
                                                    bond_ret_ins=batch_bond_ret_ins, 
                                                    bd_spread_ins=batch_bd_spread_ins, 
                                                    g_bench_ins=batch_factor_ins, 
                                                    print_state=state)
            state = False

            epoch_loss_ins += loss_ins.item()
            loss_ins.backward()
            optimizer.step()

            # Note gradient
            if epoch > 2:
                if len(data_input['layers']) == 1:
                    list_grad.append((net.output.weight.grad @ net.hidden.weight.grad).data.numpy())
                elif len(data_input['layers']) == 2:
                    list_grad.append((net.output.weight.grad @ net.hidden_2.weight.grad @ net.hidden.weight.grad).data.numpy())
                elif len(data_input['layers']) == 3:
                    list_grad.append((net.output.weight.grad @ net.hidden_3.weight.grad @ net.hidden_2.weight.grad @ net.hidden.weight.grad).data.numpy())
                elif len(data_input['layers']) == 4:
                    list_grad.append((net.output.weight.grad @ net.hidden_4.weight.grad @ net.hidden_3.weight.grad @ net.hidden_2.weight.grad @ net.hidden.weight.grad).data.numpy())

        list_loss_ins.append(epoch_loss_ins / len(ins_loader))

        if epoch % 50 == 0:
            print()
            print(f'Epoch {epoch} - Training Loss: {epoch_loss_ins / len(ins_loader)}')
            state = True

        # Early stopping
        if epoch_loss_ins / len(ins_loader) < best_loss + min_delta:
            if epoch_loss_ins / len(ins_loader) <= best_loss:
                best_loss = epoch_loss_ins / len(ins_loader)
                best_model_state = copy.deepcopy(net.state_dict())
            else:
                continue
        else:
            no_improvement_count += 1

        # Check if the model has not improved for the specified number of epochs
        if no_improvement_count >= patience:
            print(f'Early stopping after epoch {epoch}')
            break

    ##### Use best trained model to reestimate INS and OOS #####
    net.load_state_dict(best_model_state) 
    net.eval()
    with torch.no_grad():
        _, w_df_ins, F_ins_notc, tc_ins, F_ins, weight_ins, x_ins = net(tensor_input_ins, print_state=state)
        _, F_oos_notc, tc_oos, F_oos, weight_oos, x_oos = net(tensor_input_oos, sample_period='oos', print_state=state)
    
    return list_loss_ins,w_df_ins,F_ins_notc,F_oos_notc,tc_ins,tc_oos,F_ins,F_oos,weight_ins,weight_oos,x_ins,x_oos,list_grad,net


def set_seed(manualSeed=666):
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)

def numpy_describe(arr):

    flattened = arr.flatten()

    summary = {
        "count": flattened.size, 
        "mean": np.mean(flattened),
        "std": np.std(flattened, ddof=1), 
        "min": np.min(flattened),
        "25%": np.percentile(flattened, 25),
        "50%": np.percentile(flattened, 50),
        "75%": np.percentile(flattened, 75),
        "max": np.max(flattened),
    }

    summary_df = pd.DataFrame.from_dict(summary, orient="index", columns=["Value"])

    return summary_df

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
