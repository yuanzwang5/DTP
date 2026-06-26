import numpy as np
import pandas as pd
import random
import copy
import torch
import torch.nn.functional as F 
from torch.nn import init
from torch.utils.data import DataLoader, TensorDataset

def cal_portfolio_vw_return(df):
    temp = df[['excess_ret', 'size_value']].dropna()
    return (temp['excess_ret'] * temp['size_value']).sum() / temp['size_value'].sum()


def process_weights(w_array):

    squeezed = False
    if w_array.dim() == 3 and w_array.shape[2] == 1:
        w = w_array.squeeze(2)
        squeezed = True
    else:
        w = w_array

    dim_x, dim_y = w.shape
    result = torch.zeros_like(w)

    for i in range(dim_x):
        row = w[i]

        pos_mask = row > 0
        neg_mask = row < 0
        pos_idx = torch.where(pos_mask)[0]
        neg_idx = torch.where(neg_mask)[0]

        if pos_idx.numel() > 0:
            pos_vals = row[pos_idx]
            s_pos = pos_vals.sum()
            if s_pos != 0:
                result[i, pos_idx] = pos_vals / s_pos

        if neg_idx.numel() > 0:
            neg_vals = row[neg_idx]
            s_neg = neg_vals.sum() 
            denom = -s_neg if s_neg < 0 else s_neg
            if denom != 0:
                result[i, neg_idx] = neg_vals / denom

    return result

class Net_SR(torch.nn.Module):  
    def __init__(self, data_input):
        super(Net_SR, self).__init__()     
        self.bond_ret_ins = data_input['bond_return_ins']
        self.bond_ret_oos = data_input['bond_return_oos']
        self.layers = data_input['layers']
        self.sign = None
        n_firm = data_input['num_firm']
        n_feature = data_input['num_feature']
        dropout_rate = data_input['dropout_rate']

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
                sample_period=False):
        
        if bond_ret_ins == None:
            bond_ret_ins = self.bond_ret_ins
        if bond_ret_oos == None:
            bond_ret_oos = self.bond_ret_oos

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

        w_ = process_weights(x)

        if sample_period =='oos':
            f_ = (w_*bond_ret_oos).sum(axis=1).reshape(-1,1)
            f_ = f_*self.sign
            loss = - f_.mean()/f_.std()
            return loss, f_, w_, x.reshape(x.shape[0],x.shape[1])
        
        else:
            f_ = (w_*bond_ret_ins).sum(axis=1).reshape(-1,1)
            
            self.sign = torch.sign(f_.mean())
            f_ = f_*self.sign
            loss = - f_.mean()/f_.std()
            return loss, f_, w_, x.reshape(x.shape[0],x.shape[1])

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

    bond_ret_ins = data_input['bond_return_ins']
    bond_ret_oos = data_input['bond_return_oos']

    train_dataset = TensorDataset(tensor_input_train, bond_ret_ins)
    valid_dataset = TensorDataset(tensor_input_valid, bond_ret_oos)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(1, epochs + 1):

        ##### Training #####
        net.train()
        epoch_loss_train = 0.0

        for batch in train_loader:
            batch_input_train, batch_bond_ret_train = batch
            optimizer.zero_grad()

            loss_train, _, _, _ = net(batch_input_train, bond_ret_ins=batch_bond_ret_train)
            
            epoch_loss_train += loss_train.item()
            loss_train.backward()
            optimizer.step()

        list_loss_train.append(epoch_loss_train / len(train_loader))

        ##### Validation #####
        net.eval()
        epoch_loss_valid = 0.0

        with torch.no_grad():
            for batch in valid_loader:
                batch_input_valid, batch_bond_ret_valid = batch 
                loss_valid, _, _, _ = net(batch_input_valid, bond_ret_oos=batch_bond_ret_valid, sample_period='oos')

                epoch_loss_valid += loss_valid.item()

        list_loss_valid.append(epoch_loss_valid / len(valid_loader))

        if epoch % 50 == 0:
            print()
            print(f'Epoch {epoch} - Training Loss: {epoch_loss_train / len(train_loader)} - Validation Loss: {epoch_loss_valid / len(valid_loader)}')

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
        _, F_train, _, _ = net(tensor_input_train, bond_ret_ins=bond_ret_ins)
        _, F_valid, _, _ = net(tensor_input_valid, bond_ret_oos=bond_ret_oos, sample_period="oos")


    print(f'Epoch {epoch}/{epochs} - Training Loss: {epoch_loss_train / len(train_loader)} - Validation Loss: {epoch_loss_valid / len(valid_loader)}')

    return list_loss_train, list_loss_valid, F_train, F_valid, best_loss


def run_ins_oos_with_batch(data_input,
                           tensor_input_ins, tensor_input_oos,
                           learning_rate, 
                           patience=20, epochs=30, min_delta=0.01, batch_size=30):

    net = Net_SR(data_input)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.0)

    best_loss = float('inf')
    no_improvement_count = 0

    list_loss_ins = []

    # Extract useful data from data_input
    bond_ret_ins = data_input['bond_return_ins']

    # Create DataLoader for batching
    ins_dataset = TensorDataset(tensor_input_ins, bond_ret_ins)
    ins_loader = DataLoader(ins_dataset, batch_size=batch_size, shuffle=True)

    criterion = torch.nn.MSELoss()

    for epoch in range(1, epochs + 1):

        ##### INS #####
        net.train()
        epoch_loss_ins = 0.0

        for batch in ins_loader:
            batch_input_ins, batch_bond_ret_ins = batch
            optimizer.zero_grad()

            loss_ins, _, _, _ = net(batch_input_ins, bond_ret_ins=batch_bond_ret_ins)

            epoch_loss_ins += loss_ins.item()
            loss_ins.backward()
            optimizer.step()

        list_loss_ins.append(epoch_loss_ins / len(ins_loader))

        if epoch % 50 == 0:
            print()
            print(f'Epoch {epoch} - Training Loss: {epoch_loss_ins / len(ins_loader)}')
            state = True

    ##### Use best trained model to reestimate INS and OOS #####
    net.eval()
    with torch.no_grad():
        _, F_ins, weight_ins, x_ins = net(tensor_input_ins)
        _, F_oos, weight_oos, x_oos = net(tensor_input_oos, sample_period="oos")
    
    return list_loss_ins,F_ins,F_oos,weight_ins,weight_oos,x_ins,x_oos,net


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
