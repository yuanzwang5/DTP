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

def process_weights(w_array, sep_num=10):

    dim_x, dim_y = w_array.shape
    k = int(dim_y/sep_num)
    result = torch.zeros_like(w_array)
    
    for i in range(dim_x):
        row = w_array[i]
        
        # Separate Positive and Negative Values
        positive_mask = row > 0
        negative_mask = row < 0
        
        positive_values = row[positive_mask]
        negative_values = row[negative_mask]
        
        # Deal with Positive part 
        if len(positive_values) > 0:
            top_k_positive = min(k, len(positive_values))
            top_positive_values, top_positive_indices = torch.topk(positive_values, top_k_positive)
            positive_sum = torch.sum(top_positive_values)
            
            if positive_sum != 0:
                standardized_positive = top_positive_values / positive_sum
            else:
                standardized_positive = top_positive_values
            
            positive_orig_indices = torch.where(positive_mask)[0][top_positive_indices]
            result[i, positive_orig_indices] = standardized_positive
            
        # Deal with Negative part
        if len(negative_values) > 0:
            top_k_negative = min(k, len(negative_values))
            top_negative_values, top_negative_indices = torch.topk(negative_values, top_k_negative, largest=False)
            negative_sum = torch.sum(top_negative_values)
            
            if negative_sum != 0:
                standardized_negative = top_negative_values / abs(negative_sum)
            else:
                standardized_negative = top_negative_values
            
            negative_orig_indices = torch.where(negative_mask)[0][top_negative_indices]
            result[i, negative_orig_indices] = standardized_negative
            
    return result

class Net_SR(torch.nn.Module):  
    def __init__(self, data_input):
        super(Net_SR, self).__init__()     
        self.bond_ret_ins = data_input['bond_return_ins']
        self.bond_ret_oos = data_input['bond_return_oos']
        self.g_bench_ins = data_input['factor_ins']
        self.g_bench_oos = data_input['factor_oos']
        self.layers = data_input['layers']
        self.sign = None
        n_firm = data_input['num_firm']
        n_feature = data_input['num_feature']
        dropout_rate = data_input['dropout_rate']
        self.a1 = data_input['a1']
        self.a2 = data_input['a2']

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
                g_bench_ins=None, g_bench_oos=None, 
                sample_period=False, 
                print_state =False):
        
        if bond_ret_ins == None:
            bond_ret_ins = self.bond_ret_ins
        if bond_ret_oos == None:
            bond_ret_oos = self.bond_ret_oos
        if g_bench_ins == None:
            g_bench_ins = self.g_bench_ins
        if g_bench_oos == None:
            g_bench_oos = self.g_bench_oos

        if len(self.layers)>= 1:
            x = (1*self.hidden(x))
            x = self.dropout(x)
        if len(self.layers)>= 2:
            x = (1*self.hidden_2(x))
            x = self.dropout(x)
        if len(self.layers)>= 3:
            x = (1*self.hidden_3(x))
            x = self.dropout(x)
        if len(self.layers)>= 4:
            x = (1*self.hidden_4(x))
            x = self.dropout(x)
        
        x = self.output(x)
        x = self.batchnorm(x)
        
        transformed_x_a = -self.a1*torch.exp(-self.a2*x)
        transformed_x_b = -self.a1*torch.exp(self.a2*x)
        
        w_ = F.softmax(transformed_x_a, dim=1) - F.softmax(transformed_x_b, dim=1)
        w_ = w_.reshape([w_.shape[0], w_.shape[1]])
        
        if sample_period =='oos':
            f_ = (w_*bond_ret_oos).sum(axis=1).reshape(-1,1)
            f_ = f_*self.sign
            F_ = torch.cat([g_bench_oos, f_],axis=1) 
            loss = torch.exp(- F_.mean(axis=0)@torch.inverse(F_.T.cov())@F_.mean(axis=0))
            return loss, f_, w_, x.reshape(x.shape[0],x.shape[1])
        
        else:
            f_ = (w_*bond_ret_ins).sum(axis=1).reshape(-1,1)
            
            self.sign = torch.sign(f_.mean())
            f_ = f_*self.sign
            
            F_ = torch.cat([g_bench_ins, f_],axis=1)  

            self.w_allocate = torch.inverse(F_.T.cov())@F_.mean(axis=0)	
            abs_sum = torch.sum(torch.abs(self.w_allocate))	
            self.w_allocate = self.w_allocate/abs_sum	

            if print_state == True:	
                print('self.w_allocate:',self.w_allocate)	

            loss = torch.exp(- F_.mean(axis=0)@torch.inverse(F_.T.cov())@F_.mean(axis=0))
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


def run_train_valid(data_input,
                    tensor_input_train, tensor_input_valid,
                    learning_rate, 
                    state=False, patience=20, epochs=400, min_delta=0.01):

    net = Net_SR(data_input)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.0)

    best_loss = float('inf')
    best_model_state = None
    no_improvement_count = 0

    list_loss_train, list_loss_valid = [], []
    for epoch in range(1, epochs+1):

        ##### Training #####
        net.train()

        optimizer.zero_grad()

        loss_train, _, _, _ = net(tensor_input_train, print_state = state)
        
        list_loss_train.append(loss_train.item())
        loss_train.backward()
        optimizer.step()

        ##### Validation #####
        net.eval()

        with torch.no_grad():
            loss_valid, _, _, _ = net(tensor_input_valid, sample_period="oos", print_state = state)
            state = False

            list_loss_valid.append(loss_valid.item())

        if epoch % 50 == 0:
            print()
            print(f'Epoch {epoch} - Training Loss: {loss_train.item()} - Validation Loss: {loss_valid.item()}')
            state = True

        if loss_valid.item() < best_loss + min_delta:
            if loss_valid.item() <= best_loss:
                best_loss = loss_valid.item()
                best_model_state = copy.deepcopy(net.state_dict()) 
            else:
                continue
        else:
            no_improvement_count += 1

        # Check if the model has not improved for the specified number of epochs
        if no_improvement_count >= patience:
            print(f'Early stopping after epoch {epoch}')
            break
    
    print(f'Epoch {epoch}/{epochs} - Training Loss: {loss_train.item()} - Validation Loss: {loss_valid.item()}')

    net.load_state_dict(best_model_state)

    net.eval()
    with torch.no_grad():
        _, F_train, _, _ = net(tensor_input_train, print_state=False)
        _, F_valid, _, _ = net(tensor_input_valid, sample_period="oos", print_state=False)
    return list_loss_train,list_loss_valid,F_train,F_valid,best_loss


def run_ins_oos(data_input,
                tensor_input_ins, tensor_input_oos,
                learning_rate, 
                state=False, patience=20, epochs=400, min_delta=0.01):

    net = Net_SR(data_input)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.0)

    best_loss = float('inf')
    best_model_state = None
    no_improvement_count = 0

    list_loss_ins = []
    list_grad = []
    for epoch in range(epochs):

        ##### INS #####
        net.train()

        optimizer.zero_grad()

        loss_ins, _, _, _ = net(tensor_input_ins, print_state = state)
        state = False

        list_loss_ins.append(loss_ins.item())
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

        if epoch % 50 == 0:
            print()
            print(f'Epoch {epoch} - Training Loss: {loss_ins.item()}')
            state = True
        
        # Early stopping
        if loss_ins.item() < best_loss + min_delta:
            if loss_ins.item() <= best_loss:
                best_loss = loss_ins.item()
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
        _, F_ins, weight_ins, x_ins = net(tensor_input_ins, print_state = state)
        _, F_oos, weight_oos, x_oos = net(tensor_input_oos, sample_period='oos', print_state = state)
    
    return list_loss_ins,F_ins,F_oos,weight_ins,weight_oos,x_ins,x_oos,list_grad,net


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
    factor_ins = data_input['factor_ins']
    factor_oos = data_input['factor_oos']

    # Create DataLoaders for batching
    train_dataset = TensorDataset(tensor_input_train, bond_ret_ins, factor_ins)
    valid_dataset = TensorDataset(tensor_input_valid, bond_ret_oos, factor_oos)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(1, epochs + 1):

        ##### Training #####
        net.train()
        epoch_loss_train = 0.0

        for batch in train_loader:
            batch_input_train, batch_bond_ret_train, batch_factor_train = batch  
            optimizer.zero_grad()

            loss_train, _, _, _ = net(batch_input_train, bond_ret_ins=batch_bond_ret_train, g_bench_ins=batch_factor_train, print_state=state)
            
            epoch_loss_train += loss_train.item()
            loss_train.backward()
            optimizer.step()

        list_loss_train.append(epoch_loss_train / len(train_loader))

        ##### Validation #####
        net.eval()
        epoch_loss_valid = 0.0

        with torch.no_grad():
            for batch in valid_loader:
                batch_input_valid, batch_bond_ret_valid, batch_factor_valid = batch
                loss_valid, _, _, _ = net(batch_input_valid, bond_ret_oos=batch_bond_ret_valid, g_bench_oos=batch_factor_valid, sample_period="oos", print_state=state)
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
        _, F_train, _, _ = net(tensor_input_train, bond_ret_ins=bond_ret_ins, g_bench_ins=factor_ins, print_state=False)
        _, F_valid, _, _ = net(tensor_input_valid, bond_ret_oos=bond_ret_oos, g_bench_oos=factor_oos, sample_period="oos", print_state=False)


    print(f'Epoch {epoch}/{epochs} - Training Loss: {epoch_loss_train / len(train_loader)} - Validation Loss: {epoch_loss_valid / len(valid_loader)}')

    return list_loss_train, list_loss_valid, F_train, F_valid, best_loss


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

    bond_ret_ins = data_input['bond_return_ins']
    factor_ins = data_input['factor_ins']

    ins_dataset = TensorDataset(tensor_input_ins, bond_ret_ins, factor_ins)
    ins_loader = DataLoader(ins_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):

        ##### INS #####
        net.train()
        epoch_loss_ins = 0.0

        for batch in ins_loader:
            batch_input_ins, batch_bond_ret_ins, batch_factor_ins = batch
            optimizer.zero_grad()

            loss_ins, _, _, _ = net(batch_input_ins, bond_ret_ins=batch_bond_ret_ins, g_bench_ins=batch_factor_ins, print_state=state)
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
        _, F_ins, weight_ins, x_ins = net(tensor_input_ins, print_state=state)
        _, F_oos, weight_oos, x_oos = net(tensor_input_oos, sample_period='oos', print_state=state)
    
    return list_loss_ins,F_ins,F_oos,weight_ins,weight_oos,x_ins,x_oos,list_grad,net


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
