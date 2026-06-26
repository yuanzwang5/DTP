import os
import pickle
import numpy as np
from parameters import *

insert_line_0 = 13

counter = 0
for filename in ['main_retrain_test_with_batch.py']:
    for bm_type in benchmark_list:
        for i in range(1, max_layers+1):
            for char_type in chars_use_list:
                for port_type in port_type_list:
                    for gamma in list_gamma:

                        counter += 1

                        line_count = 1
                        f0 = open(filename, 'r')
                        f = open('script_retrain_test_with_batch_' + str(counter) + '.py', 'w')
                        while line_count < insert_line_0:
                            line_count = line_count + 1
                            f.write(f0.readline())

                        # Write in every parameter
                        f.write('i = {}'.format(i) + '\n')
                        f.write(f'deep_factors_num = {deep_factors_num} \n')
                        f.write('bm_type = "{}"'.format(bm_type)+ '\n')
                        f.write('char_type = "{}"'.format(char_type)+ '\n')
                        f.write('port_type = "{}"'.format(port_type)+ '\n')
                        f.write(f'time_end = "{time_end}"\n')
                        f.write(f'time_split = "{time_split}"\n')
                        f.write(f'time_start = "{time_start}"\n')
                        f.write(f'T_INS = {T_INS}\n')
                        f.write(f'T_OOS = {T_OOS}\n')
                        f.write(f'seed = {seed}\n')
                        f.write(f'epochs = {epochs}\n')
                        f.write(f'trans_cost = {trans_cost}\n')
                        f.write(f'gamma = {gamma}\n')

                        if (gamma == 0.5) or (gamma == 0.7):
                            f.write(f'batch_size = 15\n')
                        else:
                            f.write(f'batch_size = 30\n')

                        f.write(f0.read())
                        f0.close()
                        f.close()
                        
print(counter)       
        
f = open('submit_retrain_test_with_batch.sh', 'w')
pre_ = 'seq 1 {} | '.format(counter)
f.write("#!/bin/bash \n")
f.write(pre_+'parallel python3 script_retrain_test_with_batch_{}.py > log_retrain_test_with_batch.txt 2>&1\n')

for i in range(2, counter+1):
    f.write("rm -rf script_retrain_test_with_batch_{}.py \n".format(i))
f.close()

