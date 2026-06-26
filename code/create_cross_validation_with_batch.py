import os
import pickle
from parameters import *

insert_line_0 = 12

# create foldername according parameter setting
counter = 0
for filename in ['main_validation_with_batch.py']:
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
            
        for i in range(1, max_layers+1):
            for bm_type in benchmark_list:
                for port_type in port_type_list:
                    for cross_validate in cross_validation_list:
                        for learning_rate in list_learning:

                            counter += 1

                            line_count = 1
                            f0 = open(filename, 'r')
                            f = open('script_validation_with_batch_' + str(counter) + '.py', 'w')
                            while line_count < insert_line_0:
                                line_count = line_count + 1
                                f.write(f0.readline())

                            # Write in every parameter
                            f.write('i = {}'.format(i) + '\n')
                            f.write(f'deep_factors_num = {deep_factors_num} \n')
                            f.write('bm_type = "{}"'.format(bm_type)+ '\n')
                            f.write('char_type = "{}"'.format(char_type)+ '\n')
                            f.write('port_type = "{}"'.format(port_type)+ '\n')
                            f.write('cross_validate = "{}"'.format(cross_validate)+ '\n')
                            f.write('learning_rate = {}'.format(learning_rate)+'\n')
                            f.write('list_dropout = {}'.format(list_dropout)+'\n')
                            f.write('list_a1_a2 = {}'.format(list_a1_a2)+'\n')
                            f.write(f'time_end = "{time_end}"\n')
                            f.write(f'time_split = "{time_split}"\n')
                            f.write(f'time_start = "{time_start}"\n')
                            f.write(f'T_INS = {T_INS}\n')
                            f.write(f'T_OOS = {T_OOS}\n')
                            f.write(f'T_VALID = {T_VALID}\n')
                            f.write(f'seed = {seed}\n')
                            f.write(f'epochs = {epochs}\n')
                            f.write(f'batch_size = {batches}\n')

                            f.write(f0.read())
                            f0.close()
                            f.close()
print(counter)       

        
f = open('submit_validation_with_batch.sh', 'w')
f.write("#!/bin/bash \n")
pre_ = f'seq 1 {counter} | parallel -j{counter} '
f.write(pre_+'python3 script_validation_with_batch_{}.py > log_validation_with_batch.txt 2>&1\n')

for i in range(2, counter+1):
    f.write("rm -rf script_validation_with_batch_{}.py \n".format(i))
f.close()
f.close()
