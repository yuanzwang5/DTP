import os
import pickle
from parameters import *


insert_line_0 = 12

counter = 0
for filename in ['main_retrain_test_3folds_with_batch.py']:
    for bm_type in benchmark_list:
        for i in range(1, max_layers+1):
            for char_type in chars_use_list:
                for port_type in port_type_list:
                    for oos_target in oos_target_list:
                            
                        counter += 1

                        line_count = 1
                        f0 = open(filename, 'r')
                        f = open('script_retrain_test_3folds_with_batch_' + str(counter) + '.py', 'w')
                        while line_count < insert_line_0:
                            line_count = line_count + 1
                            f.write(f0.readline())

                        f.write('i = {}'.format(i) + '\n')
                        f.write(f'deep_factors_num = {deep_factors_num} \n')
                        f.write('bm_type = "{}"'.format(bm_type)+ '\n')
                        f.write('char_type = "{}"'.format(char_type)+ '\n')
                        f.write('port_type = "{}"'.format(port_type)+ '\n')
                        f.write('oos_target = "{}"'.format(oos_target)+ '\n')
                        f.write(f'time_end = "{time_end}"\n')
                        f.write(f'time_split = "{time_split}"\n')
                        f.write(f'time_start = "{time_start}"\n')
                        f.write(f'T_INS = {T_INS}\n')
                        f.write(f'T_OOS = {T_OOS}\n')
                        f.write(f'seed = {seed}\n')
                        f.write(f'epochs = {epochs}\n')
                        f.write(f'batch_size = {batches}\n')

                        f.write(f0.read())
                        f0.close()
                        f.close()
                        
print(counter)       

f = open('submit_retrain_test_3folds_with_batch.sh', 'w')
f.write('start_time=$(date +"%Y-%m-%d %H:%M:%S") \n')

pre_ = 'seq 1 {} | '.format(counter)
f.write(pre_+'parallel python3 script_retrain_test_3folds_with_batch_{}.py  > log_retrain_test_3folds_with_batch.txt 2>&1\n')

for i in range(2,counter+1):
    f.write("rm script_retrain_test_3folds_with_batch_{}.py \n".format(i))

f.write('end_time=$(date +"%Y-%m-%d %H:%M:%S") \n')
f.write('start_seconds=$(date -d "$start_time" +%s) \n')
f.write('end_seconds=$(date -d "$end_time" +%s) \n')
f.write('elapsed_seconds=$((end_seconds - start_seconds)) \n')
f.write('elapsed_hours=$((elapsed_seconds / 3600)) \n')
f.write('elapsed_minutes=$(( (elapsed_seconds % 3600) / 60 )) \n')
f.write('echo "############################  This is 3folds OOS Test #################################" >> log_time.txt \n')
f.write('echo "Start Time: $start_time" >> log_time.txt \n')
f.write('echo "End Time: $end_time" >> log_time.txt \n')
f.write('echo "Elapsed Time: $elapsed_hours hours $elapsed_minutes minutes" >> log_time.txt \n')
f.close()
