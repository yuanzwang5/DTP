##################################################
################## Main Results ##################
##################################################

start_time=$(date +%s)

cd ./code
sh submit_all.sh > log_submit_all.txt 2>&1
cd ..

end_time=$(date +%s)
echo "Main training runtime: $(( (end_time - start_time) / 60 )) minutes"


###################################################
########## Start to do Robustness Checks ##########
###################################################


##### Robustness Check 1: Randomly Split Sample #####

start_time=$(date +%s)

cd ./robust_check/sample_randomly_split
sh submit_all_3folds.sh > log_submit_all_3folds.txt 2>&1
cd ../..

end_time=$(date +%s)
echo "Robustness Check 1 runtime: $(( (end_time - start_time) / 60 )) minutes"


##### Robustness Check 2: Linear Ranking #####

start_time=$(date +%s)

cd ./robust_check/linear_ranking
sh submit_all.sh > log_submit_all.txt 2>&1
cd ../..

end_time=$(date +%s)
echo "Robustness Check 2 runtime: $(( (end_time - start_time) / 60 )) minutes"


##### Robustness Check 3: Reversed Sample #####

start_time=$(date +%s)

cd ./robust_check/reversed_sample
sh submit_all.sh > log_submit_all.txt 2>&1
cd ../..

end_time=$(date +%s)
echo "Robustness Check 3 runtime: $(( (end_time - start_time) / 60 )) minutes"


##### Robustness Check 4: Transaction Cost #####

start_time=$(date +%s)

cd ./robust_check/trans_cost/15bps
sh submit_all.sh > log_submit_all.txt 2>&1
cd ../../..

cd ./robust_check/trans_cost/20bps
sh submit_all.sh > log_submit_all.txt 2>&1
cd ../../..

cd ./robust_check/trans_cost
python3 run_other_benchmark_enhancement.py > log_run_other_benchmark_enhancement.txt 2>&1
cd ../..

end_time=$(date +%s)
echo "Robustness Check 4 Transaction Cost runtime: $(( (end_time - start_time) / 60 )) minutes"


##### Robustness Check 5: Machine Learning Alternative and Linear Activation #####

start_time=$(date +%s)

cd ./robust_check/ML_long_short_linear_activation/linear_activation
sh submit_all_linear_activation.sh > log_submit_linear_activation.txt 2>&1
cd ../../..

cd ./robust_check/ML_long_short_linear_activation/predict_return
sh submit_all_predict_return.sh > log_submit_all_predict_return.txt 2>&1
cd ../../..

cd ./robust_check/ML_long_short_linear_activation
python3 final_summary_with_other_ML.py > log_final_summary_with_other_ML.txt 2>&1
python3 final_summary_with_other_ML_all_obs.py > log_final_summary_with_other_ML_all_obs.txt 2>&1
cd ../..

cd ./robust_check/ML_long_short_linear_activation/max_SR
sh submit_all_max_SR.sh > log_submit_all_max_SR.txt 2>&1
cd ../../..

cd ./robust_check/ML_long_short_linear_activation
python3 final_summary_with_other_ML_all_obs_max_SR_objective.py > log_final_summary_with_other_ML_all_obs_max_SR_objective.txt 2>&1
cd ../..

end_time=$(date +%s)
echo "Robustness Check 5 runtime: $(( (end_time - start_time) / 60 )) minutes"



##### Calculate Performance in Main Text #####

start_time=$(date +%s)

cd ./code
python3 final_results_with_batch.py > log_final_results_with_batch.txt 2>&1
cd ..

end_time=$(date +%s)
echo "All main results runtime: $(( (end_time - start_time) / 60 )) minutes"



########## The End ##########


