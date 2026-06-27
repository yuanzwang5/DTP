
################## Main Results ##################

# start_time=$(date +%s)

# cd ./code
# sh submit_all.sh > log_submit_all.txt 2>&1
# cd ..

# end_time=$(date +%s)
# echo "Main training runtime: $(( (end_time - start_time) / 60 )) minutes"



##### Calculate Performance in Main Text #####

start_time=$(date +%s)

cd ./code
python3 final_results_with_batch.py > log_final_results_with_batch.txt 2>&1
cd ..

end_time=$(date +%s)
echo "All main results runtime: $(( (end_time - start_time) )) seconds"



########## The End ##########


