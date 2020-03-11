for((i=0;i<5;i++));  
do   
sudo docker cp output_base_wwm_new4_$i 57bb6fcd2eb6:/output_base_wwm_$i
done
sudo docker cp data/dev.csv 57bb6fcd2eb6:/tcdata/test.csv
sudo docker cp run_bert.py 57bb6fcd2eb6:/
sudo docker cp fuse.py 57bb6fcd2eb6:/
sudo docker cp run.sh 57bb6fcd2eb6:/
sudo docker cp model.py 57bb6fcd2eb6:/
