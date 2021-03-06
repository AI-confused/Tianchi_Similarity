export CUDA_VISIBLE_DEVICES=0
i=0
while [ "$i" -le 4 ];
do

python run_bert.py \
--model_name_or_path ./chinese_wwm_ext_pytorch \
--do_test \
--index $i \
--data_dir ./data/data_$i \
--output_dir ./output_base_wwm_$i \
--max_seq_length 64 \
--eval_steps 200 \
--per_gpu_train_batch_size 32 \
--gradient_accumulation_steps 1 \
--warmup_steps 0 \
--per_gpu_eval_batch_size 256 \
--learning_rate 1e-5 \
--adam_epsilon 1e-6 \
--weight_decay 0 \
--train_steps 1000

i=$(( i + 1 ))
done
python fuse.py