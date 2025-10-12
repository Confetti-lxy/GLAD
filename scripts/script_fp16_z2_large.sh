gpu_vis=4,5,6,7
gpu_num=4
MASTER_PORT=2345

python tracking/train.py --script GLAD --config baseline_large --save_dir ds_train_fp16_z2_large \
                         --mode deepspeed --nproc_per_node $gpu_num --gpu_vis $gpu_vis \
                         --master_port $MASTER_PORT --config_file experiments/deepspeed/ds_fp16_z2_config_large.json \
                         --precision fp16