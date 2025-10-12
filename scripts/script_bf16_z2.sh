gpu_vis=4,5,6,7
gpu_num=4
MASTER_PORT=3456

python tracking/train.py --script GLAD --config baseline --save_dir ds_train_bf16_z2 \
                         --mode deepspeed --nproc_per_node $gpu_num --gpu_vis $gpu_vis \
                         --master_port $MASTER_PORT --config_file experiments/deepspeed/ds_bf16_z2_config.json \
                         --precision bf16