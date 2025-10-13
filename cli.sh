gpu_vis=4,5,6,7
gpu_num=4
MASTER_PORT=24781
config_name=baseline
save_dir=ds_glad_train_bf16_z2
deepspeed_name=experiments/deepspeed/ds_bf16_z2_config.json

# skip_training=true
skip_training=false

# train
if [ "$skip_training" = false ]; then
    python tracking/train.py --script GLAD --config $config_name --save_dir $save_dir \
                            --mode deepspeed --nproc_per_node $gpu_num --gpu_vis $gpu_vis \
                            --master_port $MASTER_PORT --config_file $deepspeed_name \
                            --precision bf16
else
    echo "training is ended"
fi

checkpoint_dir="GLAD_B256.pth.tar"

# test lasot 280
# CUDA_VISIBLE_DEVICES=$gpu_vis python tracking/test.py GLAD $config_name \
#             --dataset lasot --threads 32 --num_gpus $gpu_num \
#             --params__model $checkpoint_dir \
#             --params__search_area_scale 4.0
# python tracking/analysis_results.py --dataset_name lasot --tracker_param $config_name

# test lasot_ext 150
# CUDA_VISIBLE_DEVICES=$gpu_vis python tracking/test.py GLAD $config_name \
#             --dataset lasot_ext --threads 32 --num_gpus $gpu_num \
#             --params__model $checkpoint_dir \
#             --params__search_area_scale 4.0
# python tracking/analysis_results.py --dataset_name lasot_ext --tracker_param $config_name

# # test otb99 48
# CUDA_VISIBLE_DEVICES=$gpu_vis python tracking/test.py GLAD $config_name \
#             --dataset otb99 --threads 32 --num_gpus $gpu_num \
#             --params__model $checkpoint_dir \
#             --params__search_area_scale 4.0
# python tracking/analysis_results.py --dataset_name otb99 --tracker_param $config_name

# test tnl2k 700
# CUDA_VISIBLE_DEVICES=$gpu_vis python tracking/test.py GLAD $config_name \
#             --dataset tnl2k --threads 32 --num_gpus $gpu_num \
#             --params__model $checkpoint_dir \
#             --params__search_area_scale 4.0
# python tracking/analysis_results.py --dataset_name tnl2k --tracker_param $config_name


