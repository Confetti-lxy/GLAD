PARAM=baseline
# PARAM=baseline_large

CUDA_VISIBLE_DEVICES=7 python tracking/profile_model.py \
                --script GLAD --config $PARAM --precision fp16 --display_name GLAD