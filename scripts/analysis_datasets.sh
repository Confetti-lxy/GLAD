MODE=all

CUDA_VISIBLE_DEVICES=2,3 python tracking/analysis_datasets.py CLIP --dataset otb99 --analysis_mode $MODE --threads 16 --num_gpus 2

CUDA_VISIBLE_DEVICES=2,3 python tracking/analysis_datasets.py CLIP --dataset tnl2k --analysis_mode $MODE --threads 16 --num_gpus 2

CUDA_VISIBLE_DEVICES=2,3 python tracking/analysis_datasets.py CLIP --dataset lasot --analysis_mode $MODE --threads 16 --num_gpus 2

CUDA_VISIBLE_DEVICES=2,3 python tracking/analysis_datasets.py CLIP --dataset lasot_ext --analysis_mode $MODE --threads 16 --num_gpus 2

CUDA_VISIBLE_DEVICES=2,3 python tracking/analysis_datasets.py CLIP --dataset got10k_val --analysis_mode $MODE --threads 16 --num_gpus 2