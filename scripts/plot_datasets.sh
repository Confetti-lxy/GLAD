MODE=all

python tracking/plot_datasets.py CLIP --dataset_name otb99 --analysis_mode $MODE

python tracking/plot_datasets.py CLIP --dataset_name tnl2k --analysis_mode $MODE

python tracking/plot_datasets.py CLIP --dataset_name lasot --analysis_mode $MODE

python tracking/plot_datasets.py CLIP --dataset_name lasot_ext --analysis_mode $MODE

python tracking/plot_datasets.py CLIP --dataset_name got10k_val --analysis_mode $MODE