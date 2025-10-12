# CUDA_VISIBLE_DEVICES=4,5,6,7 python tracking/test.py GLAD baseline_large --dataset otb99 --threads 32 --num_gpus 4 --params__model GLAD_ep0300.pth.tar --params__search_area_scale 4.0

# CUDA_VISIBLE_DEVICES=4,5,6,7 python tracking/test.py GLAD baseline_large --dataset got10k_test --threads 32 --num_gpus 4 --params__model GLAD_ep0300.pth.tar --params__search_area_scale 4.0

# CUDA_VISIBLE_DEVICES=4,5,6,7 python tracking/test.py GLAD baseline_large --dataset got10k_val --threads 32 --num_gpus 4 --params__model GLAD_ep0300.pth.tar --params__search_area_scale 4.0

# CUDA_VISIBLE_DEVICES=4,5,6,7 python tracking/test.py GLAD baseline_large --dataset lasot_ext --threads 32 --num_gpus 4 --params__model GLAD_ep0300.pth.tar --params__search_area_scale 4.0

# CUDA_VISIBLE_DEVICES=4,5,6,7 python tracking/test.py GLAD baseline_large --dataset lasot --threads 32 --num_gpus 4 --params__model GLAD_ep0300.pth.tar --params__search_area_scale 4.0

# CUDA_VISIBLE_DEVICES=4,5,6,7 python tracking/test.py GLAD baseline_large --dataset tnl2k --threads 32 --num_gpus 4 --params__model GLAD_ep0300.pth.tar --params__search_area_scale 4.0