# CUDA_VISIBLE_DEVICES=4,5,6,7 python tracking/test.py GLAD baseline --dataset got10k_val --threads 32 --num_gpus 4 --params__model GLAD_ep0240.pth.tar --params__search_area_scale 4.0

# CUDA_VISIBLE_DEVICES=4,5 python tracking/test.py GLAD baseline --dataset lasot --threads 16 --num_gpus 2 --params__model GLAD_ep0240.pth.tar --params__search_area_scale 4.0

# CUDA_VISIBLE_DEVICES=4,5,6,7 python tracking/test.py GLAD baseline --dataset lasot_ext --threads 8 --num_gpus 4 --params__model GLAD_ep0240.pth.tar --params__search_area_scale 4.0

# CUDA_VISIBLE_DEVICES=6,7 python tracking/test.py GLAD baseline --dataset tnl2k --threads 16 --num_gpus 2 --params__model GLAD_ep0240.pth.tar --params__search_area_scale 4.0

# CUDA_VISIBLE_DEVICES=4,5,6,7 python tracking/test.py GLAD baseline --dataset otb99 --threads 32 --num_gpus 4 --params__model GLAD_ep0240.pth.tar --params__search_area_scale 4.0

# CUDA_VISIBLE_DEVICES=4,5,6,7 python tracking/test.py GLAD baseline --dataset got10k_test --threads 32 --num_gpus 4 --params__model GLAD_ep0240.pth.tar --params__search_area_scale 4.0