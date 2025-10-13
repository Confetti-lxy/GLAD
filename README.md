# GLAD
A Baseline Implementation to Employ Diffusion Feature for Vision-Language Tracking


# GLAD: Generative Language-Assisted Visual Tracking for Low-Semantic Templates

> [Xingyu Luo](https://scholar.google.com.hk/citations?user=NqXtIPIAAAAJ),  [Yidong Cai](https://huuuuusy.github.io/), [Jie Liu](https://github.com/Xuchen-Li), [Jie Tang](https://scholar.google.com.hk/citations?user=ApH4wOcAAAAJ), [Gangshan Wu](https://scholar.google.com.hk/citations?user=fGc7NVAAAAAJ), [Limin Wang](https://github.com/XiaokunFeng/CSTrack)


[![](https://img.shields.io/badge/GLAD-arXiv%20-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2507.19875)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20GLAD-Results&ckpts-red)](https://huggingface.co/Xiaokunfeng2022/GLAD)

This is an official pytorch implementation of the paper **GLAD: Generative Language-Assisted Visual Tracking
for Low-Semantic Templates**.


<!-- ### 🔥 Updates

*   \[8/2025\] **GLAD's** code is available!
*   \[6/2025\] **GLAD**  is accepted by ICCV25 Highlight! -->

### 📣 Overview
#### Our motivation & Core modeling approach

Vision-language tracking aims to locate the target object in the video sequence using a template patch and a language description provided in the initial frame. To achieve
robust tracking, especially in complex long-term scenarios that reflect real-world conditions as recently highlighted by
MGIT, it is essential not only to characterize the target features but also to utilize the context features related to the
target. However, the visual and textual target-context cues
derived from the initial prompts generally align only with
the initial target state. Due to their dynamic nature, target states are constantly changing, particularly in complex
long-term sequences. It is intractable for these cues to continuously guide Vision-Language Trackers (VLTs). Furthermore, for the text prompts with diverse expressions, our
experiments reveal that existing VLTs struggle to discern
which words pertain to the target or the context, complicating the utilization of textual cues. 
![GLAD_motivation](asset/motivation.png)

In this work, we present a novel tracker named GLAD, which can obtain multimodal cues Aligned with the dynamic target states
through comprehensive Target-Context feature modeling,
thereby achieving robust tracking. Specifically, (1) for the
visual modality, we propose an effective temporal visual
target-context modeling approach that provides the tracker
with timely visual cues. (2) For the textual modality, we
achieve precise target words identification solely based on
textual content, and design an innovative context words
calibration method to adaptively utilize auxiliary context
words. (3) We conduct extensive experiments on mainstream benchmarks and GLAD achieves a new SOTA
performance
![GLAD_pipeline](asset/framework.png)

#### Strong performance

![GLAD_experiment](asset/experiment.png)



### 🔨 Installation
```
conda create -n glad python=3.8
conda activate glad
pip install -r requirements.txt
```

### 🔧 Usage

#### Data Preparation
Our GLAD is trained on  LaSOT, TNL2K, RefCOCOg, OTB99-Lang, GOT-10k, and TrackingNet datasets.  
Put these tracking datasets in [./data](data). It should look like:

   ```
   ${GLAD_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
        -- tnl2k
            -- train
                |-- Arrow_Video_ZZ04_done
                |-- Assassin_video_1-Done
                |-- Assassin_video_2-Done
                ...
            -- test
                |-- advSamp_Baseball_game_002-Done
                |-- advSamp_Baseball_video_01-Done
                |-- advSamp_Baseball_video_02-Done
                ...

   ```

#### Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

#### Train
##### Prepare pretrained backbone
The backbone and patch embedding of GLAD are initialized with pre-trained weights from [**Fast-iTPN**](https://github.com/sunsmarterjie/iTPN), and we adopt RoBERTa-Base as our text encoder.  
Please download the **fast_itpn_base_clipl_e1600.pt**, **fast_itpn_large_1600e_1k.pt** and **roberta-base** checkpoints and place them in [./resource/pretrained_models](./resource/pretrained_models).

##### Train GLAD
You can run the following command to train the GLAD-B256:
```
python tracking/train.py --script GLAD --config baseline --save_dir $save_dir --mode deepspeed --nproc_per_node $gpu_num --gpu_vis $gpu_vis --master_port $MASTER_PORT --config_file experiments/deepspeed/ds_bf16_z2_config.json --precision bf16
```

Besides, you can run the following command to train the GLAD-L384:
```
python tracking/train.py --script GLAD --config baseline_large --save_dir $save_dir --mode deepspeed --nproc_per_node $gpu_num --gpu_vis $gpu_vis --master_port $MASTER_PORT --config_file experiments/deepspeed/ds_bf16_z2_config_large.json --precision bf16
```

#### Test and evaluate on benchmarks
First, you need to set the paths for the various evaluation benchmarks in [./lib/test/evaluation/local.py](./lib/test/evaluation/local.py), and prepare the model weights for evaluation. 
Then, run the following command to perform evaluation on different benchmarks (taking GLAD_base as an example).
- LaSOT
```
CUDA_VISIBLE_DEVICES=$gpu_vis python tracking/test.py GLAD baseline --dataset lasot --threads 32 --num_gpus $gpu_num --params__model $checkpoint_dir --params__search_area_scale 4.0
python tracking/analysis_results.py --dataset_name lasot --tracker_param baseline
```
- LaSOT_ext
```
CUDA_VISIBLE_DEVICES=$gpu_vis python tracking/test.py GLAD baseline --dataset lasot_ext --threads 32 --num_gpus $gpu_num --params__model $checkpoint_dir --params__search_area_scale 4.0
python tracking/analysis_results.py --dataset_name lasot_ext --tracker_param baseline
```

- TNL2K
```
CUDA_VISIBLE_DEVICES=$gpu_vis python tracking/test.py GLAD baseline --dataset tnl2k --threads 32 --num_gpus $gpu_num --params__model $checkpoint_dir --params__search_area_scale 4.0
python tracking/analysis_results.py --dataset_name tnl2k --tracker_param baseline
```

- OTB99

```
CUDA_VISIBLE_DEVICES=$gpu_vis python tracking/test.py GLAD baseline --dataset otb99 --threads 32 --num_gpus $gpu_num --params__model $checkpoint_dir --params__search_area_scale 4.0
python tracking/analysis_results.py --dataset_name otb99 --tracker_param baseline
```

### 📊 Model Zoo
The trained models, and the raw tracking results are provided in the [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20GLAD-Results&ckpts-red)](https://huggingface.co/Xiaokunfeng2022/GLAD).


<!-- ### ❤️Acknowledgement
We would like to express our gratitude to the following open-source repositories that our work is based on: [SeqtrackV2](https://github.com/chenxin-dlut/SeqTrackv2),  [AQATrack](https://github.com/GXNU-ZhongLab/AQATrack), [Fast-iTPN](https://github.com/sunsmarterjie/iTPN).
Their contributions have been invaluable to this project. -->