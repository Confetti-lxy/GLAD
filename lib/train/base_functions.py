import torch
from torch.utils.data.distributed import DistributedSampler
# datasets related
from lib.train.dataset import Lasot, Got10k, MSCOCOSeq, ImagenetVID, TrackingNet, VastTrack
from lib.train.dataset import Lasot_lmdb, Got10k_lmdb, MSCOCOSeq_lmdb, ImagenetVID_lmdb, TrackingNet_lmdb
from lib.train.dataset.otb99_lang import Otb99_lang
from lib.train.dataset.lasot_ext import Lasot_ext
from lib.train.dataset.lasot_test import Lasot_test
from lib.train.dataset.lasot_new import Lasot_new
from lib.train.dataset.tnl2k import Tnl2k
from lib.train.dataset.refcoco_seq import RefCOCOSeq
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
import lib.train.data.transforms as tfm
from lib.utils.misc import is_main_process
from lib.utils.WarmupMultiStepLR import WarmupMultiStepLR


from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR,
                                   'search': cfg.DATA.SEARCH.FACTOR}
    settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE,
                          'search': cfg.DATA.SEARCH.SIZE}
    settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     'search': cfg.DATA.SEARCH.CENTER_JITTER}
    settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    'search': cfg.DATA.SEARCH.SCALE_JITTER}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE


def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in ["LASOT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full", 
                        "COCO17", "VID", "TRACKINGNET", "GOT10K_official_val", "OTB99", "LASOT_ext", 
                        "TNL2K", "TNL2K_test", "LASOT_test", "OTB99_test", "RefCOCOg", "VastTrack"]
        if name == "LASOT":
            if settings.use_lmdb:
                print("Building lasot dataset from lmdb")
                datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader))
            else:
                # datasets.append(Lasot(settings.env.lasot_dir, split='train', image_loader=image_loader))
                datasets.append(Lasot_new(settings.env.lasot_dir, split='train', image_loader=image_loader))
        if name == "GOT10K_vottrain":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='vottrain', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='vottrain', image_loader=image_loader))
        if name == "GOT10K_train_full":
            if settings.use_lmdb:
                print("Building got10k_train_full from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='train_full', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='train_full', image_loader=image_loader))
        if name == "GOT10K_votval":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='votval', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='votval', image_loader=image_loader))
        if name == "GOT10K_official_val":
            if settings.use_lmdb:
                raise ValueError("Not implement")
            else:
                datasets.append(Got10k(settings.env.got10k_dir_val, split=None, image_loader=image_loader))
        if name == "COCO17":
            if settings.use_lmdb:
                print("Building COCO2017 from lmdb")
                datasets.append(MSCOCOSeq_lmdb(settings.env.coco_lmdb_dir, version="2017", image_loader=image_loader))
            else:
                datasets.append(MSCOCOSeq(settings.env.coco_dir, version="2017", image_loader=image_loader))
        if name == "VID":
            if settings.use_lmdb:
                print("Building VID from lmdb")
                datasets.append(ImagenetVID_lmdb(settings.env.imagenet_lmdb_dir, image_loader=image_loader))
            else:
                datasets.append(ImagenetVID(settings.env.imagenet_dir, image_loader=image_loader))
        if name == "TRACKINGNET":
            if settings.use_lmdb:
                print("Building TrackingNet from lmdb")
                datasets.append(TrackingNet_lmdb(settings.env.trackingnet_lmdb_dir, image_loader=image_loader))
            else:
                datasets.append(TrackingNet(settings.env.trackingnet_dir, image_loader=image_loader))
        if name == "OTB99":
            if settings.use_lmdb:
                raise ValueError("Not implement")
            else:
                datasets.append(Otb99_lang(settings.env.otb99_dir, split='train', image_loader=image_loader))
        if name == "LASOT_ext":
            if settings.use_lmdb:
                raise ValueError("Not implement")
            else:
                datasets.append(Lasot_ext(settings.env.lasotext_dir, split='val', image_loader=image_loader))
        if name == "TNL2K":
            if settings.use_lmdb:
                raise ValueError("Not implement")
            else:
                datasets.append(Tnl2k(settings.env.tnl2k_dir, split=None, image_loader=image_loader))
        if name == "TNL2K_test":
            if settings.use_lmdb:
                raise ValueError("Not implement")
            else:
                datasets.append(Tnl2k("/data2/caiyidong/dataset/TNL2K_test", split=None, image_loader=image_loader))
        if name == "LASOT_test":
            if settings.use_lmdb:
                raise ValueError("Not implement")
            else:
                datasets.append(Lasot_test(settings.env.lasot_dir, split='val', image_loader=image_loader))
        if name == "OTB99_test":
            if settings.use_lmdb:
                raise ValueError("Not implement")
            else:
                datasets.append(Otb99_lang(settings.env.otb99_dir, split='test', image_loader=image_loader))
        if name == "RefCOCOg":
            if settings.use_lmdb:
                raise ValueError("Not implement")
            else:
                datasets.append(RefCOCOSeq(settings.env.refcoco_dir, split="train", image_loader=image_loader, name="refcocog", splitBy="google"))
        if name == "VastTrack":
            datasets.append(VastTrack(settings.env.vasttrack_dir, split='train', image_loader=image_loader))
    return datasets


def build_dataloaders(cfg, settings):
    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))
    # transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2))
    transform_train_template = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                             tfm.RandomHorizontalFlip_Norm(probability=0.5))
    transform_train_search = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                           tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                           tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))
    transform_val_template = tfm.Transform(tfm.ToTensor())
    transform_val_search = tfm.Transform(tfm.ToTensor(),
                                         tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    # Train sampler and loader
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_score = getattr(cfg.TRAIN, "TRAIN_SCORE", False)
    if is_main_process():
        print("sampler_mode", sampler_mode)

    # data_processing_train = processing.MixformerProcessing(search_area_factor=search_area_factor,
    #                                                        output_sz=output_sz,
    #                                                        center_jitter_factor=settings.center_jitter_factor,
    #                                                        scale_jitter_factor=settings.scale_jitter_factor,
    #                                                        mode='sequence',
    #                                                        transform=transform_train,
    #                                                        joint_transform=transform_joint,
    #                                                        settings=settings,
    #                                                        train_score=train_score)
    # 
    # data_processing_val = processing.MixformerProcessing(search_area_factor=search_area_factor,
    #                                                      output_sz=output_sz,
    #                                                      center_jitter_factor=settings.center_jitter_factor,
    #                                                      scale_jitter_factor=settings.scale_jitter_factor,
    #                                                      mode='sequence',
    #                                                      transform=transform_val,
    #                                                      joint_transform=transform_joint,
    #                                                      settings=settings,
    #                                                      train_score=train_score)

    data_processing_train = processing.DiffusionProcessing(search_area_factor=search_area_factor,
                                                           output_sz=output_sz,
                                                           center_jitter_factor=settings.center_jitter_factor,
                                                           scale_jitter_factor=settings.scale_jitter_factor,
                                                           mode='sequence',
                                                           transform=transform_train,
                                                        #    template_transform=transform_train_template,
                                                        #    search_transform=transform_train_search,
                                                           joint_transform=transform_joint,
                                                           settings=settings,
                                                           train_score=train_score)
    
    data_processing_val = processing.DiffusionProcessing(search_area_factor=search_area_factor,
                                                         output_sz=output_sz,
                                                         center_jitter_factor=settings.center_jitter_factor,
                                                         scale_jitter_factor=settings.scale_jitter_factor,
                                                         mode='sequence',
                                                         transform=transform_val,
                                                        #  template_transform=transform_val_template,
                                                        #  search_transform=transform_val_search,
                                                         joint_transform=transform_joint,
                                                         settings=settings,
                                                         train_score=train_score)

    if is_main_process():
        print("TrainLoader Preparing")
    dataset_train = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
                                            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_train,
                                            frame_sample_mode=sampler_mode, train_cls=train_score, pos_prob=0.5)

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)
    if is_main_process():
        print("TrainLoader Ready")

    # Validation samplers and loaders
    if is_main_process():
        print("ValidationLoader Preparing")
    dataset_val = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, opencv_loader),
                                          p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
                                          samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
                                          max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                          num_template_frames=settings.num_template, processing=data_processing_val,
                                          frame_sample_mode=sampler_mode, train_cls=train_score, pos_prob=0.5)
    val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                           num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
                           epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)
    # GOT10k_official_val
    dataset_val_got = sampler.TrackingSampler(datasets=names2datasets(["GOT10K_official_val"], settings, opencv_loader),
                                          p_datasets=[1],
                                          samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
                                          max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                          num_template_frames=settings.num_template, processing=data_processing_val,
                                          frame_sample_mode=sampler_mode, train_cls=train_score, pos_prob=0.5)
    val_sampler_got = DistributedSampler(dataset_val_got) if settings.local_rank != -1 else None
    loader_val_got = LTRLoader('got_val', dataset_val_got, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                           num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler_got,
                           epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)
    if is_main_process():
        print("ValidationLoader Ready")

    return loader_train, loader_val, loader_val_got


def get_optimizer_scheduler(net, cfg):
    same_lr = getattr(cfg.TRAIN, "SAME_LR", True)
    if same_lr:
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if p.requires_grad]},
        ]
    else:
        # param_dicts = [
        #     {
        #         "params": [p for n, p in net.named_parameters() if "head" in n and p.requires_grad],
        #         "lr": cfg.TRAIN.LR,
        #         'weight_decay': cfg.TRAIN.WEIGHT_DECAY,
        #         'betas': (0.9, 0.999)
        #     },
        #     {
        #         "params": [p for n, p in net.named_parameters() if ("fd" in n or "norm_output" in n or "text_cat_proj" in n)  and p.requires_grad],
        #         "lr": cfg.TRAIN.LR * cfg.TRAIN.DECODER_MULTIPLIER,
        #         'weight_decay': cfg.TRAIN.WEIGHT_DECAY,
        #         'betas': (0.9, 0.999),
        #     },
        #     {
        #         "params": [p for n, p in net.named_parameters() if "pm" in n and p.requires_grad],
        #         "lr": cfg.TRAIN.LR * cfg.TRAIN.POOL_MULTIPLIER,
        #         'weight_decay': cfg.TRAIN.WEIGHT_DECAY,
        #         'betas': (0.9, 0.999),
        #     },
        #     # {
        #     #     "params": [p for n, p in net.named_parameters() if "image_model" not in n and p.requires_grad],
        #     #     "lr": cfg.TRAIN.LR,
        #     #     'weight_decay': cfg.TRAIN.WEIGHT_DECAY,
        #     #     'betas': (0.9, 0.999)
        #     # },
        #     {
        #         "params": [p for n, p in net.named_parameters() if "image_model" in n and p.requires_grad],
        #         "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
        #         'weight_decay': cfg.TRAIN.WEIGHT_DECAY,
        #         'betas': (0.9, 0.999)
        #     },
        # ]
        param_dicts = [
            # {
            #     "params": [p for n, p in net.named_parameters() if "head" in n and p.requires_grad],
            #     "lr": cfg.TRAIN.LR * 0.5,
            #     'weight_decay': cfg.TRAIN.WEIGHT_DECAY,
            #     'betas': (0.9, 0.999)
            # },
            # {
            #     "params": [p for n, p in net.named_parameters() if "encoder" not in n and "head" not in n and p.requires_grad],
            #     "lr": cfg.TRAIN.LR,
            #     'weight_decay': cfg.TRAIN.WEIGHT_DECAY,
            #     'betas': (0.9, 0.999)
            # },
            {
                "params": [p for n, p in net.named_parameters() if "encoder" not in n and p.requires_grad],
                "lr": cfg.TRAIN.LR,
                'weight_decay': cfg.TRAIN.WEIGHT_DECAY,
                'betas': (0.9, 0.999)
            },
            {
                "params": [p for n, p in net.named_parameters() if "encoder" in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
                'weight_decay': cfg.TRAIN.WEIGHT_DECAY,
                'betas': (0.9, 0.999)
            },
        ]

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR, betas=(0.9, 0.999), 
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIMIZER == "FusedAdam":
        optimizer = FusedAdam(
            param_dicts, adam_w_mode=True,
            lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY, 
            betas=(0.9, 0.999), eps=1e-8)
    else:
        raise ValueError("Unsupported Optimizer")
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH, gamma=cfg.TRAIN.SCHEDULER.DECAY_RATE)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                            gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    elif cfg.TRAIN.SCHEDULER.TYPE == "WarmMstep":
        lr_scheduler = WarmupMultiStepLR(optimizer, milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                         gamma=cfg.TRAIN.SCHEDULER.GAMMA, warmup_iters=cfg.TRAIN.SCHEDULER.WARM_EPOCH)
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler
