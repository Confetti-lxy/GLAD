import os
# loss function related
from lib.utils.box_ops import giou_loss, ciou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import DeepSpeedTrainer
# distributed training related
import deepspeed
# some more advanced functions
from .base_functions import *
# network related
from lib.models.GLAD import build_pipeline
# forward propagation related
from lib.train.actors import GLADActor
# for import modules
import importlib

from ..utils.focal_loss import FocalLoss


def run(settings, args):
    settings.description = 'Training script for deepspeed'

    # Update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # Update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_train, loader_val, loader_val_got = build_dataloaders(cfg, settings)

    # Create network
    if settings.script_name == "GLAD":
        net = build_pipeline(cfg, args.precision)
        if is_main_process():
            print("building diffusion pipeline for tracking")
    else:
        raise ValueError("illegal script name")

    # Move net(image_model, pooling_modules, fusion_decoders, head) to cuda, together with text_model and sd_model
    net.to("cuda")
    if settings.script_name == 'GLAD':
        # net.text_model[0].to("cuda")
        # net.text_model[2].to("cuda")
        net.sd_model.to("cuda")

    # Optimizer and LearnRateScheduler
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)

    for p in net.parameters():
        if not p.data.is_contiguous():
            p.data = p.data.contiguous()
    
    # Initialize DeepSpeed for the model
    net, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=net,
        optimizer=optimizer,
        model_parameters=net.parameters(),
    )

    # Initialize device for each gpu
    if settings.local_rank != -1:
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    
    # settings.save_every_epoch = True  # set to save every epoch, default is False

    # Loss functions and Actors
    if settings.script_name == 'GLAD':
        focal_loss = FocalLoss()
        objective = {'ciou': ciou_loss, 'l1': l1_loss, 'focal': focal_loss}
        loss_weight = {'ciou': cfg.TRAIN.IOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': cfg.TRAIN.FOCAL_WEIGHT}
        actor = GLADActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
    else:
        raise ValueError("illegal script name")
    
    use_deepspeed = getattr(cfg.TRAIN, "DEEPSPEED", True)
    trainer = DeepSpeedTrainer(actor, [loader_train, loader_val, loader_val_got], optimizer, settings, lr_scheduler, use_deepspeed=use_deepspeed)

    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
