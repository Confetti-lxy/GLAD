class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/data3/luoxingyu/DFTrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/data3/luoxingyu/DFTrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/data3/luoxingyu/DFTrack/pretrained_networks'
        self.lasot_dir = '/data3/luoxingyu/data/dataset/lasot'
        self.got10k_dir = '/data3/luoxingyu/data/dataset/got10k/train'
        self.got10k_dir_val = '/data3/luoxingyu/data/dataset/got10k/val'
        self.lasot_lmdb_dir = '/data3/luoxingyu/data/dataset/lasot_lmdb'
        self.got10k_lmdb_dir = '/data3/luoxingyu/data/dataset/got10k_lmdb'
        self.trackingnet_dir = '/data3/luoxingyu/data/dataset/trackingnet'
        self.trackingnet_lmdb_dir = '/data3/luoxingyu/data/dataset/trackingnet_lmdb'
        self.tnl2k_dir = "/data3/luoxingyu/data/dataset/tnl2k/train"
        self.otb99_dir = "/data3/luoxingyu/data/dataset/OTB_sentences"
        self.lasotext_dir = "/data3/luoxingyu/data/dataset/lasot_ext"
        self.coco_dir = '/data3/luoxingyu/data/dataset/coco'
        self.coco_lmdb_dir = '/data3/luoxingyu/data/dataset/coco_lmdb'
        self.refcoco_dir =  '/data3/luoxingyu/data/dataset/refcoco'
        self.vasttrack_dir = '/data3/luoxingyu/data/dataset/VastTrack/unisot_train_final_backup'
        # /home/luoxingyu/data/luoxingyu/data/dataset/VastTrack/unisot_train_final_backup
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/data3/luoxingyu/data/dataset/vid'
        self.imagenet_lmdb_dir = '/data3/luoxingyu/data/dataset/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
