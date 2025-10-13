from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/data3/luoxingyu/data/dataset/got10k_lmdb'
    settings.got10k_path = '/data3/luoxingyu/data/dataset/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = '/data3/luoxingyu/data/dataset/lasot_ext'
    settings.lasot_lmdb_path = '/data3/luoxingyu/data/dataset/lasot_lmdb'
    settings.lasot_path = '/data3/luoxingyu/data/dataset/lasot'
    settings.network_path = '/data3/luoxingyu/GLAD/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/data3/luoxingyu/data/dataset/nfs'
    settings.otb_path = '/data3/luoxingyu/data/dataset/OTB2015'
    settings.prj_dir = '/data3/luoxingyu/GLAD'
    settings.result_plot_path = '/data3/luoxingyu/GLAD/test_base/result_plots'
    settings.results_path = '/data3/luoxingyu/GLAD/test_base/tracking_results'    # Where to store tracking results
    settings.save_dir = '/data3/luoxingyu/GLAD'
    settings.segmentation_path = '/data3/luoxingyu/GLAD/test/segmentation_results'
    settings.tc128_path = '/data3/luoxingyu/data/dataset/TC128'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/data3/luoxingyu/data/dataset/trackingnet'
    settings.uav_path = '/data3/luoxingyu/data/dataset/UAV123'
    settings.tnl2k_path = '/data3/luoxingyu/data/dataset/tnl2k/test'
    settings.otb99_path = '/data3/luoxingyu/data/dataset/OTB_sentences'
    settings.vot20_path = '/data3/luoxingyu/data/dataset/vot2020'
    settings.vot_path = '/data3/luoxingyu/data/dataset/VOT2019'
    settings.youtubevos_dir = ''
    settings.videocube_path = '/data3/luoxingyu/data/dataset/MGIT'

    return settings

