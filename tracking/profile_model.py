import argparse
import torch
import os
import time
import importlib
import _init_paths
from torch import nn
from thop import profile
from thop.utils import clever_format
from lib.utils.torch_utils import is_torch_bf16_available


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='GLAD', choices=['GLAD'],
                        help='training script name')
    parser.add_argument('--config', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--precision', type=str, default='fp16', help='[fp16, fp32, bf16]')
    parser.add_argument('--detailed', action='store_true')
    parser.add_argument('--display_name', type=str, default='GLAD')
    args = parser.parse_args()

    return args


# For sd_model
def evaluate_GLAD_sd_model():
    print("``` sd model:")
    # using results from https://github.com/ThanatosShinji/onnx-tool#results-of-onnx-model-zoo-and-sota-models
    #   text_encoder    - 123.13M  , 6782M
    #   UNet2DCondition - 859.52M  , 888870M
    #   VAE_encoder     - 34.16M   , 566371M
    #   VAE_decoder     - 49.49M   , 1271959M

    # NOTE: VAE_decoder isn't used in our work and only part of UNet2DCondition is used in our LCM-based implementation
    macs = (6782 + 888870 + 566371) / 1000
    params = 123.13 + 859.52 + 34.16
    print('    ==>MACs is {} G'.format(macs))
    print('    ==>Params is {} M'.format(params))
    print("\033[0;32;40m Warning: This result is not accurate and only for reference, please refer to official Stable Diffusion v1.5! \033[0m")


# For image_model
def evaluate_GLAD_image_model(model, template, search):
    print("``` image model:")
    macs, params = profile(model.encoder, inputs=(template, search))
    macs, params = clever_format([macs, params], "%.3f")
    print('    ==>MACs is ', macs)
    print('    ==>Params is ', params)


# For pooling
def get_latent_features(bs, device):
    latent_features = [torch.randn(bs, 1280, 16, 16).to(device),
                       torch.randn(bs, 1280, 16, 16).to(device),
                       torch.randn(bs, 1280, 8, 8).to(device)]
    return latent_features

def evaluate_GLAD_pooling(model, latent_features):
    print("``` pooling module:")
    total_macs, total_params = 0, 0
    for i in range(3):
        macs, params = profile(model.pm[i], inputs=(latent_features[i], None, "attention"))
        total_macs += macs
        total_params += params
        macs, params = clever_format([macs, params], "%.3f")
        print("      pm[{}]: MACs is {}, Params is {}".format(i, macs, params))
    total_macs, total_params = clever_format([total_macs, total_params], "%.3f")
    print('    ==>MACs is ', total_macs)
    print('    ==>Params is ', total_params)


# For decoder
def get_pooled_latent_features(bs, image_dim, device):
    pooled_latent_features = [torch.randn(bs, 256, image_dim).to(device),
                              torch.randn(bs, 256, image_dim).to(device),
                              torch.randn(bs, 64, image_dim).to(device)]
    return pooled_latent_features

def get_image_features(bs, img_size, image_dim, device):
    if img_size == 384:
        patch_num = ((384//16) ** 2) * 2
    elif img_size == 256:
        patch_num = ((256//16) ** 2) * 2
    image_features = torch.randn(bs, patch_num, image_dim).to(device)
    return image_features

def evaluate_GLAD_decoder(model, pooled_latent_features, image_features):
    print("``` decoder module:")
    total_macs, total_params = 0, 0
    for i in range(3):
        macs, params = profile(model.fd[i], inputs=(image_features, pooled_latent_features[i], None))
        total_macs += macs
        total_params += params
        macs, params = clever_format([macs, params], "%.3f")
        print("      fd[{}]: MACs is {}, Params is {}".format(i, macs, params))
    norm_macs, norm_params = profile(model.norm_output, inputs=(image_features))
    total_macs += norm_macs
    total_params += norm_params
    norm_macs, norm_params = clever_format([norm_macs, norm_params], "%.3f")
    print("      norm_output: MACs is {}, Params is {}".format(norm_macs, norm_params))
    total_macs, total_params = clever_format([total_macs, total_params], "%.3f")
    print('    ==>MACs is ', total_macs)
    print('    ==>Params is ', total_params)


# For head
def get_image_features_resized(bs, img_size, image_dim, device):
    if img_size == 384:
        f_sz = 384 // 16
    elif img_size == 256:
        f_sz = 256 // 16
    image_features_resized = torch.randn(bs, image_dim, f_sz, f_sz).to(device)
    return image_features_resized

def evaluate_GLAD_head(model, image_features, head_type):
    """Compute MACs, Params and FPS"""
    print("``` head:")
    if head_type == "CORNER":
        macs, params = profile(model.head, inputs=(image_features))
    else:
        macs, params = profile(model.head, inputs=(image_features, None))
    macs, params = clever_format([macs, params], "%.3f")
    print('    ==>MACs is ', macs)
    print('    ==>Params is ', params)


# For overall
def get_data(bs, sz, device):
    img_patch = torch.randn(bs, 3, sz, sz).to(device)
    return img_patch

def get_text(bs):
    text = ["Something in the scene."]
    return text * bs

def evaluate_GLAD(model, template, search, text, display_info='GLAD'):
    """Compute MACs, Params and FPS"""
    print("Overall Model:")
    macs, params = profile(model, inputs=(template, search, text, "overall"))
    macs, params = clever_format([macs, params], "%.3f")
    print('==>MACs is ', macs)
    print('==>Params is ', params)
    print("\033[0;32;40m Attention: This only includes trainable part! \033[0m")

    T_w = 20
    T_t = 1000
    print("testing speed for evaluation ...")
    with torch.no_grad():
        # warm-up for T_w rounds
        for i in range(T_w):
            _ = model(template, search, text, "overall")

        # time starts
        start = time.time()
        # set template and text only once
        _ = model.set_template_text(template, text)
        # tracking for T_t rounds
        end_1 = time.time()
        for i in range(T_t):
            _ = model.forward_test(template, search)
        end = time.time()
        # time ends
        print(f"diffusion time: {end_1 - start}")
        print(f"frame inference total time: {end - end_1}")
        print(f"frame inference avg time: {(end - end_1) / T_t}")

        # calculate FPS
        avg_lat = (end - start) / T_t
        print("\033[0;32;40m The average overall FPS of {} is {}.\033[0m".format(display_info, 1.0 / avg_lat))


if __name__ == "__main__":
    device = "cuda:0"
    torch.cuda.set_device(device)
    # device = "cpu"

    args = parse_args()
    '''update cfg'''
    prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    yaml_fname = os.path.join(prj_dir, 'experiments/%s/%s.yaml' % (args.script, args.config))
    print("yaml_fname: {}".format(yaml_fname))
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    print("cfg: {}".format(cfg))

    '''set some values'''
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE
    image_dim = cfg.MODEL.HIDDEN_DIM
    if is_torch_bf16_available() and args.precision == 'bf16':
        torch.set_default_dtype(torch.bfloat16)
    else:
        torch.set_default_dtype(torch.float16)
    head_type = cfg.MODEL.HEAD_TYPE

    '''import GLAD network module'''
    model_module = importlib.import_module('lib.models.GLAD')

    if args.script == "GLAD":
        model_constructor = model_module.build_pipeline
        model = model_constructor(cfg, precision=args.precision, check_nan_tensor=False)
        model.transfer_param_dtype()
        # transfer to device
        model = model.to(device)
        model.sd_model.to(device)

        # get template and search
        template = get_data(bs, z_sz, device)
        template_encoder = get_data(bs, x_sz//2, device)
        search = get_data(bs, x_sz, device)
        # get text_features
        text = get_text(bs)

        if args.detailed:
            # evaluate sd_model
            evaluate_GLAD_sd_model()

            # evaluate image_model
            evaluate_GLAD_image_model(model, template_encoder, search)

            # evaluate pooling
            latent_features = get_latent_features(bs, device)
            evaluate_GLAD_pooling(model, latent_features)

            # evaluate decoder
            pooled_latent_features = get_pooled_latent_features(bs, image_dim, device)
            image_features = get_image_features(bs, x_sz, image_dim, device)
            evaluate_GLAD_decoder(model, pooled_latent_features, image_features)

            # evaluate head
            image_features_resized = get_image_features_resized(bs, x_sz, image_dim, device)
            evaluate_GLAD_head(model, image_features_resized, head_type)

        # evaluate overall model
        template, search = template.float(), search.float()
        evaluate_GLAD(model, template, search, text, args.display_name)