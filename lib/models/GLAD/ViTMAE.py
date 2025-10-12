from functools import partial
from itertools import repeat
# from torch._six import container_abcs
import collections.abc as container_abcs

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from timm.models.layers import Mlp, DropPath
from timm.layers import use_fused_attn

from lib.utils.misc import is_main_process


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]  # 768

        # for template
        num_patches_t = model.patch_embed_t.num_patches  # 8*8=64
        num_extra_tokens_t = model.pos_embed_t.shape[-2] - num_patches_t + 1  # need to drop cls_token
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens_t) ** 0.5)  # 14
        # height (== width) for the new position embedding
        new_size = int(num_patches_t ** 0.5)  # 8
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Template Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens_t]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens_t:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            # new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            new_pos_embed = pos_tokens
            checkpoint_model['pos_embed_t'] = new_pos_embed
        else:
            print("Template Position %dx%d to %dx%d, no need to interpolate" % (orig_size, orig_size, new_size, new_size))
            checkpoint_model['pos_embed_t'] = pos_embed_checkpoint[:, num_extra_tokens_t:]

        # for search
        num_patches_s = model.patch_embed_s.num_patches  # 16*16=256
        num_extra_tokens_s = model.pos_embed_s.shape[-2] - num_patches_s + 1  # need to drop cls_token
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens_t) ** 0.5)  # 14
        # height (== width) for the new position embedding
        new_size = int(num_patches_s ** 0.5)  # 16
        if orig_size != new_size:
            print("Search Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens_s]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens_s:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            # new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            new_pos_embed = pos_tokens
            checkpoint_model['pos_embed_s'] = new_pos_embed
        else:
            print("Search Position %dx%d to %dx%d, no need to interpolate" % (orig_size, orig_size, new_size, new_size))
            checkpoint_model['pos_embed_t'] = pos_embed_checkpoint[:, num_extra_tokens_s:]


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True,
                 stride=16, padding=0):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # (input - kernel + 2 * padding) / stride + 1
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class EmptyPatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding without param
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True,
                 stride=16, padding=0):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten


class Attention(nn.Module):
    def __init__(self, t_size, s_size, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        if is_main_process():
            print("use_fused_attn =", self.fused_attn)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.t_size = t_size  # 64 for ROMTrack, 144 for ROMTrack-384
        self.s_size = s_size  # 256 for ROMTrack, 576 for ROMTrack-384
        self.mix_size = self.t_size + self.s_size

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # B, num_heads, N, C//num_heads
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # if self.fused_attn:
        #     x = F.scaled_dot_product_attention(
        #         q, k, v,
        #         dropout_p=self.attn_drop.p if self.training else 0.,
        #     )
        # else:
        #     q = q * self.scale
        #     attn = q @ k.transpose(-2, -1)
        #     attn = attn.softmax(dim=-1)
        #     attn = self.attn_drop(attn)
        #     x = attn @ v
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
            self, t_size, s_size, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(t_size, s_size, dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self, img_size_t=128, img_size_s=256, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, init_values=None,
            class_token=True, fc_norm=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='',
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            class_token (bool): use class token
            fc_norm (Optional[bool]): pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init (str): weight init scheme
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        # use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1 if class_token else 0
        self.grad_checkpointing = False

        self.patch_embed_t = embed_layer(
            img_size=img_size_t, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, stride=patch_size)
        num_patches_t = self.patch_embed_t.num_patches

        self.patch_embed_s = EmptyPatchEmbed(
            img_size=img_size_s, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, stride=patch_size)
        num_patches_s = self.patch_embed_s.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if self.num_tokens > 0 else None
        self.pos_embed_t = nn.Parameter(torch.randn(1, num_patches_t + self.num_tokens, embed_dim) * .02)
        self.pos_embed_s = nn.Parameter(torch.randn(1, num_patches_s + self.num_tokens, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        t_size = (img_size_t // patch_size) ** 2
        s_size = (img_size_s // patch_size) ** 2
        self.patch_size = patch_size
        self.depth = depth
        self.blocks = nn.Sequential(*[
            block_fn(
                t_size=t_size, s_size=s_size, dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        # self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()
        self.norm = norm_layer(embed_dim)

    def forward_features(self, template, search):
        x = torch.cat([template, search], dim=1)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x

    def forward(self, template, search):
        t_B, t_C, t_H, t_W = template.shape
        s_B, s_C, s_H, s_W = search.size()
        t_H, t_W, s_H, s_W = t_H // self.patch_size, t_W // self.patch_size, s_H // self.patch_size, s_W // self.patch_size

        template = self.patch_embed_t(template)
        template = self.pos_drop(template + self.pos_embed_t)
        search = self.patch_embed_t(search)
        search = self.pos_drop(search + self.pos_embed_s)

        x = self.forward_features(template, search)
        template, search = torch.split(x, [t_H * t_W, s_H * s_W], dim=1)

        return template, search


# Load models
def load_checkpoint(cur_encoder, ckpt_path):
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if "small" in ckpt_path:
            ckpt_type = "DINO"
        else:
            ckpt_type = "MAE"
            ckpt = ckpt['model']
        if "tiny" in ckpt_path:
            ckpt = {k.replace("module.model.", ""): v for k, v in ckpt.items()}
        # interpolate position embedding
        interpolate_pos_embed(cur_encoder, ckpt)
        # copy patch_embed
        ckpt['patch_embed_t.proj.weight'] = ckpt['patch_embed.proj.weight']
        ckpt['patch_embed_t.proj.bias'] = ckpt['patch_embed.proj.bias']
        # load
        model_ckpt = cur_encoder.state_dict()
        state_ckpt = {k: v for k, v in ckpt.items() if k in model_ckpt.keys()}
        model_ckpt.update(state_ckpt)
        missing_keys, unexpected_keys = cur_encoder.load_state_dict(model_ckpt, strict=False)
        # print to check
        for k, v in cur_encoder.named_parameters():
            if k in ckpt.keys():
                if is_main_process():
                    print(k)
            else:
                if is_main_process():
                    print("# not in ckpt: " + k)
        if is_main_process():
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
            print("Loading pretrained {} done.".format(ckpt_type))
    except Exception as e:
        print("Warning: Pretrained weights are not loaded.")
        print(e.args)


def build_transformer_base(cfg, pretrained):
    model = VisionTransformer(
        img_size_t=cfg.DATA.SEARCH.SIZE // 2, img_size_s=cfg.DATA.SEARCH.SIZE, patch_size=16, 
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        class_token=False, drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool='')
    load_checkpoint(model, ckpt_path=pretrained)

    return model


def build_transformer_large(cfg, pretrained):
    model = VisionTransformer(
        img_size_t=cfg.DATA.SEARCH.SIZE // 2, img_size_s=cfg.DATA.SEARCH.SIZE, patch_size=16, 
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        class_token=False, drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool='')
    load_checkpoint(model, ckpt_path=pretrained)

    return model

