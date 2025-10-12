from functools import partial
import torch
from torch import nn
import logging
# logging.getLogger().setLevel(logging.INFO)

from torch.nn.init import trunc_normal_

from timm.models.layers import Mlp, DropPath


CHECK_NAN_POOL = True


class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, seq_dim=1, offset=0):
        t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq) + offset
        sinusoid_inp = torch.einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb[None, :, :]


class SelfAttention(nn.Module):
    def __init__(self, in_dim=1280, out_dim=512, num_heads=20, qkv_bias=False, attn_drop=0., proj_drop=0., check_nan_tensor=False, name=""):
        super().__init__()
        assert in_dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = in_dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.check_nan_tensor = check_nan_tensor
        self.name = name
    
    def check_inf(self, x, tensor_type):
        # check inf tensor and then clamp to avoid nan
        if torch.isinf(x).any():
            print("INF Tensor at `" + tensor_type + "`, Self Attention, " + self.name + ".")
            print("Clamp tensor to [-10000, 10000]")
            x.clamp_(min=-1e4, max=1e4)

    def check_nan(self, x, tensor_type):
        if torch.isnan(x).any():
            print("NAN Tensor at `" + tensor_type + "`, Self Attention.")

    def forward(self, x):
        if self.check_nan_tensor:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            # B, num_heads, N, C//num_heads
            q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

            # self attention
            self.check_nan(q, "q")
            self.check_nan(k, "k")
            self.check_nan(v, "v")
            attn = (q @ k.transpose(-2, -1)) * self.scale
            self.check_nan(attn, "QK^T")
            self.check_inf(attn, "QK^T")
            attn = attn.softmax(dim=-1)
            # # try log_softmax to avoid nan
            # attn = attn.log_softmax(dim=-1).exp()
            self.check_nan(attn, "Softmax")
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            self.check_nan(x, "*V")
            x = self.proj(x)
            self.check_nan(x, "proj")
            x = self.proj_drop(x)
        else: 
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            # B, num_heads, N, C//num_heads
            q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

            # self attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            # # try log_softmax to avoid nan
            # attn = attn.log_softmax(dim=-1).exp()
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            
        return x


class CrossAttention(nn.Module):
    def __init__(self, q_dim=512, kv_dim=512, hidden_dim=512, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., check_nan_tensor=False, name=""):
        super().__init__()
        assert hidden_dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(q_dim, hidden_dim, bias=qkv_bias)
        self.to_k = nn.Linear(kv_dim, hidden_dim, bias=qkv_bias)
        self.to_v = nn.Linear(kv_dim, hidden_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(hidden_dim, q_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.check_nan_tensor = check_nan_tensor
        self.name = name

    def check_inf(self, x, tensor_type):
        # check inf tensor and then clamp to avoid nan
        if torch.isinf(x).any():
            print("INF Tensor at `" + tensor_type + "`, Cross Attention, " + self.name + ".")
            print("Clamp tensor to [-10000, 10000]")
            x.clamp_(min=-1e4, max=1e4)
    
    def check_nan(self, x, tensor_type):
        if torch.isnan(x).any():
            print("NAN Tensor at `" + tensor_type + "`, Cross Attention.")

    def forward(self, unet_feature, text_feature):
        if self.check_nan_tensor:
            B, N, C = unet_feature.shape
            q, k, v = self.to_q(unet_feature), self.to_k(text_feature), self.to_v(text_feature)

            # cross attention
            self.check_nan(q, "q")
            self.check_nan(k, "k")
            self.check_nan(v, "v")
            attn = (q @ k.transpose(-2, -1)) * self.scale
            self.check_nan(attn, "QK^T")
            self.check_inf(attn, "QK^T")
            attn = attn.softmax(dim=-1)
            # # try log_softmax and exp
            # attn = attn.log_softmax(dim=-1).exp()
            self.check_nan(attn, "Softmax")
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            self.check_nan(x, "*V")
            x = self.proj(x)
            self.check_nan(x, "proj")
            x = self.proj_drop(x)
        else:
            B, N, C = unet_feature.shape
            q, k, v = self.to_q(unet_feature), self.to_k(text_feature), self.to_v(text_feature)

            # cross attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            # # try log_softmax and exp
            # attn = attn.log_softmax(dim=-1).exp()
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)

        return x
    

class Pool2dAttention(nn.Module):
    r'''
    Pooling Function for Latent Diffusion Features.
    '''
    def __init__(
          self, 
          size=256, 
          unet_dim=1280, 
          text_dim=512, 
          image_dim=768, 
          out_dim=512, 
          num_heads=8, 
          mlp_ratio=4.0, 
          qkv_bias=False,
          attn_drop=0., 
          proj_drop=0., 
          drop_path=0., 
          act_layer=nn.GELU, 
          norm_layer=partial(nn.LayerNorm, eps=1e-5), 
          pm_name="pool", 
          check_nan_tensor=False
    ):
        super().__init__()
        # positional encoding
        # self.pos_embed = FixedPositionalEmbedding(unet_dim)
        self.pos_embed = nn.Parameter(torch.randn(size, unet_dim) / unet_dim ** 0.5)

        self.norm_sd = norm_layer(unet_dim)
        self.self_attention = SelfAttention(in_dim=unet_dim, 
                                            out_dim=out_dim, 
                                            num_heads=20, 
                                            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop, check_nan_tensor=check_nan_tensor, 
                                            name=pm_name)
        
        self.norm_text = norm_layer(out_dim)
        self.cross_attention_text = CrossAttention(q_dim=out_dim, 
                                              kv_dim=text_dim, 
                                              hidden_dim=out_dim, 
                                              num_heads=num_heads, 
                                              qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop, check_nan_tensor=check_nan_tensor, 
                                              name="Text Pooling, " + pm_name)
        self.norm_image = norm_layer(out_dim)
        self.cross_attention_image = CrossAttention(q_dim=out_dim,
                                              kv_dim=image_dim,
                                              hidden_dim=out_dim,
                                              num_heads=num_heads,
                                              qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop, check_nan_tensor=check_nan_tensor, 
                                              name="Image Pooling, " + pm_name)

        # self.text_proj = nn.Linear(text_dim, out_dim, bias=True)
        # self.image_proj = nn.Linear(image_dim, out_dim, bias=True)

        self.norm_mlp = norm_layer(out_dim)
        self.mlp = Mlp(in_features=out_dim, hidden_features=int(out_dim * mlp_ratio), act_layer=act_layer, drop=proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # self.norm_latent = norm_layer([size, out_dim])
        self.norm_latent = norm_layer(out_dim)

        self.act = act_layer()

        self.pm_name = pm_name
        self.check_nan_tensor = check_nan_tensor

        self._reset_parameters('trunc_norm')

    def _reset_parameters(self, init):
        if init == 'xavier':
            self.apply(self._init_weights_xavier)
        elif init == 'trunc_norm':
            self.apply(self._init_weights_trunc_normal)
        elif init == 'xavier_stable':
            self.apply(self._init_weights_xavier_stable)
        else:
            raise RuntimeError(F"init method should be xavier/trunc_norm, not {init}.")

    def _init_weights_xavier_stable(self):
        for i, layer in enumerate(self.encoder.layers):
            for p in layer.paramters():
                if isinstance(p, nn.Linear):
                    logging.info('=> init weight of Linear from xavier uniform')
                    nn.init.xavier_uniform_(p.weight, gain=self.num_encoder_layers - i + 1)
                    if p.bias is not None:
                        logging.info('=> init bias of Linear to zeros')
                        nn.init.constant_(p.bias, 0)
                elif isinstance(p, (nn.LayerNorm, nn.BatchNorm2d)):
                    nn.init.constant_(p.bias, 0)
                    nn.init.constant_(p.weight, 1.0)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from xavier uniform')
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def check_nan(self, x, tensor_type):
        if torch.isnan(x).any():
            print("NAN Tensor at `" + tensor_type + "`, " + self.pm_name + ".")

    def forward_context_pool(self, x, context):
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

        if self.check_nan_tensor:
            self.check_nan(x, "Input")
            # x = x + self.pos_embed(x)  # add fixed positional encoding
            x = x + self.pos_embed  # add learnable positional encoding
            self.check_nan(x, "PositionalEmbedding")

            x = self.self_attention(self.norm_sd(x))
            # x = self.self_attention(x)
            self.check_nan(x, "SelfAttention")
            x = self.cross_attention_text(self.norm_text(x), context[0]) + x
            self.check_nan(x, "CrossAttention_Text")
            x = self.cross_attention_image(self.norm_image(x), context[1]) + x
            self.check_nan(x, "CrossAttention_Image")
            x = self.drop_path(self.mlp(self.norm_mlp(x))) + x
            self.check_nan(x, "MLP")
        else:
            # x = x + self.pos_embed(x)  # add fixed positional encoding
            x = x + self.pos_embed  # add learnable positional encoding

            x = self.self_attention(self.norm_sd(x))
            # x = self.self_attention(x)
            x = self.cross_attention_text(self.norm_text(x), context[0]) + x
            x = self.cross_attention_image(self.norm_image(x), context[1]) + x
            x = self.drop_path(self.mlp(self.norm_mlp(x))) + x

        return x
    
    def forward_mix_pool(self, x, context):
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

        if self.check_nan_tensor:
            self.check_nan(x, "Input")
            # x = x + self.pos_embed(x)  # add fixed positional encoding
            x = x + self.pos_embed  # add learnable positional encoding
            self.check_nan(x, "PositionalEmbedding")

            x = self.self_attention(self.norm_sd(x))
            # x = self.self_attention(x)
            self.check_nan(x, "SelfAttention")
            context_text = self.act(self.text_proj(context[0]))
            context_text = context_text.unsqueeze(1)
            self.check_nan(x, "Text Mix")
            context_image = self.act(self.image_proj(context[1]))
            context_image = context_image.unsqueeze(1)
            self.check_nan(x, "Image Mix")
            x = x + 0.5 * torch.mul(x, context_text) + 0.5 * torch.mul(x, context_image)
            self.check_nan(x, "Mix Fusion")
            x = self.drop_path(self.mlp(self.norm_mlp(x))) + x
            self.check_nan(x, "MLP")
        else:
            # x = x + self.pos_embed(x)  # add fixed positional encoding
            x = x + self.pos_embed  # add learnable positional encoding

            x = self.self_attention(self.norm_sd(x))
            # x = self.self_attention(x)
            context_text = self.act(self.text_proj(context[0]))
            context_text = context_text.unsqueeze(1)
            context_image = self.act(self.image_proj(context[1]))
            context_image = context_image.unsqueeze(1)
            x = x + 0.5 * torch.mul(x, context_text) + 0.5 * torch.mul(x, context_image)
            x = self.drop_path(self.mlp(self.norm_mlp(x))) + x
        
        return x

    def forward_pool(self, x):
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

        if self.check_nan_tensor:
            self.check_nan(x, "Input")
            # x = x + self.pos_embed(x)  # add fixed positional encoding
            x = x + self.pos_embed.to(x.device)  # add learnable positional encoding
            self.check_nan(x, "PositionalEmbedding")
            x = self.self_attention(self.norm_sd(x))
            # x = self.self_attention(x)
            self.check_nan(x, "SelfAttention")
        else:
            # x = x + self.pos_embed(x)  # add fixed positional encoding
            x = x + self.pos_embed  # add learnable positional encoding

            x = self.self_attention(self.norm_sd(x))
            # x = self.self_attention(x)

        return x
    
    def forward(self, x, context, mode="attention"):
        if context is None:
            ret = self.forward_pool(x)
        else:
            if mode == "attention":
                ret = self.forward_context_pool(x, context)
            else:
                ret = self.forward_mix_pool(x, context)
        
        if self.check_nan_tensor:
            ret = self.norm_latent(ret)
            self.check_nan(ret, "NormLatent")
        else:
            ret = self.norm_latent(ret)
        
        return ret


def build_pooling_module(cfg):
    dpr = [0.0, 0.0, 0.0, 0.0]
    # dpr = [x.item() for x in torch.linspace(0, 0.1, 4)]  # stochastic depth decay rule
    left_block5_pool = Pool2dAttention(size=256, unet_dim=1280, text_dim=512, image_dim=768, out_dim=768, num_heads=12, drop_path=dpr[0], 
                                       pm_name="block_5th pooling module", check_nan_tensor=CHECK_NAN_POOL)
    left_block6_pool = Pool2dAttention(size=256, unet_dim=1280, text_dim=512, image_dim=768, out_dim=768, num_heads=12, drop_path=dpr[1], 
                                       pm_name="block_6th pooling module", check_nan_tensor=CHECK_NAN_POOL)
    mid_block7_pool = Pool2dAttention(size=64, unet_dim=1280, text_dim=512, image_dim=768, out_dim=768, num_heads=12, drop_path=dpr[2], 
                                      pm_name="block_7th pooling module", check_nan_tensor=CHECK_NAN_POOL)

    return (left_block5_pool, left_block6_pool, mid_block7_pool)


def build_pooling_module_large(cfg):
    dpr = [0.0, 0.0, 0.0, 0.0]
    # dpr = [x.item() for x in torch.linspace(0, 0.1, 4)]  # stochastic depth decay rule
    left_block5_pool = Pool2dAttention(size=256, unet_dim=1280, text_dim=768, image_dim=1024, out_dim=1024, num_heads=16, drop_path=dpr[0], 
                                       pm_name="block_5th pooling module", check_nan_tensor=CHECK_NAN_POOL)
    left_block6_pool = Pool2dAttention(size=256, unet_dim=1280, text_dim=768, image_dim=1024, out_dim=1024, num_heads=16, drop_path=dpr[1], 
                                       pm_name="block_6th pooling module", check_nan_tensor=CHECK_NAN_POOL)
    mid_block7_pool = Pool2dAttention(size=64, unet_dim=1280, text_dim=768, image_dim=1024, out_dim=1024, num_heads=16, drop_path=dpr[2], 
                                      pm_name="block_7th pooling module", check_nan_tensor=CHECK_NAN_POOL)
    return (left_block5_pool, left_block6_pool, mid_block7_pool)

