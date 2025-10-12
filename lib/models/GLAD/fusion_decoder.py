from functools import partial
import torch
from torch import nn
import logging
# logging.getLogger().setLevel(logging.INFO)

from torch.nn.init import trunc_normal_

from timm.models.layers import Mlp, DropPath


CHECK_NAN_DECODER = True


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
    def __init__(self, in_dim=768, out_dim=768, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0., check_nan_tensor=False, name=""):
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
    def __init__(self, q_dim=768, kv_dim=512, hidden_dim=768, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0., check_nan_tensor=False, name=""):
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

        self.pos_embed_q = FixedPositionalEmbedding(q_dim)
        self.pos_embed_k = FixedPositionalEmbedding(kv_dim)

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

    def forward(self, image_feature, latent_feature):
        if self.check_nan_tensor:
            B, N, C = image_feature.shape
            q = self.to_q(image_feature + self.pos_embed_q(image_feature))
            k = self.to_k(latent_feature + self.pos_embed_k(latent_feature))
            # q = self.to_q(image_feature)
            # k = self.to_k(latent_feature)
            v = self.to_v(latent_feature)

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
            B, N, C = image_feature.shape
            q, k, v = self.to_q(image_feature), self.to_k(latent_feature), self.to_v(latent_feature)

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


class FusionDecoder(nn.Module):
    r'''
    Fusion Decoder for Pooled Latent Diffusion Features.
    '''
    def __init__(
        self,
        image_dim=768, 
        latent_dim=512, 
        text_dim=512, 
        out_dim=768, 
        num_heads=12, 
        mlp_ratio=4.0, 
        qkv_bias=False,
        attn_drop=0., 
        proj_drop=0., 
        drop_path=0., 
        act_layer=nn.GELU, 
        norm_layer=partial(nn.LayerNorm, eps=1e-5), 
        fd_name="decoder", 
        check_nan_tensor=False
    ):
        super().__init__()
        # Self Attn.
        self.norm1 = norm_layer(out_dim)
        self.attn1 = SelfAttention(in_dim=image_dim, 
                                   out_dim=image_dim, 
                                   num_heads=num_heads, 
                                   qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop, check_nan_tensor=check_nan_tensor, 
                                   name=fd_name)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # # Search to Text Attn.
        # self.norm2 = norm_layer(out_dim)
        # self.attn2 = CrossAttention(q_dim=image_dim,
        #                             kv_dim=text_dim, 
        #                             hidden_dim=image_dim, 
        #                             num_heads=num_heads, 
        #                             qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop, check_nan_tensor=check_nan_tensor, 
        #                             name="Context Fusion, " + fd_name)
        # self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # # Search to Template(Frozen/Siamese) Attn.
        # self.norm2 = norm_layer(out_dim)
        # self.attn2 = CrossAttention(q_dim=image_dim,
        #                             kv_dim=image_dim, 
        #                             hidden_dim=image_dim, 
        #                             num_heads=num_heads, 
        #                             qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop, check_nan_tensor=check_nan_tensor, 
        #                             name="Context Fusion, " + fd_name)
        # self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Search to Latent Attn.
        self.norm3 = norm_layer(out_dim)
        self.attn3 = CrossAttention(q_dim=image_dim, 
                                   kv_dim=latent_dim, 
                                   hidden_dim=out_dim, 
                                   num_heads=num_heads, 
                                   qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop, check_nan_tensor=check_nan_tensor, 
                                   name="Latent Fusion, " + fd_name)
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # MLP
        self.norm4 = norm_layer(out_dim)
        self.mlp = Mlp(in_features=out_dim, hidden_features=int(out_dim * mlp_ratio), act_layer=act_layer, drop=proj_drop)
        self.drop_path4 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.fd_name = fd_name
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
            print("NAN Tensor at `" + tensor_type + "`, " + self.fd_name + ".")
    
    def forward(self, x, latent, context):
        if self.check_nan_tensor:
            self.check_nan(x, "Input")
            x = self.drop_path1(self.attn1(self.norm1(x))) + x
            self.check_nan(x, "SelfAttention")
            # x = self.drop_path2(self.attn2(self.norm2(x), context)) + x
            # self.check_nan(x, "CrossAttention_context")
            x = self.drop_path3(self.attn3(self.norm3(x), latent)) + x
            self.check_nan(x, "CrossAttention_latent")
            x = self.drop_path4(self.mlp(self.norm4(x))) + x
            self.check_nan(x, "MLP")
        else:
            x = self.drop_path1(self.attn1(self.norm1(x))) + x
            # x = self.drop_path2(self.attn2(self.norm2(x), context)) + x
            x = self.drop_path3(self.attn3(self.norm3(x), latent)) + x
            x = self.drop_path4(self.mlp(self.norm4(x))) + x
        return x


def build_fusion_decoder(cfg):
    dpr = [0.0, 0.0, 0.0, 0.0]
    # dpr = [x.item() for x in torch.linspace(0, 0.1, 4)]  # stochastic depth decay rule
    left_decoder_1 = FusionDecoder(image_dim=768, latent_dim=768, text_dim=512, out_dim=768, num_heads=12, drop_path=dpr[0], 
                                 fd_name="left_decoder_1", check_nan_tensor=CHECK_NAN_DECODER)
    left_decoder_2 = FusionDecoder(image_dim=768, latent_dim=768, text_dim=512, out_dim=768, num_heads=12, drop_path=dpr[1], 
                                 fd_name="left_decoder_2", check_nan_tensor=CHECK_NAN_DECODER)
    mid_decoder = FusionDecoder(image_dim=768, latent_dim=768, text_dim=512, out_dim=768, num_heads=12, drop_path=dpr[2], 
                                fd_name="mid_decoder", check_nan_tensor=CHECK_NAN_DECODER)
    
    return (left_decoder_1, left_decoder_2, mid_decoder)

def build_fusion_decoder_large(cfg):
    dpr = [0.0, 0.0, 0.0, 0.0]
    # dpr = [x.item() for x in torch.linspace(0, 0.1, 4)]  # stochastic depth decay rule
    left_decoder_1 = FusionDecoder(image_dim=1024, latent_dim=1024, text_dim=768, out_dim=1024, num_heads=16, drop_path=dpr[0], 
                                 fd_name="left_decoder_1", check_nan_tensor=CHECK_NAN_DECODER)
    left_decoder_2 = FusionDecoder(image_dim=1024, latent_dim=1024, text_dim=768, out_dim=1024, num_heads=16, drop_path=dpr[1], 
                                 fd_name="left_decoder_2", check_nan_tensor=CHECK_NAN_DECODER)
    mid_decoder = FusionDecoder(image_dim=1024, latent_dim=1024, text_dim=768, out_dim=1024, num_heads=16, drop_path=dpr[2], 
                                fd_name="mid_decoder", check_nan_tensor=CHECK_NAN_DECODER)
    
    return (left_decoder_1, left_decoder_2, mid_decoder)
