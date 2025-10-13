import os
import torch
import copy
from functools import partial
from torch import nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, AutoTokenizer, CLIPTokenizer, CLIPTextConfig, CLIPVisionConfig, CLIPConfig

from .diffusion_block import build_diffusion_block
from .pooling_attention import build_pooling_module, build_pooling_module_large
from .fusion_decoder import build_fusion_decoder, build_fusion_decoder_large
from .head import build_box_head, build_box_head_large

from lib.utils.misc import is_main_process
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.utils.torch_utils import is_torch_bf16_available

from .ViTMAE import build_transformer_base, build_transformer_large


class GLAD(nn.Module):
    """ This is the base class for GLAD(Employ Diffusion Feature for Vision-Language Tracking) """
    def __init__(self, 
                 clip_model, 
                 diffusion_block, 
                 pooling_modules, 
                 fusion_decoders, 
                 head, 
                 head_type='CORNER',
                 text_dim=512,
                 image_dim=768,
                 img_size=320,
                 precision='fp16',
                 check_nan_tensor=True):
        super().__init__()
        # CLIP
        self.encoder = clip_model[0]

        # Stable Diffusion
        self.sd_model = diffusion_block

        # Pooling Module
        self.pm = nn.ModuleList([pooling_modules[0], pooling_modules[1], pooling_modules[2]])

        # Fusion Decoder
        self.fd = nn.ModuleList([fusion_decoders[0], fusion_decoders[1], fusion_decoders[2]])

        # add output norm
        norm_layer=partial(nn.LayerNorm, eps=1e-5)
        self.norm_output = norm_layer(image_dim)
        nn.init.constant_(self.norm_output.bias, 0)
        nn.init.constant_(self.norm_output.weight, 1.0)

        # Prediction Head
        self.head = head
        self.head_type = head_type

        self.img_size = img_size
        self.text_dim = text_dim
        self.image_dim = image_dim

        self.use_bf16 = is_torch_bf16_available() and precision == 'bf16'

        self.check_nan_tensor = check_nan_tensor
    
    def forward_vis_sd(self, template, search, text):
        # Use SD with LCM - Img to Img Visualization
        if not os.path.exists("./vis_sd/"):
            os.makedirs("./vis_sd/")
        for batch_id in range(len(text)):
            prompt = text[batch_id]
            image = template[batch_id]
            # visualize template and search
            pre_img = image.permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")
            pre_img = Image.fromarray(pre_img)
            pre_img.save("./vis_sd/[" + str(image.device)[-1] + " " + str(batch_id) + ']-template_pre_[' + text[batch_id] + '].png')
            search_img = search[batch_id].permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")
            search_img = Image.fromarray(search_img)
            search_img.save("./vis_sd/[" + str(image.device)[-1] + " " + str(batch_id) + ']-search.png')
            # generate Img2Img result
            image = self.sd_model.get_latent_features(prompt=prompt, image=image, strength=0.5, num_inference_steps=2, guidance_scale=1, 
                                                      return_latent=False, display_progress_bar=True).images[0]
            image.save("./vis_sd/[" + str(search[batch_id].device)[-1] + " " + str(batch_id) + ']-template_post_[' + text[batch_id] + '].png')
        exit(0)

    def check_nan(self, x, tensor_type):
        if torch.isnan(x).any():
            print("NAN Tensor at `" + tensor_type + "`.")

    def forward_train(self, template, search, text):
        r'''
        # Example of using stable diffusion model - Text to Img (StableDiffusionPipeline)
        text = "a photo of an astronaut riding a horse on mars"
        image = self.sd_model(text).images[0]
        image.save("astronaut_rides_horse.png")
        '''
        r'''
        # Example of using stable diffusion model - Img to Img (StableDiffusionImg2ImgPipeline)
        prompt = text[1]
        image = template[1]
        # visualize template and search
        pre_img = image.permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")
        pre_img = Image.fromarray(pre_img)
        pre_img.save(text[1] + '_pre.png')
        search_img = search[1].permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")
        search_img = Image.fromarray(search_img)
        search_img.save(str(image.device) + ".png")
        # generate Img2Img result
        image = self.sd_model(prompt=prompt, image=image, strength=0.5, guidance_scale=7.5).images[0]
        image.save(text[1] + '_post.png')
        '''

        # LCM Img2Img pipeline, 2~8 steps recommended
        latent_features = self.sd_model.get_latent_features(prompt=text, 
                                                            image=template, 
                                                            # strength=0.5, 
                                                            # num_inference_steps=2, 
                                                            # guidance_scale=1,  # originally set to 1
                                                            strength=0.125, 
                                                            num_inference_steps=8, 
                                                            guidance_scale=8,
                                                            return_latent=True,
                                                            display_progress_bar=False)

        latent_features = latent_features[:3]
        if self.check_nan_tensor:
            self.check_nan(latent_features[0], "UNet 5th Block")
            self.check_nan(latent_features[1], "UNet 6th Block")
            self.check_nan(latent_features[2], "UNet 7th Block")

        if self.use_bf16:
            template = template.to(torch.bfloat16)
            search = search.to(torch.bfloat16)
        template = torch.nn.functional.interpolate(template, size=(self.img_size // 2, self.img_size // 2))
        template, search = self.encoder(template, search)
        _, t_N, _ = template.shape
        _, s_N, _ = search.shape

        # concat template_image_features with image_features
        '''image_features = torch.cat([template_image_features, image_features], dim=1)'''
        image_features = torch.cat([template, search], dim=1)

        # convert tensor type
        if self.use_bf16:
            latent_features_list = list(latent_features)
            latent_features_list[0] = latent_features_list[0].to(torch.bfloat16)
            latent_features_list[1] = latent_features_list[1].to(torch.bfloat16)
            latent_features_list[2] = latent_features_list[2].to(torch.bfloat16)
            latent_features = tuple(latent_features_list)


        # Attention Pool of latent_features
        pooled_latent_features = ()
        for idx, lf in enumerate(latent_features):
            pooled_lf = self.pm[idx](lf, context=None, mode="attention")
            pooled_latent_features += (pooled_lf, )
        if self.check_nan_tensor:
            self.check_nan(pooled_latent_features[0], "Pooling Module - 1")
            self.check_nan(pooled_latent_features[1], "Pooling Module - 2")
            self.check_nan(pooled_latent_features[2], "Pooling Module - 3")

        # Decoder Fusion of latent_features
        for idx in range(len(self.fd)):
            image_features = self.fd[idx](image_features, latent=pooled_latent_features[idx], context=None)
            if self.check_nan_tensor:
                self.check_nan(image_features, f"Fusion Decoder - {idx+1}")

        # Output norm
        image_features = self.norm_output(image_features)
        if self.check_nan_tensor:
            self.check_nan(image_features, "Output Norm")

        # split image_features to template_image_features and image_features
        _, image_features = torch.split(image_features, [t_N, s_N], dim=1)

        # Head Prediction
        b, n, c = image_features.shape
        image_features = image_features.transpose(1, 2).reshape(b, c, int(n ** 0.5), -1).contiguous()

        out, outputs_coord_new = self.forward_box_head(image_features)
        return out, outputs_coord_new
    
    def forward_box_head(self, search):
        r"""
        search: (b, c, h, w)
        """
        if self.head_type == "CORNER":
            # run the corner head
            b = search.size(0)
            outputs_coord = box_xyxy_to_cxcywh(self.head(search))
            outputs_coord_new = outputs_coord.view(b, 1, 4)
            out = {'pred_boxes': outputs_coord_new}
            return out, outputs_coord_new
        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.head(search, None)
            b = search.size(0)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(b, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out, outputs_coord_new
        else:
            raise KeyError
    
    def forward_test(self, template, search):
        template = torch.nn.functional.interpolate(template, size=(self.img_size // 2, self.img_size // 2))
        if self.use_bf16:
            template = template.to(torch.bfloat16)
            search = search.to(torch.bfloat16)
        template, search = self.encoder(template, search)
        _, t_N, _ = template.shape
        _, s_N, _ = search.shape
        image_features = torch.cat([template, search], dim=1)
        
        # Decoder Fusion of latent_features
        for idx in range(len(self.fd)):
            image_features = self.fd[idx](image_features, latent=self.pooled_latent_features[idx], context=None)
        
        # Output norm
        image_features = self.norm_output(image_features)

        # split image_features to template_image_features and image_features
        '''_, image_features = torch.split(image_features, [self.t_N, s_N], dim=1)'''
        _, image_features = torch.split(image_features, [t_N, s_N], dim=1)

        # Head Prediction
        b, n, c = image_features.shape
        image_features = image_features.transpose(1, 2).reshape(b, c, int(n ** 0.5), -1).contiguous()
        out, outputs_coord_new = self.forward_box_head(image_features)
        return out, outputs_coord_new

    def set_template_text(self, template, text):
        # When evaluating, there are 2 ways: using text or using ""
        # text_sp = ""
        text_sp = text

        # LCM Img2Img pipeline, 2~8 steps recommendeds
        latent_features = self.sd_model.get_latent_features(prompt=text_sp, 
                                                            image=template, 
                                                            # strength=0.5, 
                                                            # num_inference_steps=2, 
                                                            # guidance_scale=1, 
                                                            strength=0.125, 
                                                            num_inference_steps=8, 
                                                            guidance_scale=8,
                                                            return_latent=True,
                                                            display_progress_bar=False)
        latent_features = latent_features[:3]

        # convert tensor type
        if self.use_bf16:
            latent_features_list = list(latent_features)
            latent_features_list[0] = latent_features_list[0].to(torch.bfloat16)
            latent_features_list[1] = latent_features_list[1].to(torch.bfloat16)
            latent_features_list[2] = latent_features_list[2].to(torch.bfloat16)
            latent_features = tuple(latent_features_list)

        # Attention Pool of latent_features
        pooled_latent_features = ()
        for idx, lf in enumerate(latent_features):
            pooled_lf = self.pm[idx](lf, context=None, mode="attention")
            pooled_latent_features += (pooled_lf, )

        self.pooled_latent_features = pooled_latent_features

    def transfer_param_dtype(self):
        # transfer param to fp16 or bf16
        if self.use_bf16:
            self.pm.bfloat16()
            self.fd.bfloat16()
            self.norm_output.bfloat16()
            self.head.bfloat16()
            torch.set_default_dtype(torch.bfloat16)
        else:
            self.pm.half()
            self.fd.half()
            self.norm_output.half()
            self.head.half()
            torch.set_default_dtype(torch.float16)

    def forward(self, template, search, text, mode="overall"):
        # only for Param & MAC profile
        if mode == "overall":
            # self.forward_profile_overall(template, search, text)
            return self.forward_train(template, search, text)
        else:
            raise ValueError("Wrong mode for profile")

    def forward_profile_overall(self, template, search, text):
        # NOTE: sd_model and text_model are not included!
        # LCM Img2Img pipeline, 2~8 steps recommended
        latent_features = self.sd_model.get_latent_features(prompt=text, 
                                                            image=template, 
                                                            strength=0.5, 
                                                            num_inference_steps=2, 
                                                            guidance_scale=7.5,  # originally set to 1
                                                            return_latent=True,
                                                            display_progress_bar=False)
        
        # Text features generated by tokenizer and text_model from CLIP
        text_inputs = self.text_model[1](text, max_length=77, truncation=True, padding=True, return_tensors="pt")
        text_inputs['input_ids'] = text_inputs['input_ids'].to(next(self.text_model[0].parameters()).device)
        text_inputs['attention_mask'] = text_inputs['attention_mask'].to(next(self.text_model[0].parameters()).device)
        text_outputs = self.text_model[0](**text_inputs)  # ('last_hidden_state', 'pooler_output', 'hidden_states'=None, 'attentions'=None)
        text_features = text_outputs[1]  # pooler_output
    
        # Image features of template
        # (2) Siamese structure, concat with search tokens in decoder
        template_for_clip = torch.nn.functional.interpolate(template, size=(self.img_size, self.img_size))
        template_image_outputs = self.image_model(pixel_values=template_for_clip)
        template_image_features = template_image_outputs[0][:, 1:, :]
        pooled_template_image_features = template_image_outputs[1]
        _, t_N, _ = template_image_features.shape

        # Image features generated by image_model from CLIP
        image_outputs = self.image_model(pixel_values=search)  # ('last_hidden_state', 'pooler_output', 'hidden_states'=None, 'attentions'=None)
        image_features = image_outputs[0][:, 1:, :]
        _, s_N, _ = image_features.shape

        # concat template_image_features with image_features
        image_features = torch.cat([template_image_features, image_features], dim=1)

        # Attention Pool of latent_features
        pooled_latent_features = ()
        for idx, lf in enumerate(latent_features):
            pooled_lf = self.pm[idx](lf, context=(text_features, pooled_template_image_features), mode="attention")
            pooled_latent_features += (pooled_lf, )

        # Decoder Fusion of latent_features
        for idx in range(len(self.fd)):
            image_features = self.fd[idx](image_features, latent=pooled_latent_features[idx], context=None)

        # Output norm
        image_features = self.norm_output(image_features)

        # split image_features to template_image_features and image_features
        _, image_features = torch.split(image_features, [t_N, s_N], dim=1)

        # Head Prediction
        b, n, c = image_features.shape
        image_features = image_features.transpose(1, 2).reshape(b, c, int(n ** 0.5), -1).contiguous()

        out, outputs_coord_new = self.forward_box_head(image_features)

def interpolate_pos_embed(checkpoint_model, img_size=256):
    pos_embed_checkpoint = checkpoint_model['embeddings.position_embedding.weight']
    embedding_size = pos_embed_checkpoint.shape[-1]  # 768
    patch_size = checkpoint_model['embeddings.patch_embedding.weight'].shape[-1]
    num_patches = (img_size // patch_size) ** 2  # (256 // 16) ** 2 = 256
    num_extra_tokens = 1
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)  # 14
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)  # 16
    # class_token is kept unchanged
    if orig_size != new_size:
        if is_main_process():
            print("Position Embeddings interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
        extra_tokens = pos_embed_checkpoint[:num_extra_tokens, :]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[num_extra_tokens:, :]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(0, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=0)
        checkpoint_model['embeddings.position_embedding.weight'] = new_pos_embed

def freeze_parameter(net):
    for param in net.parameters():
        param.requires_grad = False

def unfreeze_parameter(net):
    for param in net.parameters():
        param.requires_grad = True

def build_pipeline_mae_base(cfg, precision='fp16', check_nan_tensor=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../resource/pretrained')

    if cfg.MODEL.PRETRAIN_FILE:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    # build mae vit
    image_encoder = build_transformer_base(cfg, pretrained)
    
    # float16 -> bfloat16 on Ampere GPUs
    if is_torch_bf16_available() and precision == 'bf16':
        image_encoder.bfloat16()
    else:
        image_encoder.half()
    
    unfreeze_parameter(image_encoder)

    clip_model = (image_encoder,)

    if cfg.MODEL.DIFF_PRETRAIN_ID:
        diff_pretrained = os.path.join(pretrained_path, cfg.MODEL.DIFF_PRETRAIN_ID)
    else:
        diff_pretrained = ''

    if cfg.MODEL.LCM_ID:
        lcm_pretrained = os.path.join(pretrained_path, cfg.MODEL.LCM_ID)
    else:
        lcm_pretrained = ''

    # build sd_model from RunwayML/SD-1.5
    diffusion_block = build_diffusion_block(cfg, diff_pretrained, lcm_pretrained)
    freeze_parameter(diffusion_block.vae)
    freeze_parameter(diffusion_block.text_encoder)
    freeze_parameter(diffusion_block.unet)


    # build pooling module
    pooling_modules = build_pooling_module(cfg)

    # build fusion decoder
    fusion_decoders = build_fusion_decoder(cfg)
    
    # build prediction head (CORNER/CENTER)
    head = build_box_head(cfg)

    # build pipeline
    pipeline = GLAD(clip_model, diffusion_block, pooling_modules, fusion_decoders, head, cfg.MODEL.HEAD_TYPE, 
                       text_dim=768, image_dim=cfg.MODEL.HIDDEN_DIM, img_size=cfg.DATA.SEARCH.SIZE, 
                       precision=precision, check_nan_tensor=check_nan_tensor)

    return pipeline

def build_pipeline_mae_large(cfg, precision='fp16', check_nan_tensor=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../resource/pretrained')

    if cfg.MODEL.PRETRAIN_FILE:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    # build mae vit
    image_encoder = build_transformer_large(cfg, pretrained)
    
    # float16 -> bfloat16 on Ampere GPUs
    if is_torch_bf16_available() and precision == 'bf16':
        image_encoder.bfloat16()
    else:
        image_encoder.half()
    
    unfreeze_parameter(image_encoder)

    clip_model = (image_encoder,)

    if cfg.MODEL.DIFF_PRETRAIN_ID:
        diff_pretrained = os.path.join(pretrained_path, cfg.MODEL.DIFF_PRETRAIN_ID)
    else:
        diff_pretrained = ''

    if cfg.MODEL.LCM_ID:
        lcm_pretrained = os.path.join(pretrained_path, cfg.MODEL.LCM_ID)
    else:
        lcm_pretrained = ''

    # build sd_model from RunwayML/SD-1.5
    diffusion_block = build_diffusion_block(cfg, diff_pretrained, lcm_pretrained)
    freeze_parameter(diffusion_block.vae)
    freeze_parameter(diffusion_block.text_encoder)
    freeze_parameter(diffusion_block.unet)

    # build pooling module
    pooling_modules = build_pooling_module_large(cfg)

    # build fusion decoder
    fusion_decoders = build_fusion_decoder_large(cfg)
    
    # build prediction head (CORNER/CENTER)
    head = build_box_head_large(cfg)

    # build pipeline
    pipeline = GLAD(clip_model, diffusion_block, pooling_modules, fusion_decoders, head, cfg.MODEL.HEAD_TYPE, 
                       text_dim=768, image_dim=cfg.MODEL.HIDDEN_DIM, img_size=cfg.DATA.SEARCH.SIZE, 
                       precision=precision, check_nan_tensor=check_nan_tensor)

    return pipeline

def build_pipeline(cfg, precision='fp16', check_nan_tensor=True):
    if cfg.MODEL.HIDDEN_DIM == 768:
        pipeline = build_pipeline_mae_base(cfg, precision, check_nan_tensor)
    else:
        pipeline = build_pipeline_mae_large(cfg, precision, check_nan_tensor)
    
    return pipeline
