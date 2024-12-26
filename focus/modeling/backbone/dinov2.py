# ----------------------------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py
# Modified by Zuyao You (https://github.com/geshang777)
# ----------------------------------------------------------------------------------------------------

import math
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from functools import partial
import cv2
import math
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
from ..edge_enhancer.edge_enhancer import InteractionBlock, InteractionBlockWithCls,EdgePriorModule, deform_inputs, Bottleneck
from mmengine.model.base_module import BaseModule
import logging
from torch import Tensor
from timm.models.layers import DropPath, to_2tuple
import torchvision.transforms.functional as TF
import math
from functools import partial
from typing import Callable, Optional
import torch.utils.checkpoint as cp

class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class ConvTokenizer(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        torch.cuda.empty_cache()
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import SwiGLU
        XFORMERS_AVAILABLE = True
    else:
        raise ImportError
except ImportError:
    SwiGLU = SwiGLUFFN
    XFORMERS_AVAILABLE = False

class SwiGLUFFNFused(SwiGLU):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""

    def __init__(
        self, img_size=384, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, bias=True
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        torch.cuda.empty_cache()
        x = self.proj(x)
        _, _, H, W = x.shape
        if self.flatten:
            x = x.flatten(2).transpose(1, 2).contiguous()  # BCHW -> BNC
        x = self.norm(x)
        return x, H, W


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W, return_attn=False):
        torch.cuda.empty_cache()
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attn:
            return attn

        return x


class MemEffAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, H, W, return_attn=False) -> Tensor:
        torch.cuda.empty_cache()
        from xformers.ops import memory_efficient_attention, unbind
        # import xformers.ops as xops
        if return_attn:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

            q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
            attn = q @ k.transpose(-2, -1).contiguous()

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)


            return attn

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)
        # q = q.half()
        # k = k.half()
        # v = v.half()
        x = memory_efficient_attention(q, k, v)
        # x = xops.memory_efficient_attention(q, k, v)
        # x = x.float()
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowedAttention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, window_size=14, pad_mode="constant"
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.window_size = window_size
        self.pad_mode = pad_mode

    def forward(self, x, H, W):
        torch.cuda.empty_cache()
        B, N, C = x.shape
        N_ = self.window_size * self.window_size
        H_ = math.ceil(H / self.window_size) * self.window_size
        W_ = math.ceil(W / self.window_size) * self.window_size

        qkv = self.qkv(x)  # [B, N, C]
        qkv = qkv.transpose(1, 2).reshape(B, C * 3, H, W).contiguous()  # [B, C, H, W]
        qkv = F.pad(qkv, [0, W_ - W, 0, H_ - H], mode=self.pad_mode)

        qkv = F.unfold(
            qkv, kernel_size=(self.window_size, self.window_size), stride=(self.window_size, self.window_size)
        )
        B, C_kw_kw, L = qkv.shape  # L - the num of windows
        qkv = qkv.reshape(B, C * 3, N_, L).permute(0, 3, 2, 1).contiguous()  # [B, L, N_, C]
        qkv = qkv.reshape(B, L, N_, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5).contiguous()
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # q,k,v [B, L, num_head, N_, C/num_head]
        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale  # [B, L, num_head, N_, N_]
        # if self.mask:
        #     attn = attn * mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # [B, L, num_head, N_, N_]
        # attn @ v = [B, L, num_head, N_, C/num_head]
        x = (attn @ v).permute(0, 2, 4, 3, 1).reshape(B, C_kw_kw // 3, L).contiguous()

        x = F.fold(
            x,
            output_size=(H_, W_),
            kernel_size=(self.window_size, self.window_size),
            stride=(self.window_size, self.window_size),
        )  # [B, C, H_, W_]
        x = x[:, :, :H, :W].reshape(B, C, N).transpose(-1, -2).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x




class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        windowed=False,
        window_size=14,
        pad_mode="constant",
        layer_scale=False,
        with_cp=False,
        ffn_layer=Mlp,
        memeff=False,
    ):
        super().__init__()
        self.with_cp = with_cp
        self.norm1 = norm_layer(dim)
        if windowed:
            self.attn = WindowedAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                window_size=window_size,
                pad_mode=pad_mode,
            )
        elif memeff:
            self.attn = MemEffAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
            )
        else:
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.layer_scale = layer_scale
        if layer_scale:
            self.gamma1 = nn.Parameter(torch.ones((dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(torch.ones((dim)), requires_grad=True)

    def forward(self, x, H, W, return_attention=False):
        torch.cuda.empty_cache()
        def _inner_forward(x):
            if self.layer_scale:
                x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x), H, W))
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x), H, W))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x, use_reentrant=False)
        else:
            x = _inner_forward(x)
        if return_attention:
            return self.attn(self.norm1(x), H, W,return_attn=True)


        return x


class TIMMVisionTransformer(BaseModule):
    """Vision Transformer.

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(
        self,
        img_size=518, 
        patch_size=4,
        in_chans=3,
        num_classes=2,
        embed_dim= 1536, #768,
        depth=40, #12,
        num_heads=24,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.4, #0.0,
        layer_scale=True,
        embed_layer=PatchEmbed,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        window_attn=False,
        window_size=14,
        pretrained=None,
        with_cp=False,
        pre_norm=False,
        ffn_type="swiglu",
        memeff=True,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            pretrained: (str): pretrained path
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.pretrain_size = img_size
        self.drop_path_rate = drop_path_rate
        self.drop_rate = drop_rate
        self.patch_size = patch_size

        window_attn = [window_attn] * depth if not isinstance(window_attn, list) else window_attn
        window_size = [window_size] * depth if not isinstance(window_size, list) else window_size
        logging.info("window attention:", window_attn)
        logging.info("window size:", window_size)
        logging.info("layer scale:", layer_scale)

        self.patch_embed = embed_layer(

            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, bias=not pre_norm
        )
        num_patches = self.patch_embed.num_patches


        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        
        self.pos_drop = nn.Dropout(p=drop_rate)

        ffn_types = {"mlp": Mlp, "swiglu": SwiGLUFFNFused}

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    windowed=window_attn[i],
                    window_size=window_size[i],
                    layer_scale=layer_scale,
                    with_cp=with_cp,
                    ffn_layer=ffn_types[ffn_type],
                    memeff=memeff,
                )
                for i in range(depth)
            ]
        )

        # self.norm = norm_layer(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        # For CLIP
        if pre_norm:
            norm_pre = norm_layer(embed_dim)
            self.norm_pre = norm_pre
        else:
            self.norm_pre = nn.Identity()


    def forward_features(self, x):
        torch.cuda.empty_cache()
        x, H, W = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        # For CLIP
        x = self.norm_pre(x)

        for blk in self.blocks:
            x = blk(x, H, W)
        x = self.norm(x)
        return x
    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def forward(self, x):
        torch.cuda.empty_cache()
        x = self.forward_features(x)
        return x

class EdgeEnhancer(TIMMVisionTransformer):
    def __init__(
        self,
        pretrain_size=518,
        num_heads=24,
        embed_dim = 1536,
        patch_size = 14,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=24,
        init_values=0.0,
        interaction_indexes=[[0, 9], [10, 19], [20, 29], [30, 39]],
        with_cffn=True,
        ffn_type="mlp",
        depth=40,
        cffn_ratio=0.25,
        deform_ratio= 0.5,
        add_vit_feature=True,
        pretrained=None,
        use_extra_extractor=True,
        freeze_vit=True,
        use_cls=True,#True
        with_cp=False,
        frozen_stages=-1,
        *args,
        **kwargs


    ):
 

        super().__init__(num_heads=num_heads, 
                         pretrained=pretrained, 
                         with_cp=with_cp, 
                         embed_dim = embed_dim,
                         patch_size=patch_size,
                         ffn_type=ffn_type,
                         depth=depth,

                         *args, **kwargs)
        
        if freeze_vit:
            for param in self.parameters():
                param.requires_grad = False
        # self.num_classes = 80
        self.use_cls = use_cls
        if not self.use_cls:
            self.cls_token = None
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        self.patch_size = patch_size

        block_fn = InteractionBlockWithCls if use_cls else InteractionBlock

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.edge = EdgePriorModule(Bottleneck, [3, 4, 6, 3], embed_dim=embed_dim)
        # self.pos_embed = nn.Parameter(torch.zeros(1, 14 + self.num_tokens, embed_dim))
        self.interactions = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=deform_num_heads,
                    n_points=n_points,
                    init_values=init_values,
                    drop_path=self.drop_path_rate,
                    norm_layer=self.norm_layer,
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    extra_extractor=((True if i == len(interaction_indexes) - 1 else False) and use_extra_extractor),
                    with_cp=with_cp,
                )
                for i in range(len(interaction_indexes))
            ]
        )
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)
        self.frozen_stages = frozen_stages

        normal_(self.level_embed)
        self._freeze_stages()

    def _get_pos_embed(self, pos_embed, H, W):

        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // self.patch_size, self.pretrain_size[1] // self.patch_size, -1
        ).permute(0, 3, 1, 2).contiguous()
        pos_embed = (
            F.interpolate(pos_embed, size=(H, W), mode="bicubic", align_corners=False)
            .reshape(1, -1, H * W)
            .permute(0, 2, 1).contiguous()
        )
        return pos_embed
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for param in self.parameters():
                param.requires_grad = False

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4
    def _inject_edges(self, batch_tensor):
        # batch_tensor: shape [B, C, H, W] and should be on CPU
        batch_tensor = batch_tensor.clone() 
        for i in range(batch_tensor.shape[0]):  
            image_tensor = batch_tensor[i]

            image = image_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            image_np = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_np = cv2.GaussianBlur(image_np, (5, 5), 1.4)

            edges = np.zeros_like(image_np)
            for c in range(image_np.shape[2]):
                edges[:, :, c] = cv2.Canny(image_np[:, :, c], 100, 200)

            combined_image = cv2.addWeighted(image_np, 0.7, edges, 0.3, 0)

            batch_tensor[i] = TF.to_tensor(combined_image / 255.0)  

        return batch_tensor
    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
        sx = float(w0 + 0.1) / M
        sy = float(h0 + 0.1) / M
        kwargs["scale_factor"] = (sx, sy)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2).contiguous(),
            mode="bicubic",
            antialias=False,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim).contiguous()
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)
    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x, H, W = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)



        return x, H, W
    def get_last_selfattention(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)
        
        x, H, W = self.prepare_tokens_with_masks(x, masks)
        # x, H, W = self.patch_embed(x)
        # Run through model, at the last block just return the attention.
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, H, W)
            else: 
                return blk(x, H, W, return_attention=True)

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x, self.patch_size)

        # Edge Information Extraction
        c1, c2, c3, c4 = self.edge(self._inject_edges(x).type(x.dtype).to(x.device))
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        x, H_toks, W_toks = self.patch_embed(x)

        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H_toks, W_toks)

        
        if self.use_cls:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_token, x), dim=1)
            pos_embed = torch.cat((self.pos_embed[:, :1], pos_embed), dim=1)
        x = self.pos_drop(x + pos_embed)
        # For CLIP
        x = self.norm_pre(x)

        # Interaction
        if self.use_cls:
            cls, x = (
                x[
                    :,
                    :1,
                ],
                x[
                    :,
                    1:,
                ],
            )
        outs = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            if self.use_cls:
                x, c, cls = layer(
                    x,
                    c,
                    cls,
                    self.blocks[indexes[0] : indexes[-1] + 1],
                    deform_inputs1,
                    deform_inputs2,
                    H_c,
                    W_c,
                    H_toks,
                    W_toks,
                )
            else:
                x, c = layer(
                    x,
                    c,
                    self.blocks[indexes[0] : indexes[-1] + 1],
                    deform_inputs1,
                    deform_inputs2,
                    H_c,
                    W_c,
                    H_toks,
                    W_toks,
                )
            outs.append(x.transpose(1, 2).view(bs, dim, H_toks, W_toks).contiguous())
        mask = None
        #PCA for feature mask initialization
        if self.training:
            fg_pca = PCA(n_components=1)
            patch_tokens = x.cpu().detach().numpy().reshape([bs,dim,-1])
            all_patches = patch_tokens.reshape([-1,dim])
            reduced_patches = fg_pca.fit_transform(all_patches)
            norm_patches = minmax_scale(reduced_patches)
            image_norm_patches = norm_patches.reshape(*(bs,H_toks,W_toks))
            mask = (image_norm_patches > 0.6).astype(np.uint8) 


        # Split & Reshape
        c2 = c[:, 0 : c2.size(1), :]
        c3 = c[:, c2.size(1) : c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1) :, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H_c * 2, W_c * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H_c, W_c).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H_c // 2, W_c // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs

            x1 = F.interpolate(x1, size=(4 * H_c, 4 * W_c), mode="bilinear", align_corners=False)
            x2 = F.interpolate(x2, size=(2 * H_c, 2 * W_c), mode="bilinear", align_corners=False)
            x3 = F.interpolate(x3, size=(1 * H_c, 1 * W_c), mode="bilinear", align_corners=False)
            x4 = F.interpolate(x4, size=(H_c // 2, W_c // 2), mode="bilinear", align_corners=False)
            # print(c1.shape, c2.shape, c3.shape, c4.shape, x1.shape, x2.shape, x3.shape, x4.shape, H_c, H_toks)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        # import pdb;pdb.set_trace()
        # f2 = self.conv_layer1(f2)
        # f3 = self.conv_layer2(f3)
        # f4 = self.conv_layer3(f4)
        # f2 = f2.repeat(1, 2, 1, 1)
        # f3 = f3.repeat(1, 4, 1, 1)
        # f4 = f4.repeat(1, 8, 1, 1)

        elements = [f1, f2, f3, f4]

        result = {f"res{i+2}": element for i, element in enumerate(elements)}

        return result , mask




@BACKBONE_REGISTRY.register()
class D2DINOV2(EdgeEnhancer,Backbone):
    def __init__(self,cfg,input_shape):
        pretrain_size = cfg.MODEL.DINOV2.PRETRAIN_IMG_SIZE
        num_heads = cfg.MODEL.DINOV2.NUM_HEADS
        embed_dim = cfg.MODEL.DINOV2.EMBED_DIM
        patch_size = cfg.MODEL.DINOV2.IN_PATCH_SIZE
        conv_inplane = cfg.MODEL.DINOV2.CONV_INPLANE
        n_points = cfg.MODEL.DINOV2.N_POINTS
        deform_num_heads = cfg.MODEL.DINOV2.DEFORM_NUM_HEADS
        init_values = cfg.MODEL.DINOV2.INIT_VALUES
        interaction_indexes = cfg.MODEL.DINOV2.INTERACTION_INDEXES
        with_cffn = cfg.MODEL.DINOV2.WITH_CFFN
        ffn_type = cfg.MODEL.DINOV2.FFN_TYPE
        depth = cfg.MODEL.DINOV2.DEPTH
        cffn_ratio = cfg.MODEL.DINOV2.CFFN_RATIO
        deform_ratio = cfg.MODEL.DINOV2.DEFORM_RATIO
        add_vit_feature = cfg.MODEL.DINOV2.ADD_VIT_FEATURE
        pretrained = cfg.MODEL.DINOV2.PRETRAINED
        use_extra_extractor = cfg.MODEL.DINOV2.USE_EXTRA_EXTRATOR
        freeze_vit = cfg.MODEL.DINOV2.FREEZE_VIT #False
        use_cls = cfg.MODEL.DINOV2.USE_CLS #True
        with_cp = cfg.MODEL.DINOV2.WITH_CP


        super().__init__(
            pretrain_size = pretrain_size,
            num_heads=num_heads,
            embed_dim = embed_dim,
            patch_size = patch_size,
            
            conv_inplane=conv_inplane,
            n_points=n_points,
            deform_num_heads=deform_num_heads,
            init_values=init_values,

            interaction_indexes = interaction_indexes,
            with_cffn = with_cffn,
            ffn_type = ffn_type,
            depth = depth,
            cffn_ratio=cffn_ratio,
            deform_ratio=deform_ratio,
            add_vit_feature=add_vit_feature,
            pretrained=pretrained,
            use_extra_extractor=use_extra_extractor,
            freeze_vit=freeze_vit,
            use_cls=use_cls,
            with_cp=with_cp)



        self.num_levels = 4

        self.num_feat = [int(embed_dim) for i in range(self.num_levels)]
        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": self.num_feat[0],
            "res3": self.num_feat[1],
            "res4": self.num_feat[2],
            "res5": self.num_feat[3],
        }
        self._out_features = cfg.MODEL.DINOV2.OUT_FEATURES




    def forward(self, x):
        torch.cuda.empty_cache()
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
            x.dim() == 4
        ), f"DiNOv2 takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        y, mask = super().forward(x)
        for k in y.keys():
            if k in self._out_features:
                outputs[k] = y[k]
        return outputs , mask
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32