from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.models.embeddings import (
    GaussianFourierProjection,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import (
    AdaLayerNormContinuous,
    FP32LayerNorm,
    LayerNorm,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import maybe_allow_in_graph
from torch import nn


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@maybe_allow_in_graph
class DiTBlock(nn.Module):
    r"""
    Transformer block used in Hunyuan-DiT model (https://github.com/Tencent/HunyuanDiT). Allow skip connection and
    QKNorm

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of headsto use for multi-head attention.
        cross_attention_dim (`int`,*optional*):
            The size of the encoder_hidden_states vector for cross attention.
        dropout(`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        activation_fn (`str`,*optional*, defaults to `"geglu"`):
            Activation function to be used in feed-forward. .
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, *optional*, defaults to 1e-6):
            A small constant added to the denominator in normalization layers to prevent division by zero.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*):
            The size of the hidden layer in the feed-forward block. Defaults to `None`.
        ff_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the feed-forward block.
        skip (`bool`, *optional*, defaults to `False`):
            Whether to use skip connection. Defaults to `False` for down-blocks and mid-blocks.
        qk_norm (`bool`, *optional*, defaults to `True`):
            Whether to use normalization in QK calculation. Defaults to `True`.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        use_self_attention: bool = True,
        use_cross_attention: bool = False,
        self_attention_norm_type: Optional[str] = None,  # ada layer norm
        cross_attention_dim: Optional[int] = None,
        cross_attention_norm_type: Optional[str] = "fp32_layer_norm",
        use_cross_attention_2: bool = False,
        cross_attention_2_dim: Optional[int] = None,
        cross_attention_2_norm_type: Optional[str] = None,
        dropout=0.0,
        activation_fn: str = "gelu",
        norm_type: str = "fp32_layer_norm",  # TODO
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        ff_inner_dim: Optional[int] = None,  # int(dim * 4) if None
        ff_bias: bool = True,
        skip: bool = False,
        skip_concat_front: bool = False,  # [x, skip] or [skip, x]
        skip_norm_last: bool = False,  # this is an error
        qk_norm: bool = True,
        qkv_bias: bool = True,
    ):
        super().__init__()

        self.use_self_attention = use_self_attention    # True
        self.use_cross_attention = use_cross_attention  # True
        self.use_cross_attention_2 = use_cross_attention_2  # True
        self.skip_concat_front = skip_concat_front          # True
        self.skip_norm_last = skip_norm_last
        self.dim = dim                                      # 2048
        self.num_attention_heads = num_attention_heads      # 16
        self.norm_eps = norm_eps                            # 1e-5
        self.norm_elementwise_affine = norm_elementwise_affine  # True
        self.qk_norm = qk_norm                                  # True
        self.qkv_bias = qkv_bias                                # False
        self.gradient_checkpointing = False
        # Define 3 blocks. Each block has its own normalization layer.
        # NOTE: when new version comes, check norm2 and norm 3
        # 1. Self-Attn
        if use_self_attention:
            if (
                self_attention_norm_type == "fp32_layer_norm"
                or self_attention_norm_type is None
            ):
                self.norm1 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)
            else:
                raise NotImplementedError

            self.attn1 = Attention(
                query_dim=dim,
                cross_attention_dim=None,
                dim_head=dim // num_attention_heads,
                heads=num_attention_heads,
                qk_norm="rms_norm" if qk_norm else None,
                eps=1e-6,
                bias=qkv_bias,
                processor=TripoSGAttnProcessor2_0(),
            )

        # 2. Cross-Attn
        if use_cross_attention:
            assert cross_attention_dim is not None

            self.norm2 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)

            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                dim_head=dim // num_attention_heads,
                heads=num_attention_heads,
                qk_norm="rms_norm" if qk_norm else None,
                cross_attention_norm=cross_attention_norm_type,
                eps=1e-6,
                bias=qkv_bias,
                processor=TripoSGAttnProcessor2_0(),
            )

        # 2'. Second Cross-Attn
        if use_cross_attention_2:
            assert cross_attention_2_dim is not None

            self.norm2_2 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)

            self.attn2_2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_2_dim,
                dim_head=dim // num_attention_heads,
                heads=num_attention_heads,
                qk_norm="rms_norm" if qk_norm else None,
                cross_attention_norm=cross_attention_2_norm_type,
                eps=1e-6,
                bias=qkv_bias,
                processor=TripoSGAttnProcessor2_0(),
            )

        self.additional_cross_attention = nn.ModuleDict()
        self.additional_norm = nn.ModuleDict()

        # 3. Feed-forward
        self.norm3 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)

        self.ff = FeedForward(
            dim,
            dropout=dropout,  ### 0.0
            activation_fn=activation_fn,  ### approx GeLU
            final_dropout=final_dropout,  ### 0.0
            inner_dim=ff_inner_dim,  ### int(dim * mlp_ratio)
            bias=ff_bias,
        )

        # 4. Skip Connection
        if skip:
            self.skip_norm = FP32LayerNorm(dim, norm_eps, elementwise_affine=True)
            self.skip_linear = nn.Linear(2 * dim, dim)
        else:
            self.skip_linear = None

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_topk(self, topk):
        self.flash_processor.topk = topk

    def set_flash_processor(self, flash_processor):
        self.flash_processor = flash_processor
        self.attn2.processor = self.flash_processor

    def update_cross_attention(self, name, cross_attention_dim, cross_attention_norm_type = None):
        assert cross_attention_dim is not None
        norm = FP32LayerNorm(self.dim, self.norm_eps, self.norm_elementwise_affine)
        attn = Attention(
                query_dim=self.dim,
                cross_attention_dim=cross_attention_dim,
                dim_head=self.dim // self.num_attention_heads,
                heads=self.num_attention_heads,
                qk_norm="rms_norm" if self.qk_norm else None,
                cross_attention_norm=cross_attention_norm_type,
                eps=1e-6,
                bias=self.qkv_bias,
                processor=TripoSGAttnProcessor2_0(),
            )
        
        if name == "attn2":
            self.norm2 = norm
            self.attn2 = attn
        elif name == "attn2_2":
            self.norm2_2 = norm
            self.attn2_2 = attn
        else:
            raise NotImplementedError

    def add_additional_cross_attention(self, name, cross_attention_dim, cross_attention_norm_type=None):
        
        assert cross_attention_dim is not None
        norm = FP32LayerNorm(self.dim, self.norm_eps, self.norm_elementwise_affine)

        attn = Attention(
            query_dim=self.dim,
            cross_attention_dim=cross_attention_dim,
            dim_head=self.dim // self.num_attention_heads,
            heads=self.num_attention_heads,
            qk_norm="rms_norm" if self.qk_norm else None,
            cross_attention_norm=cross_attention_norm_type,
            eps=1e-6,
            bias=self.qkv_bias,
            processor=TripoSGAttnProcessor2_0(),
        )

        self.additional_cross_attention[name] = attn
        self.additional_norm[name] = norm

    # Copied from diffusers.models.attention.BasicTransformerBlock.set_chunk_feed_forward
    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def _set_gradient_checkpointing(self, module, value: bool = False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def _forward_cross_attention(self, 
        hidden_states, 
        cross_attentions: List[Dict[str, Any]], 
        image_rotary_emb: Optional[torch.Tensor] = None, 
        attention_kwargs: Optional[Dict[str, Any]] = None, 
    ):
        attention_kwargs = attention_kwargs or {}

        for cross_attention in cross_attentions:
            attn_hidden_states = cross_attention["weight"] * cross_attention["module"](
                cross_attention["norm"](hidden_states), 
                cross_attention["encoder_hidden_states"], 
                image_rotary_emb=image_rotary_emb, 
                **attention_kwargs
            )
            hidden_states = hidden_states + attn_hidden_states

        return hidden_states
    
    def _build_cross_attenion_kwargs(self, name, encoder_hidden_states, weight=1.0):
        if name == "attn2":
            kwargs = {
                "module": self.attn2,
                "weight": weight,
                "norm": self.norm2,
                
            }
        elif name == "attn2_2":
            kwargs = {
                "module": self.attn2_2,
                "weight": weight,
                "norm": self.norm2_2,
            }
        else:
            kwargs = {
                "module": self.additional_cross_attention[name],
                "weight": weight,
                "norm": self.additional_norm[name],
            }

        kwargs["encoder_hidden_states"] = encoder_hidden_states
        kwargs["name"] = name
        return kwargs

    def _forward(
        self,
        hidden_states: torch.Tensor,        # [B, N + 1, 2048] latent tokens + time
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_2: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        skip: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        additional_cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        # Prepare attention kwargs
        attention_kwargs = attention_kwargs or {}

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Long Skip Connection
        if self.skip_linear is not None:
            cat = torch.cat(
                (
                    [skip, hidden_states]
                    if self.skip_concat_front
                    else [hidden_states, skip]
                ),
                dim=-1,
            )
            if self.skip_norm_last:
                # don't do this
                hidden_states = self.skip_linear(cat)
                hidden_states = self.skip_norm(hidden_states)
            else:
                cat = self.skip_norm(cat)
                hidden_states = self.skip_linear(cat)

        # 1. Self-Attention
        if self.use_self_attention:
            norm_hidden_states = self.norm1(hidden_states)
            attn_output = self.attn1(
                norm_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **attention_kwargs,
            )
            hidden_states = hidden_states + attn_output

        # 2. Cross-Attention 
        # a. cross attention with part embedding: encoder hidden states --> part embedding
        # b. cross attention with whole embedding: encoder hidden states 2 --> whole embedding
        hidden_states = self._forward_cross_attention(
            hidden_states,
            (
                [
                    self._build_cross_attenion_kwargs("attn2", encoder_hidden_states)
                ] if self.use_cross_attention else []
            ) + (
                [
                    self._build_cross_attenion_kwargs("attn2_2", encoder_hidden_states_2)  
                ] if self.use_cross_attention_2 else []
            ),
            image_rotary_emb=image_rotary_emb,
            attention_kwargs=attention_kwargs,
        )

        if additional_cross_attention_kwargs is not None:
            hidden_states = self._forward_cross_attention(
                hidden_states,
                [
                    self._build_cross_attenion_kwargs(name, value["encoder_hidden_states"], value["weight"])
                    for name, value in additional_cross_attention_kwargs.items()
                ],
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
            )
            
        # FFN Layer ### TODO: switch norm2 and norm3 in the state dict
        mlp_inputs = self.norm3(hidden_states)
        hidden_states = hidden_states + self.ff(mlp_inputs)

        return hidden_states
    
    def forward(self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_2: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        skip: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        additional_cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        if self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(
                self._forward,
                hidden_states,
                encoder_hidden_states,
                encoder_hidden_states_2,
                temb,
                image_rotary_emb,
                skip,
                attention_kwargs,
                additional_cross_attention_kwargs,
                use_reentrant=False
            )
        else:
            return self._forward(
                hidden_states,
                encoder_hidden_states,
                encoder_hidden_states_2,
                temb,
                image_rotary_emb,
                skip,
                attention_kwargs,
                additional_cross_attention_kwargs,
            )
