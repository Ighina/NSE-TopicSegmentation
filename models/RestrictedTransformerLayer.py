# -*- coding: utf-8 -*-
import warnings
from typing import Optional, Tuple, Union, Callable

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn import functional as F
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from transformers import LongformerConfig, LongformerModel, BertConfig, BertModel


class Classic_Transformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, n_layers:int, 
                dim_feedforward: int = 2048, 
                 dropout: float = 0.1,
                 # activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-12,
                 tagset_size: int = 2,
                 device=None, max_position_embedding: int = 512):
        
        super(Classic_Transformer, self).__init__()
        
        self.configuration = BertConfig()
        
        self.configuration.hidden_dropout_prob = dropout
        self.configuration.num_hidden_layers = n_layers
        self.configuration.hidden_size = d_model
        self.configuration.intermediate_size = dim_feedforward
        self.configuration.num_attention_heads = nhead
        self.configuration.num_labels = tagset_size 
        self.configuration.max_position_embeddings = max_position_embedding
        self.configuration.layer_norm_eps = 1e-6 # this is the default parameter in Transformer^2... it can also be set to be the input layer_norm_eps later
        self.configuration.hidden_act = "relu"
        
        self.model = BertModel(self.configuration)
    
    def create_masks_huggingface(self, src, lengths):
        """Create a mask hiding future tokens
        Parameters:
            src (tensor): the source tensor having shape [batch_size, number_of_steps, features_dimensions]
            length (list): a list of integers representing the length (i.e. number_of_steps) of each sample in the batch."""
        mask = []
        
        max_len = src.shape[1]
        for index, i in enumerate(src):
            # The mask consists in tensors having false at the step number that doesn't need to be hidden and true otherwise
            mask.append([1 if (i)<lengths[index] else 0 for i in range(max_len)])
        
        mask = torch.tensor(mask)
        
        return mask
        
    def forward(self, inputs, lengths):
        device = inputs.device
        
        mask = self.create_masks_huggingface(inputs, lengths)
        mask = mask.to(device)
        
        out = self.model(input_ids = None, inputs_embeds = inputs, attention_mask = mask)
        
        return out.last_hidden_state
        
class Causal_Transformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, n_layers:int, 
                dim_feedforward: int = 2048, 
                 dropout: float = 0.1,
                 # activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-12,
                 tagset_size: int = 2,
                 device=None, max_position_embedding: int = 1024, backward = False, window = None):
        
        super(Causal_Transformer, self).__init__()
        
        self.configuration = BertConfig()
        
        self.configuration.hidden_dropout_prob = dropout
        self.configuration.num_hidden_layers = n_layers
        self.configuration.hidden_size = d_model
        self.configuration.intermediate_size = dim_feedforward
        self.configuration.num_attention_heads = nhead
        self.configuration.num_labels = tagset_size 
        self.configuration.max_position_embeddings = max_position_embedding
        self.configuration.layer_norm_eps = 1e-6 # this is the default parameter in Transformer^2... it can also be set to be the input layer_norm_eps later
        self.configuration.hidden_act = "relu"
        
        self.model = BertModel(self.configuration)
        self.backward = backward
        self.window = window
    
    def create_masks_huggingface(self, src, lengths, backward = False):
        """Create a mask hiding future tokens
        Parameters:
            src (tensor): the source tensor having shape [batch_size, number_of_steps, features_dimensions]
            length (list): a list of integers representing the length (i.e. number_of_steps) of each sample in the batch."""
        mask = []

        max_len = src.shape[1]
        window = max_len if self.window is None else self.window
        for index in range(len(src)):
            # The mask consists in tensors having false at the step number that doesn't need to be hidden and true otherwise
            l = lengths[index]
            if not backward:
                # forward transformer
                mask.append([[1 if i-window<x<=i else 0 for x in range(l)]+[0 for x in range(max_len-l)] if (i)<l else [0 for x in range(max_len)] for i in range(max_len)])
            else:
                # backward transformer
                mask.append([[1 if i-window<x<i else 0 for x in reversed(range(l))]+[0 for x in range(max_len-l)] if (i)<l else [0 for x in range(max_len)] for i in reversed(range(max_len))])
        
        mask = torch.tensor(mask)
        
        return mask
        
    def forward(self, inputs, lengths, backward = None):
        device = inputs.device
        
        if backward is None:
            bacward = self.backward
        
        mask = self.create_masks_huggingface(inputs, lengths, backward = False)
        mask = mask.to(device)
        
        out = self.model(input_ids = None, inputs_embeds = inputs, attention_mask = mask)
        
        return out.last_hidden_state

class Longformer_Local_Attention(nn.Module):
    def __init__(self, d_model: int, nhead: int, n_layers:int, 
                dim_feedforward: int = 2048, 
                 window_size: Union[int, list] = [8,4],
                 dropout: float = 0.1,
                 # activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5,
                 tagset_size: int = 2,
                 device=None, max_position_embedding: int = 4096):
        
        super(Longformer_Local_Attention, self).__init__()
        if isinstance(window_size, list):
            assert all([x%2==0 for x in window_size]), "All window sizes must be divisible by 2!"
        else:
            assert window_size%2==0, "Window size must be divisible by 2!"
                     
        self.configuration = LongformerConfig()
        
        self.configuration.attention_window = window_size
        self.configuration.hidden_dropout_prob = dropout
        self.configuration.num_hidden_layers = n_layers
        self.configuration.hidden_size = d_model
        self.configuration.intermediate_size = dim_feedforward
        self.configuration.num_attention_heads = nhead
        self.configuration.num_labels = tagset_size 
        self.configuration.max_position_embeddings = max_position_embedding
        
        self.model = LongformerModel(self.configuration)
    
    def create_masks_huggingface(self, src, lengths):
        """Create a mask hiding future tokens
        Parameters:
            src (tensor): the source tensor having shape [batch_size, number_of_steps, features_dimensions]
            length (list): a list of integers representing the length (i.e. number_of_steps) of each sample in the batch."""
        mask = []
        
        max_len = src.shape[1]
        for index, i in enumerate(src):
            # The mask consists in tensors having false at the step number that doesn't need to be hidden and true otherwise
            mask.append([1 if (i)<lengths[index] else 0 for i in range(max_len)])
        
        mask = torch.tensor(mask)
        local_attention_mask = torch.zeros_like(mask)
            
        return mask, local_attention_mask
        
    def forward(self, inputs, lengths):
        device = inputs.device
        
        mask, local_attention_mask = self.create_masks_huggingface(inputs, lengths)
        mask = mask.to(device)
        local_attention_mask = local_attention_mask.to(device)
        
        out = self.model(input_ids = None, inputs_embeds = inputs, attention_mask = mask, global_attention_mask = local_attention_mask)
        
        return out.last_hidden_state




# BELOW ARE THE LEGACY CLASSES CONSISTING OF MY IMPLEMENTATION OF SLIDING WINDOW ATTENTION. THAT WAS TOO SLOW, SO I RESOLVED TO USE LONGFORMER IMPLEMENTATION BY HUGGINGFACE INsTEAD


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class RestrictedTransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, 
                 window_size: int = None,
                 dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(RestrictedTransformerEncoderLayer, self).__init__()
        self.self_attn = RestrictedMultiheadAttention(d_model, nhead, window_size=window_size, 
                                                      dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(RestrictedTransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.norm_first:
            # x = x + self._sa_block(x, src_mask, src_key_padding_mask)
            # x = x + self._ff_block(x)
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            #x = x + self._sa_block(x, src_mask, src_key_padding_mask)
            #x = x + self._ff_block(x)
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x


    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x, weights = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True)
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class RestrictedMultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces as described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    Multi-Head Attention is defined as:

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0.,  window_size = None, bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(RestrictedMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.window = window_size

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(RestrictedMultiheadAttention, self).__setstate__(state)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
            or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
            :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
            Queries are compared against key-value pairs to produce the output.
            See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
            or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
            :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
            See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
            ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
            sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
            See "Attention Is All You Need" for more details.
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
            Binary and byte masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a byte mask, a non-zero value indicates that the corresponding ``key``
            value will be ignored.
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Default: ``True``.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
            heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
            effect when ``need_weights=True.``. Default: True (i.e. average weights across heads)

    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
          :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
          where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
          embedding dimension ``embed_dim``.
        - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
          returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.

        .. note::
            `batch_first` argument is ignored for unbatched inputs.
        """
        
        
        key_padding_mask = None # TODO: find a way in which we can still use a mask but without all the nan problems I was having
        
        is_batched = query.dim() == 3
        assert is_batched, "Restricted attention for non-batched inputs not implemented yet"
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        
        L = query.shape[0]
        # print(L)
            
        if self.window is None:
            if not self._qkv_same_embed_dim:
                attn_output, attn_output_weights = F.multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads,
                    self.in_proj_weight, self.in_proj_bias,
                    self.bias_k, self.bias_v, self.add_zero_attn,
                    self.dropout, self.out_proj.weight, self.out_proj.bias,
                    training=self.training,
                    key_padding_mask=key_padding_mask, need_weights=need_weights,
                    attn_mask=attn_mask, use_separate_proj_weight=True,
                    q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                    v_proj_weight=self.v_proj_weight) 
                    # average_attn_weights=average_attn_weights)
            else:
                attn_output, attn_output_weights = F.multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads,
                    self.in_proj_weight, self.in_proj_bias,
                    self.bias_k, self.bias_v, self.add_zero_attn,
                    self.dropout, self.out_proj.weight, self.out_proj.bias,
                    training=self.training,
                    key_padding_mask=key_padding_mask, need_weights=need_weights,
                    attn_mask=attn_mask) 
                    # average_attn_weights=average_attn_weights)
            if self.batch_first and is_batched:
                return attn_output.transpose(1, 0), attn_output_weights
            else:
                return attn_output, attn_output_weights
        else:
            attention_weights = []
            attention_outputs = []
            

            for i in range(L):
                # print(i)
                
                if i<self.window:
                    # print(i)
                    if key_padding_mask is not None:
                        new_pad = key_padding_mask[:, :i+self.window+1]
                    else:
                        new_pad = None
                    if attn_mask is not None:
                        new_attn = attn_mask[:i+self.window+1, :i+self.window+1]
                    else:
                        new_attn = None
                        
                    if not self._qkv_same_embed_dim:
                        attn_output, attn_output_weights = F.multi_head_attention_forward(
                            query[:i+self.window+1], 
                            key[:i+self.window+1], 
                            value[:i+self.window+1], 
                            self.embed_dim, self.num_heads,
                            self.in_proj_weight, self.in_proj_bias,
                            self.bias_k, self.bias_v, self.add_zero_attn,
                            self.dropout, self.out_proj.weight, self.out_proj.bias,
                            training=self.training,
                            key_padding_mask=new_pad, need_weights=need_weights,
                            attn_mask=new_attn, use_separate_proj_weight=True,
                            q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                            v_proj_weight=self.v_proj_weight) 
                            # average_attn_weights=average_attn_weights)
                    else:
                        attn_output, attn_output_weights = F.multi_head_attention_forward(
                            query[:i+self.window+1], 
                            key[:i+self.window+1], 
                            value[:i+self.window+1], 
                            self.embed_dim, self.num_heads,
                            self.in_proj_weight, self.in_proj_bias,
                            self.bias_k, self.bias_v, self.add_zero_attn,
                            self.dropout, self.out_proj.weight, self.out_proj.bias,
                            training=self.training,
                            key_padding_mask=new_pad, need_weights=need_weights,
                            attn_mask=new_attn) 
                            # average_attn_weights=average_attn_weights)
                        
                    attention_outputs.append(attn_output[i])
                    attention_weights.append(attn_output_weights[:,i])
                elif L<self.window+i:
                    if key_padding_mask is not None:
                        new_pad = key_padding_mask[:, i-self.window:]
                    else:
                        new_pad = None
                    if attn_mask is not None:
                        new_attn = attn_mask[i-self.window:, i-self.window:]
                    else:
                        new_attn = None
                        
                    if not self._qkv_same_embed_dim:
                        attn_output, attn_output_weights = F.multi_head_attention_forward(
                            query[i-self.window:], 
                            key[i-self.window:], 
                            value[i-self.window:], 
                            self.embed_dim, self.num_heads,
                            self.in_proj_weight, self.in_proj_bias,
                            self.bias_k, self.bias_v, self.add_zero_attn,
                            self.dropout, self.out_proj.weight, self.out_proj.bias,
                            training=self.training,
                            key_padding_mask=new_pad, need_weights=need_weights,
                            attn_mask=new_attn, use_separate_proj_weight=True,
                            q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                            v_proj_weight=self.v_proj_weight) 
                            # average_attn_weights=average_attn_weights)
                    else:
                        attn_output, attn_output_weights = F.multi_head_attention_forward(
                            query[i-self.window:], 
                            key[i-self.window:], 
                            value[i-self.window:], 
                            self.embed_dim, self.num_heads,
                            self.in_proj_weight, self.in_proj_bias,
                            self.bias_k, self.bias_v, self.add_zero_attn,
                            self.dropout, self.out_proj.weight, self.out_proj.bias,
                            training=self.training,
                            key_padding_mask=new_pad, need_weights=need_weights,
                            attn_mask=new_attn) 
                            # average_attn_weights=average_attn_weights)
                        
                    attention_outputs.append(attn_output[self.window])
                    attention_weights.append(attn_output_weights[:,self.window])
                    
                else:
                    if key_padding_mask is not None:
                        new_pad = key_padding_mask[:, i-self.window:i+self.window+1]
                    else:
                        new_pad = None
                    if attn_mask is not None:
                        new_attn = attn_mask[i-self.window:i+self.window+1, i-self.window:i+self.window+1]
                    else:
                        new_attn = None
                        
                    if not self._qkv_same_embed_dim:
                        attn_output, attn_output_weights = F.multi_head_attention_forward(
                            query[i-self.window:i+self.window+1], 
                            key[i-self.window:i+self.window+1], 
                            value[i-self.window:i+self.window+1], 
                            self.embed_dim, self.num_heads,
                            self.in_proj_weight, self.in_proj_bias,
                            self.bias_k, self.bias_v, self.add_zero_attn,
                            self.dropout, self.out_proj.weight, self.out_proj.bias,
                            training=self.training,
                            key_padding_mask=new_pad, need_weights=need_weights,
                            attn_mask=new_attn, use_separate_proj_weight=True,
                            q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                            v_proj_weight=self.v_proj_weight) 
                            # average_attn_weights=average_attn_weights)
                    else:
                        attn_output, attn_output_weights = F.multi_head_attention_forward(
                            query[i-self.window:i+self.window+1], 
                            key[i-self.window:i+self.window+1], 
                            value[i-self.window:i+self.window+1], 
                            self.embed_dim, self.num_heads,
                            self.in_proj_weight, self.in_proj_bias,
                            self.bias_k, self.bias_v, self.add_zero_attn,
                            self.dropout, self.out_proj.weight, self.out_proj.bias,
                            training=self.training,
                            key_padding_mask=new_pad, need_weights=need_weights,
                            attn_mask=new_attn) 
                            # average_attn_weights=average_attn_weights)
                        
                    attention_outputs.append(attn_output[self.window])
                    attention_weights.append(attn_output_weights[:,self.window])
                
            attn_output = torch.stack(attention_outputs, dim=0)
            # attn_output[torch.isnan(attn_output)] = 0
            # return attention_outputs
            if self.batch_first and is_batched:
                return attn_output.transpose(1, 0), attention_weights
            else:
                return attn_output, attention_weights
                    

class PyramidalTransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``False`` (disabled).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layers, norm=None, enable_nested_tensor=False):
        super(PyramidalTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(encoder_layers)
        self.num_layers = len(encoder_layers)
        self.norm = norm
        self.enable_nested_tensor = enable_nested_tensor

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        #if isinstance(first_layer, torch.nn.TransformerEncoderLayer):
        if True: # TODO: change with condition to include the restricted attention layer
            if (not first_layer.norm_first and not first_layer.training and
                    first_layer.self_attn.batch_first and
                    first_layer.self_attn._qkv_same_embed_dim and first_layer.activation_relu_or_gelu and
                    first_layer.norm1.eps == first_layer.norm2.eps and
                    src.dim() == 3 and self.enable_nested_tensor) :
                if src_key_padding_mask is not None and not output.is_nested and mask is None:
                    tensor_args = (
                        src,
                        first_layer.self_attn.in_proj_weight,
                        first_layer.self_attn.in_proj_bias,
                        first_layer.self_attn.out_proj.weight,
                        first_layer.self_attn.out_proj.bias,
                        first_layer.norm1.weight,
                        first_layer.norm1.bias,
                        first_layer.norm2.weight,
                        first_layer.norm2.bias,
                        first_layer.linear1.weight,
                        first_layer.linear1.bias,
                        first_layer.linear2.weight,
                        first_layer.linear2.bias,
                    )
                    if not torch.overrides.has_torch_function(tensor_args):
                        if not torch.is_grad_enabled() or all([not x.requires_grad for x in tensor_args]):
                            if output.is_cuda or 'cpu' in str(output.device):
                                convert_to_nested = True
                                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not())

        for mod in self.layers:
            if convert_to_nested:
                output = mod(output, src_mask=mask)
            else:
                output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if convert_to_nested:
            output = output.to_padded_tensor(0.)

        if self.norm is not None:
            output = self.norm(output)

        return output