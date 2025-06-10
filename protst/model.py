import os
import warnings

import torch
from torch import nn
from torch.nn import functional as F

from torchdrug import core, layers, models, utils
from torchdrug.layers import functional
from torchdrug.core import Registry as R 

# from transformers import BertFor MaskedLM, BertTokenizer, BertModel, pipeline
from torch_scatter import scatter_add

from torchdrug.layers.functional import variadic_to_padded
from protst import layer, data

import esm
import sys
sys.path.append('..')
from models.protein_bert_seq import SEQ_BERT, SEQ_Wrapper

@R.register("models.PretrainESM")
class PretrainEvolutionaryScaleModeling(models.ESM):
    """
    Enable to pretrain ESM with MLM.
    """

    def __init__(self, path, model="ESM-1b", 
        output_dim=512, num_mlp_layer=2, activation='relu', 
        readout="mean", mask_modeling=False, use_proj=True):
        super(PretrainEvolutionaryScaleModeling, self).__init__(path, model, readout)
        self.mask_modeling = mask_modeling

        self.last_hidden_dim = self.output_dim
        self.output_dim = output_dim if use_proj else self.last_hidden_dim
        self.num_mlp_layer = num_mlp_layer
        self.activation = activation
        self.use_proj = use_proj

        self.graph_mlp = layers.MLP(self.last_hidden_dim,
                                    [self.last_hidden_dim] * (num_mlp_layer - 1) + [output_dim],
                                    activation=self.activation)
        self.residue_mlp = layers.MLP(self.last_hidden_dim,
                                      [self.last_hidden_dim] * (num_mlp_layer - 1) + [output_dim],
                                      activation=self.activation)
        if model == "ESM-2-8M":
            self.esm_layer = 6
        elif model == "ESM-2-35M":
            self.esm_layer = 12
        elif model == "ESM-2-650M":
            self.esm_layer = 33

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the residue representations and the graph representation(s).
        """
        input = graph.residue_type
        if self.mask_modeling:
            non_mask = ~(input == self.alphabet.mask_idx)
            input[non_mask] = self.mapping[input[non_mask]]
        else:
            input = self.mapping[input]
        size = graph.num_residues
        if (size > self.max_input_length).any():
            warnings.warn("ESM can only encode proteins within %d residues. Truncate the input to fit into ESM."
                          % self.max_input_length)
            starts = size.cumsum(0) - size
            size = size.clamp(max=self.max_input_length)
            ends = starts + size
            mask = functional.multi_slice_mask(starts, ends, graph.num_residue)
            input = input[mask]
            graph = graph.subresidue(mask)
        size_ext = size
        if self.alphabet.prepend_bos:
            bos = torch.ones(graph.batch_size, dtype=torch.long, device=self.device) * self.alphabet.cls_idx
            input, size_ext = functional._extend(bos, torch.ones_like(size_ext), input, size_ext)
        if self.alphabet.append_eos:
            eos = torch.ones(graph.batch_size, dtype=torch.long, device=self.device) * self.alphabet.eos_idx
            input, size_ext = functional._extend(input, size_ext, eos, torch.ones_like(size_ext))
        input = functional.variadic_to_padded(input, size_ext, value=self.alphabet.padding_idx)[0]

        output = self.model(input, repr_layers=[self.esm_layer])  # esm-8m: 6; esm-35m: 12; esm-150m: 30; 650m: 33
        residue_feature = output["representations"][self.esm_layer]

        residue_feature = functional.padded_to_variadic(residue_feature, size_ext)
        starts = size_ext.cumsum(0) - size_ext
        if self.alphabet.prepend_bos:
            starts = starts + 1
        ends = starts + size
        mask = functional.multi_slice_mask(starts, ends, len(residue_feature))
        residue_feature = residue_feature[mask]
        graph_feature = self.readout(graph, residue_feature)

        if self.use_proj:
            graph_feature = self.graph_mlp(graph_feature)
            residue_feature = self.residue_mlp(residue_feature)

        return {
            "graph_feature": graph_feature,
            "residue_feature": residue_feature, 
        }

@R.register("models.ProtAC")
class ProtAC(nn.Module, core.Configurable):
    """
    Enable to pretrain ESM with MLM.
    """
    last_hidden_dim = 128
    max_input_length = 512
    def __init__(self, path, model="PB_tiny", 
        output_dim=512, num_mlp_layer=2, activation='relu', 
        readout="mean", mask_modeling=False, use_proj=True):
        super(ProtAC, self).__init__()
        self.model = model
        self.mask_modeling = mask_modeling
        self.seq_encoder = self.load_weight()
        self.seq_globatten = nn.Sequential(nn.Linear(128, 1), nn.Softmax(dim=1))
        # self.proj = nn.Linear(128, 256)
        self.layer_norm = nn.LayerNorm(256)
        self.dropout = nn.Dropout(0.1)
        self.tokenizer = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                          'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                          'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                          'N': 2, 'Y': 18, 'M': 12, '-': 20, 'X': 21, 'B': 22,
                          'Z': 23, 'O': 24, 'U': 25}
        self.pad_token_id = 20
        mapping = self.construct_mapping(self.tokenizer)
        self.register_buffer("mapping", mapping)
        self.output_dim = output_dim if use_proj else self.last_hidden_dim
        self.num_mlp_layer = num_mlp_layer
        self.activation = activation
        self.use_proj = use_proj

        self.graph_mlp = layers.MLP(self.last_hidden_dim,
                                    [self.last_hidden_dim] * (num_mlp_layer - 1) + [output_dim],
                                    activation=self.activation)
        self.residue_mlp = layers.MLP(self.last_hidden_dim,
                                      [self.last_hidden_dim] * (num_mlp_layer - 1) + [output_dim],
                                      activation=self.activation)
        if readout == "sum":
            self.readout = layers.SumReadout("residue")
        elif readout == "mean":
            self.readout = layers.MeanReadout("residue")
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def load_weight(self):
        protein_bert = SEQ_BERT(
            num_tokens=26,
            dim=128,
            dim_global=512,
            depth=6,
            narrow_conv_kernel=9,
            wide_conv_kernel=9,
            wide_conv_dilation=5,
            attn_heads=4,
            attn_dim_head=64,
            local_self_attn=True,
            glu_conv=False,
        )
        seq_encoder = SEQ_Wrapper(
            protein_bert,
            model_type="seq_bert_itc",
            args=None
        )
        return seq_encoder

    def construct_mapping(self, alphabet):
        mapping = [0] * len(data.Protein.id2residue_symbol)
        for i, token in data.Protein.id2residue_symbol.items():
            mapping[i] = alphabet[token]
        mapping = torch.tensor(mapping)
        return mapping

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the residue representations and the graph representation(s).
        """
        input = graph.residue_type
        if self.mask_modeling:
            non_mask = ~(input == self.pad_token_id)
            input[non_mask] = self.mapping[input[non_mask]]
        else:
            input = self.mapping[input]

        size = graph.num_residues
        # print("size: ", size)
        if (size > self.max_input_length).any():
            starts = size.cumsum(0) - size
            size = size.clamp(max=self.max_input_length)
            ends = starts + size
            mask = functional.multi_slice_mask(starts, ends, graph.num_residue)
            input = input[mask]
            graph = graph.subresidue(mask)
        size_ext = size
        # bos = torch.ones(graph.batch_size, dtype=torch.long, device=self.device) * self.tokenizer.cls_token_id
        # input, size_ext = functional._extend(bos, torch.ones_like(size_ext), input, size_ext)
        # eos = torch.ones(graph.batch_size, dtype=torch.long, device=self.device) * self.tokenizer.sep_token_id
        # input, size_ext = functional._extend(input, size_ext, eos, torch.ones_like(size_ext))
        input = functional.variadic_to_padded(input, size_ext, value=self.pad_token_id)[0]
        # print(input)
        # print("input shape: ", input.shape)
        # sys.exit()
        residue_feature = self.seq_encoder(input)
        # seq_glob_atten = self.seq_globatten(residue_feature.float())  ## (batch_size, seq_len, 1)
        # seq_g = torch.bmm(seq_glob_atten.transpose(-1, 1),
        #                 residue_feature).squeeze()  ## (batch_size, seq_embed_dim=512)
        # graph_feature = F.normalize(seq_g, dim=-1) ## (batch_size, blip_embed_dim=256)

        residue_feature = functional.padded_to_variadic(residue_feature, size_ext)
        starts = size_ext.cumsum(0) - size_ext
        # starts = starts + 1
        ends = starts + size
        mask = functional.multi_slice_mask(starts, ends, len(residue_feature))
        residue_feature = residue_feature[mask]
        if self.readout:
            graph_feature = self.readout(graph, residue_feature)

        if self.use_proj:
            graph_feature = self.graph_mlp(graph_feature)
            residue_feature = self.residue_mlp(residue_feature)
        # print(graph_feature.shape, residue_feature.shape)
        # sys.exit()
        return {
            "graph_feature": graph_feature,
            "residue_feature": residue_feature
        }
