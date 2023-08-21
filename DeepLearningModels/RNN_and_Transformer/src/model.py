from cmath import sqrt
from colorsys import hls_to_rgb
import enum
from heapq import nlargest
from tokenize import Double
from turtle import forward
from xml.sax.xmlreader import InputSource
from importlib_metadata import requires
from pkg_resources import require
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np


def AttnHook(self, input, output):
    global attn
    _, attn = output


class LMModel(nn.Module):
    def __init__(self, nvoc, ninput, nhid, nlayers):
        super(LMModel, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.encoder = nn.Embedding(nvoc, ninput)
        self.model = LSTM(ninput, nhid, nlayers)
        self.decoder = nn.Linear(nhid, nvoc)
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input):
        embeddings = self.drop(self.encoder(input))
        output = self.model(embeddings)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1))


class LSTMCell(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.vec_sz = input_sz +  hidden_sz * 2
        self.forget_gate = nn.Linear(in_features=self.vec_sz, out_features=self.hidden_size)
        self.input_gate = nn.Linear(in_features=self.vec_sz, out_features=self.hidden_size)
        self.output_gate = nn.Linear(in_features=self.vec_sz, out_features=self.hidden_size)
        self.gate_gate = nn.Linear(in_features=self.vec_sz, out_features=self.hidden_size)
        self.linear_ci = Parameter(torch.randn(self.hidden_size), requires_grad = True)
        self.linear_cf = Parameter(torch.randn(self.hidden_size), requires_grad = True)
        self.linear_co = Parameter(torch.randn(self.hidden_size), requires_grad = True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def forward(self, x, h_ln, h_lt, c):
        input_vec = torch.cat([x, h_ln, h_lt], dim = -1)
        c = c.clone()
        i_t = self.sigmoid(self.input_gate(input_vec) + torch.mul(self.linear_ci, c))
        f_t = self.sigmoid(self.forget_gate(input_vec) + torch.mul(self.linear_cf, c))    
        c_t = i_t * self.tanh(self.gate_gate(input_vec)) + f_t * c
        o_t = self.sigmoid(self.output_gate(input_vec) + self.linear_co * c)
        h_t = o_t * self.tanh(c_t)
        return h_t, c_t


class LSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, layers = 3):
        super().__init__()
        self.input_size = input_sz 
        self.hidden_sz = hidden_sz
        self.lstm_layer1 = LSTMCell(input_sz, hidden_sz)
        self.lstm_layer2 = LSTMCell(input_sz, hidden_sz)
        self.lstm_layer3 = LSTMCell(input_sz, hidden_sz)
        self.num_layers = layers
        self.linear_hy = nn.Linear(hidden_sz * self.num_layers, hidden_sz)
        self.lstm_layers = torch.nn.ModuleList([LSTMCell(input_sz, hidden_sz) for i in range(self.num_layers)])

    def forward(self, x):
        sql_len, batch_sz, vec_len  = x.shape
        y = torch.zeros(sql_len, batch_sz, self.hidden_sz).to(x.device)
        inputs = torch.chunk(x, chunks=sql_len)
        last_hn1 = torch.zeros(batch_sz, self.hidden_sz).to(x.device)
        hns = [torch.zeros_like(last_hn1) for i in range(self.num_layers)] 
        cns = [torch.zeros_like(last_hn1) for i in range(self.num_layers)] 

        for i, input in enumerate(inputs):
            input = torch.squeeze(input)
            for j, lstm_layer in enumerate(self.lstm_layers):
                if j == 0:
                    hns[j][:], cns[j][:] = lstm_layer(input, last_hn1, hns[j], cns[j])
                else:
                    hns[j][:], cns[j][:] = lstm_layer(input, hns[j-1], hns[j], cns[j])

            y[i] = self.linear_hy(torch.cat(hns, dim = -1))
        return y   


class Transformer(nn.Module):
    def __init__(self, sql_len, nvoc, en_nlayers = 2, de_nlayers = 3, masked = True, n_head = 8, d_model = 256, d_ffn = 1024, layer_norm = True):
        super().__init__()
        self.sql_len, self.nvoc, self.en_nlayers, self.de_nlayers, self.masked, self.n_head, self.d_model, self.d_ffn, self.layer_norm = \
            sql_len, nvoc, en_nlayers, de_nlayers, masked, n_head, d_model, d_ffn, layer_norm
        self.encoder = TransformerEncoder(sql_len, nvoc, en_nlayers, n_head, d_model, d_ffn, layer_norm)
        self.decoder = TransformerDecoder(sql_len, nvoc, de_nlayers, masked, n_head, d_model, d_ffn, layer_norm)
        self.embedding = nn.Embedding(self.nvoc, self.d_model)
        self.y_linear = nn.Linear(self.d_model, self.nvoc)
        
    def forward(self, src, tgt):
        tgt = torch.transpose(tgt, 0, 1).contiguous() 
        src_e = self.embedding(src)
        tgt_e = self.embedding(tgt)
        kv = self.encoder(src_e)
        y = self.decoder(kv, tgt_e)
        y = self.y_linear(y)
        y = torch.transpose(y, 0, 1).contiguous()
        return y


class TF_OnlyEncoder(nn.Module):
    def __init__(self, sql_len, nvoc, nlayers = 6, masked = True, n_head = 8, d_model = 512, d_ffn = 2048, layer_norm = True):
        super().__init__()
        self.sql_len, self.nvoc, self.nlayers, self.masked, self.n_head, self.d_model, self.d_ffn, self.layer_norm = \
            sql_len, nvoc, nlayers, masked, n_head, d_model, d_ffn, layer_norm
        self.encoder = TransformerEncoder(sql_len, nvoc, self.nlayers, n_head, d_model, d_ffn, layer_norm, self.masked)
        self.embedding = nn.Embedding(self.nvoc, self.d_model)
        self.y_linear = nn.Linear(self.d_model, self.nvoc)
        
    def forward(self, src):
        x = torch.transpose(src, 0, 1).contiguous() 
        x = self.embedding(x)
        y = self.encoder(x)
        y = self.y_linear(y)
        y = torch.transpose(y, 0, 1).contiguous()
        return y


class TransformerBlock(nn.Module):
    def __init__(self, num_head = 8, masked = False, d_model = 512) -> None:
        super().__init__()
        self.num_head, self.masked, self.d_model = num_head, masked, d_model
        assert (self.d_model % self.num_head == 0)

        self.head_dim = self.d_model // self.num_head
        self.q_linear = nn.Linear(self.d_model, self.d_model)
        self.k_linear = nn.Linear(self.d_model, self.d_model)
        self.v_linear = nn.Linear(self.d_model, self.d_model)
        self.attn_linear = nn.Linear(self.d_model, self.d_model)
        self.softmax = nn.Softmax(dim = -1)
        self.mask = None

    def forward(self, q, k, v):
        q_bz, q_sl, q_c = q.shape
        kv_bz, kv_sl, kv_c = v.shape
        q = self.q_linear(q).view(q_bz, q_sl, self.num_head, -1)
        k = self.k_linear(k).view(kv_bz, kv_sl, self.num_head, -1)
        v = self.v_linear(v).view(kv_bz, kv_sl, self.num_head, -1)
        q = torch.transpose(q, -2, -3).contiguous()
        k = torch.transpose(k, -2, -3).contiguous()
        v = torch.transpose(v, -2, -3).contiguous()

        qk_weight = torch.matmul(q, torch.transpose(k, -1, -2))
        qk_weight = qk_weight / np.sqrt(self.head_dim)
        if self.masked:
            if self.mask is None:
                # print('Init mask!')
                self.mask = torch.tril(torch.ones(q_sl, q_sl, dtype = torch.double))
                self.mask = torch.where(self.mask > 0, 0., -float('inf')).float()\
                                .to(q.device)
            qk_weight = qk_weight + self.mask[:q_sl, :q_sl]

        qk_weight = self.softmax(qk_weight)
        attn = torch.matmul(qk_weight, v).transpose(-2, -3).contiguous()
        attn = attn.view(q_bz, q_sl, -1)
        attn = self.attn_linear(attn)
        return attn, qk_weight


class TransformerEncoder(nn.Module):
    def __init__(self, sql_len, n_voc, n_layers = 12, n_head = 8, d_model = 512, d_ffn = 2048, layer_norm = True, masked = False):
        super().__init__()
        self.n_layers, self.n_voc, self.n_head, self.d_model, self.d_ffn = \
            n_layers, n_voc,  n_head, d_model, d_ffn
        self.layer_norm = layer_norm
        self.masked = masked
        self.weight_mul = sqrt(self.d_model)
        self.sql_len = sql_len
        self.pos_encoding = Parameter(self.get_position_encoding_table(), requires_grad = False)
        self.layers = nn.ModuleList([self.get_layer() for i in range(self.n_layers)])
        self.dropout = nn.Dropout(0.1)

    def get_layer(self):
        layer = nn.ModuleDict()
        layer['mha'] = TransformerBlock(num_head = self.n_head, d_model = self.d_model, masked=self.masked) 
        if self.layer_norm:
            layer['norm1'] = nn.LayerNorm(self.d_model)
            layer['norm2'] = nn.LayerNorm(self.d_model)
        
        if self.d_ffn:
            layer['ffn'] = nn.Sequential(
                    nn.Linear(self.d_model, self.d_ffn),
                    nn.ReLU(),
                    nn.Linear(self.d_ffn, self.d_model)
            )
        return layer

    def get_position_encoding_table(self):
        def func(pos):
            pe =  [pos / np.power(10000, 2 * i / self.d_model) for i in range(self.d_model)]
            pe[0:self.d_model:2] = np.sin(pe[0:self.d_model:2])
            pe[1:self.d_model:2] = np.cos(pe[1:self.d_model:2])
            return pe
        table = np.array([func(p) for p in range(self.sql_len)])
        return torch.FloatTensor(table).contiguous()
        
    def forward(self, x):
        # sql_len, bacth_sz, channel -> batch_sz, sql_len, channel
        b_sz, s_len, channel = x.shape
        x = x + self.pos_encoding[:s_len]
        x = self.dropout(x)
        for layer in self.layers:
            # print('done!')
            y, _ = layer['mha'](x, x, x)
            y = self.dropout(y)
            x = x + y
            if self.layer_norm:
                x = layer['norm1'](x)
            
            if self.d_ffn:
                y = layer['ffn'](x)
                y = self.dropout(y)
                x = x + y
            
            if self.layer_norm:
                x = layer['norm2'](x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, sql_len, n_voc, n_layers = 12, masked = True, n_head = 8, d_model = 512, d_ffn = 2048, layer_norm = True):
        super().__init__()
        self.n_layers, self.n_voc, self.masked, self.n_head, self.d_model, self.d_ffn = \
            n_layers, n_voc, masked, n_head, d_model, d_ffn
        self.layer_norm = layer_norm
        self.weight_mul = sqrt(self.d_model)
        self.sql_len = sql_len
        self.pos_encoding = Parameter(self.get_position_encoding_table(), requires_grad = False)
        self.layers = nn.ModuleList([self.get_layer() for i in range(self.n_layers)])
        self.dropout = nn.Dropout(0.1)

    def get_layer(self):
        layer = nn.ModuleDict()
        layer['masked_mha'] = TransformerBlock(self.n_head, self.masked, self.d_model)
        layer['mha'] = TransformerBlock(self.n_head, d_model = self.d_model) 
        if self.layer_norm:
            layer['norm1'] = nn.LayerNorm(self.d_model)
            layer['norm2'] = nn.LayerNorm(self.d_model)
            layer['norm3'] = nn.LayerNorm(self.d_model)
        
        if self.d_ffn:
            layer['ffn'] = nn.Sequential(
                    nn.Linear(self.d_model, self.d_ffn),
                    nn.ReLU(),
                    nn.Linear(self.d_ffn, self.d_model)
            )
        return layer

    def get_position_encoding_table(self):
        def func(pos):
            return [pos / np.power(10000, 2 * i / self.d_model) for i in range(self.d_model)]
        table = np.array([func(p) for p in range(self.sql_len)])
        return torch.FloatTensor(table).contiguous()
        
    def forward(self, kv, x):
        # sql_len, bacth_sz, channel -> batch_sz, sql_len, channel
        b_sz, s_len, channel = x.shape
        x = x + self.pos_encoding[:s_len]
        x = self.dropout(x)
        for layer in self.layers:
            y, _ = layer['masked_mha'](x, x, x)
            y = self.dropout(y)
            x = x + y
            if self.layer_norm:
                x = layer['norm1'](x)
            y, _ = layer['mha'](x, kv, kv)
            y = self.dropout(y)
            x = x + y
            if self.layer_norm:
                x = layer['norm2'](x)
            
            if self.d_ffn:
                y = layer['ffn'](x)
                y = self.dropout(y)
                x = x + y
            
            if self.layer_norm:
                x = layer['norm3'](x)
        return x




 