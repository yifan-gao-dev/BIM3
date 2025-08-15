import torch
import torch.nn as nn
import torch.nn.functional as F
from ts_benchmark.baselines.bim3.layers.TCALayers import ReviewTimestampCrossAttentionLayer as RTCALayer
from ts_benchmark.baselines.bim3.layers.TCALayers import ForeseeTimestampCrossAttentionLayer as FTCALayer

from .RevIN import RevIN

class TimeStampModule(nn.Module):
    def __init__(self, enc_in, seq_len, pred_len, down_sampling_layers, down_sampling_window, down_sampling_method='avg', hidden_dim=128, attention_dropout=0.1, heads=8, in_dim=11, epsilon=0.001):
        super(TimeStampModule, self).__init__()
        
        self.n_vars = enc_in
        
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_window = down_sampling_window
        self.down_sampling_method = down_sampling_method
        self.epsilon = epsilon
        self.revin = RevIN(self.n_vars)
        
        self.weight_mlp_lst = torch.nn.ModuleList(
            nn.Sequential(
                nn.Linear(seq_len // (down_sampling_window**i), hidden_dim // (down_sampling_window**i)),
                nn.GELU(),
                nn.Linear(hidden_dim // (down_sampling_window**i), 2),
                nn.Softmax(dim=-1)
            )
            for i in range(self.down_sampling_layers+1)
        )
        self.review_tca_layer = RTCALayer(pred_len, in_dim, hidden_dim, heads, enc_in, attention_dropout)
        self.foresee_tca_layer = FTCALayer(seq_len, in_dim, hidden_dim, heads, enc_in,attention_dropout)
        self.mlp = nn.Sequential(nn.Linear(pred_len, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
    
    def __multi_scale_process_inputs(self, x_enc: torch.Tensor)->list[torch.Tensor]:
        if self.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.down_sampling_window, return_indices=False)
        elif self.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.down_sampling_window)
        else:
            raise NotImplementedError(f"No other down_pool method named {self.down_sampling_method}")
    
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc

        x_enc_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        
        for i in range(self.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

        return x_enc_sampling_list
    
    def forward(self, input_data_, input_mark, output_base, output_mark):
        input_data = self.revin(input_data_, "norm")
        B, _, n_vars = input_data.shape

        ### down sample input_data(X), input_mark(\bar{X}), output_mark(\bar{Y})
        input_data_lst = self.__multi_scale_process_inputs(input_data)
        input_mark_lst = self.__multi_scale_process_inputs(input_mark)
        
        ### get review and foresee features
        tca_dec_lst = []
        tca_enc_lst = []
        
        for i, ds_input_mark in enumerate(input_mark_lst):
            tca_dec_i = self.review_tca_layer(output_mark, ds_input_mark)
            tca_enc_i = self.foresee_tca_layer(ds_input_mark, output_mark)
            tca_dec_lst.append(tca_dec_i)
            tca_enc_lst.append(tca_enc_i)
        
        ### combine: output_base and tca_output
        
        latent_feat_lst = []
        for i, (ds_input_data, tca_enc_i, tca_dec_i) in enumerate(zip(input_data_lst, tca_enc_lst, tca_dec_lst)):
            x_enc_i = tca_enc_i + ds_input_data
            weight_i = self.weight_mlp_lst[i](x_enc_i.permute(0, 2, 1)).unsqueeze(1)
            x_dec_i = torch.stack([output_base, tca_dec_i], dim=-1)
            latent_feat_i = torch.sum((x_dec_i * weight_i), dim=-1)
            latent_feat_i = torch.sum((x_dec_i * weight_i), dim=-1)
            latent_feat_lst.append(latent_feat_i)
        
        latent_feat = torch.mean(torch.stack(latent_feat_lst, dim=-1), dim=-1)

        _latent_feat = self.mlp(latent_feat.permute(0, 2, 1)) # [B, C]
        tstamp_feat = _latent_feat.mean(0) # shape [C,1]
        L_constraint = self.epsilon * torch.abs(tstamp_feat.mean(0))
        return L_constraint