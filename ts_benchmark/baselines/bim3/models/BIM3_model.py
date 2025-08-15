
import torch
import torch.nn as nn
from einops import rearrange

from ts_benchmark.baselines.bim3.layers.MoE import MultiScaleMoE
from ts_benchmark.baselines.bim3.layers.TstampDualCrossAttenModule import TimeStampModule
from ts_benchmark.baselines.bim3.utils.masked_attention import Mahalanobis_mask, Encoder, EncoderLayer, FullAttention, AttentionLayer


class BIM3_model(nn.Module):
    def __init__(self, config):
        super(BIM3_model, self).__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.horizon
        self.moe = MultiScaleMoE(config)
        self.n_vars = config.enc_in
        self.cor_matrix = Mahalanobis_mask(config.seq_len)
        self.decoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
                            config.factor,
                            attention_dropout=config.dropout,
                            output_attention=config.output_attention,
                        ),
                        config.d_model,
                        config.n_heads,
                    ),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for _ in range(config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model)
        )

        self.linear_head = nn.Sequential(nn.Linear(config.d_model, config.horizon), nn.Dropout(config.fc_dropout))
        
        self.tstamp_module = TimeStampModule(config.enc_in, config.seq_len, config.horizon, config.down_sampling_layers, config.down_sampling_window, 'avg', hidden_dim=config.d_model, attention_dropout=config.dropout)

    def forward(self, input_data, input_mark, target_mark):
        # x: [batch_size, seq_len, n_vars]
        channel_independent_input = rearrange(input_data, 'b l n -> (b n) l 1')
        ms_temporal_feat, L_importance = self.moe(channel_independent_input)
        temporal_feature = rearrange(ms_temporal_feat, '(b n) l 1 -> b l n', b=input_data.shape[0])

        # B x d_model x n_vars -> B x n_vars x d_model
        temporal_feature = rearrange(temporal_feature, 'b d n -> b n d')
        if self.n_vars > 1:
            changed_input = rearrange(input_data, 'b l n -> b n l')
            channel_corre_matrix = self.cor_matrix(changed_input) # channel_mask: shape(C, C)
            x_fused, _ = self.decoder(x=temporal_feature, attn_mask=channel_corre_matrix)
            output = self.linear_head(x_fused)
        else:
            output = temporal_feature
            output = self.linear_head(output)

        output = rearrange(output, 'b n d -> b d n')
        output_pred = self.moe.revin(output, "denorm")

        L_constraint = self.tstamp_module(input_data, input_mark, output[:, -self.pred_len:, :], target_mark[:, -self.pred_len:, :])
        
        return output_pred, L_importance, L_constraint
