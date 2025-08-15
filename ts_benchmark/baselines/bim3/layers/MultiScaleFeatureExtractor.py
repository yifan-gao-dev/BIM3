import torch
import torch.nn as nn

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time data
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time data
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

"""

class `MultiScaleSeasonMixing`, `MultiScaleTrendMixing` and `PastDecomposableMixing` are referring to TiemMixer's implementation.
Github repo link: https://github.com/kwuking/TimeMixer 
Paper link: https://openreview.net/pdf?id=7oLshfEIC2

"""

class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """
    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1))
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1))
                    ),

                )
                for i in range(configs.down_sampling_layers)
            ]
        )
        self.init_weight()
    
    def init_weight(self):
        for i, sub_module in enumerate(self.down_sampling_layers):
            for j, sub_linear in enumerate(sub_module):
                if isinstance(sub_linear, nn.Linear):
                    _in_dim = sub_linear.in_features
                    _out_dim = sub_linear.out_features
                    sub_linear.weight = nn.Parameter((1/_in_dim) * torch.ones([_out_dim, _in_dim]))

    def forward(self, season_list):

        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """
    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i)
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i)
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])
        self.init_weight()
        
    def init_weight(self):
        for i, us_layer in enumerate(self.up_sampling_layers):
            for j, sub_linear in enumerate(us_layer):
                if isinstance(sub_linear, nn.Linear):
                    _in_dim = sub_linear.in_features
                    _out_dim = sub_linear.out_features
                    sub_linear.weight = nn.Parameter((1/_in_dim) * torch.ones([_out_dim, _in_dim]))
    
    
    def forward(self, trend_list):
        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list

class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len

        self.down_sampling_window = configs.down_sampling_window

        self.decompsition = series_decomp(configs.moving_avg)
        
        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)



    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))
        
        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for out_season, out_trend, length in zip(out_season_list, out_trend_list, length_list):
            out = out_season + out_trend
            out_list.append(out[:, :length, :])
        return out_list

class MultiScaleFeatureExtractor(nn.Module):

    def __init__(self, configs):

        super(MultiScaleFeatureExtractor, self).__init__()
        self.configs = configs
        
        self.ori_seq_len = configs.seq_len
        
        self.seq_len = configs.seq_len
        self.out_dim = configs.d_model
        """
        self.out_dim, the output dim of linear extractor
        All scales' output feature dim should be aligned to self.out_dim
        """

        self.channels = configs.enc_in
        self.enc_in = 1 if configs.CI else configs.enc_in
        
        ### here add multi-scaled related paras
        self.decomp_multi_scale_layer = PastDecomposableMixing(configs)

        self.pdb_fusion_layers = nn.ModuleList(
            [
                torch.nn.Linear(
                self.seq_len // (configs.down_sampling_window ** i),
                self.out_dim
                )
                for i in range(configs.down_sampling_layers + 1)
            ]
        )
        # parameter initialization
        for layer in self.pdb_fusion_layers:
            # for layer in sub_module:
            if isinstance(layer, nn.Linear):
                _in_dim = layer.in_features
                _out_dim = layer.out_features
                layer.weight = nn.Parameter((1/_in_dim) * torch.ones([_out_dim, _in_dim]))
    
    def __multi_scale_process_inputs(self, x_enc: torch.Tensor)->list[torch.Tensor]:
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        else:
            raise NotImplementedError(f"No other down_pool method named {self.configs.down_sampling_method}")
    
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc

        x_enc_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        
        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

        return x_enc_sampling_list

    def encoder(self, x):        
        # multi scale process
        x_multi_scale_lst = self.__multi_scale_process_inputs(x)
        # x_ms_norm_lst store the normalized multi-scale tensor
        pdm_out_lst = self.decomp_multi_scale_layer(x_multi_scale_lst)
        
        pdm_align_lst = []
        for i in range(len(pdm_out_lst)):
            pdm_align_i = self.pdb_fusion_layers[i](pdm_out_lst[i].permute(0, 2, 1))
            pdm_align_lst.append(pdm_align_i.permute(0, 2, 1))
        # pdm_align_lst[i]: shape [B', self.pred_len(d_model), 1]
        
        # fusion strategy: sum
        pdm_fused = torch.stack(pdm_align_lst, dim=-1).sum(-1)

        return pdm_fused

    def forward(self, x_enc):
        if x_enc.shape[0] == 0:
            return torch.empty((0, self.out_dim, self.enc_in)).to(x_enc.device)
        return self.encoder(x_enc)

