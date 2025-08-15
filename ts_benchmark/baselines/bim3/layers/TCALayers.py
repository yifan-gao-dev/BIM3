import torch
import torch.nn as nn
from math import sqrt

class ReviewTimestampCrossAttentionLayer(nn.Module):
    def __init__(self, pred_len, in_dim, hidden_dim, heads, n_vars, attention_dropout=0.1):
        super(ReviewTimestampCrossAttentionLayer, self).__init__()
    
        # qkv
        self.pred_len = pred_len
        self.heads = heads
        self.query = nn.Linear(in_dim, hidden_dim)
        self.key = nn.Linear(in_dim, hidden_dim)
        self.value = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(attention_dropout)
        
        self.output_projection = nn.Linear(hidden_dim, n_vars)
        
    def forward(self, pred_tstamp, his_tstamp):
        B, S, _ = pred_tstamp.shape
        B, T, _ = his_tstamp.shape
        H = self.heads
        
        pred_t_q = self.query(pred_tstamp).view(B, S, H, -1)
        his_t_k = self.key(his_tstamp).view(B, T, H, -1)
        his_t_v = self.value(his_tstamp).view(B, T, H, -1)
        
        _, _, _, d_embed = pred_t_q.shape
        scale = 1. / sqrt(d_embed)
        
        scores = torch.einsum("bshe, bthe -> bhst", pred_t_q, his_t_k)
        A = self.dropout(torch.softmax(scores * scale, dim=-1))
        
        review_embed = torch.einsum("bhst, bthe -> bshe", A, his_t_v).reshape(B, S, -1)
        output = self.output_projection(review_embed)
        return output   # [B, H, n_vars]
    
class ForeseeTimestampCrossAttentionLayer(nn.Module):
    def __init__(self, seq_len, in_dim, hidden_dim, heads, n_vars, attention_dropout=0.1):
        super(ForeseeTimestampCrossAttentionLayer, self).__init__()
        
        self.seq_len = seq_len
        self.heads = heads
        self.query = nn.Linear(in_dim, hidden_dim)
        self.key = nn.Linear(in_dim, hidden_dim)
        self.value = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(attention_dropout)
        
        self.output_projection = nn.Linear(hidden_dim, n_vars)
        
    def forward(self, his_tstamp, pred_tstamp):
        B, S, _ = his_tstamp.shape
        B, T, _ = pred_tstamp.shape
        H = self.heads
        
        his_t_q = self.query(his_tstamp).view(B, S, H, -1)
        pred_t_k = self.key(pred_tstamp).view(B, T, H, -1)
        pred_t_v = self.value(pred_tstamp).view(B, T, H, -1)
        
        _, _, _, dim_embed = his_t_q.shape
        scale = 1. / sqrt(dim_embed)
        
        scores = torch.einsum("bshe, bthe -> bhst", his_t_q, pred_t_k)
        A = self.dropout(torch.softmax(scores * scale, dim=-1))
        
        foresee_embed = torch.einsum("bhst, bthe -> bshe", A, pred_t_v).reshape(B, S, -1)
        
        output = self.output_projection(foresee_embed)
        
        return output