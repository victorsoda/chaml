import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import pickle
import numpy as np



class Learner(nn.Module):


    def __init__(self, config):

        super(Learner, self).__init__()

        self.config = config
        # p_size = self.config['num_pois'] # + 1
        p_type_size = self.config['num_poi_types']
        time_size = self.config['num_time']
        embed_dim = self.config['embed_dim']
        poiid_dim = self.config['poiid_dim']
        mlp_hidden = self.config['mlp_hidden']
        self.with_cont_feat = self.config['with_cont_feat']

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()

        # embedding modules
        self.init_emb(1, poiid_dim)  # poi_id_embedding
        self.init_emb(p_type_size, embed_dim)  # poi_type_embedding
        self.init_emb(time_size, embed_dim)  # time_embedding


        # attention module
        if self.with_cont_feat:
            candi_dim = poiid_dim + embed_dim * 2 + 2   # +2: continuous features
        else:
            candi_dim = poiid_dim + embed_dim * 2
        self.init_fc(candi_dim * 4, embed_dim)  # (K, V, K-V, K*V)
        self.init_fc(embed_dim, 1)

        # final MLP module
        self.init_fc(candi_dim * 2, mlp_hidden)
        self.init_fc(mlp_hidden, mlp_hidden)
        self.init_fc(mlp_hidden, 2)

        for i in range(self.config['global_fix_var']):
            self.vars[i].requires_grad = False


    def init_emb(self, max_size, embed_dim):
        w = nn.Parameter(torch.ones(max_size, embed_dim))
        init.xavier_normal_(w)
        self.vars.append(w)

        
    def init_fc(self, input_dim, output_dim):
        w = nn.Parameter(torch.ones(output_dim, input_dim)) 
        b = nn.Parameter(torch.zeros(output_dim))
        init.xavier_normal_(w)
        self.vars.append(w)
        self.vars.append(b)   

    
    def attention(self, att_w1, att_b1, att_w2, att_b2, K, V, mask=None):
        '''
        :param K: (batch_size, d)
        :param V: (batch_size, hist_len, d)
        :return: (batch_size, d)
        '''
        K = K.unsqueeze(dim=1).expand(V.size())
        fusion = torch.cat([K, V, K-V, K*V], dim=-1)

        x = F.linear(fusion, att_w1, att_b1)
        x = F.relu(x)
        score = F.linear(x, att_w2, att_b2)

        if mask is not None:
            mask = mask.unsqueeze(dim=-1)
            score = score.masked_fill(mask, -2 ** 32 + 1)

        alpha = F.softmax(score, dim=1)
        alpha = F.dropout(alpha, p=0.5)  # TODO: training=True? False?
        att = (alpha * V).sum(dim=1)
        return att

    
    def forward(self, batch_uid, batch_hist, batch_candi, vars=None, scaler=None):  # TODO
        '''
        :param batch_uid: (bsz, )
        :param batch_hist: (bsz, time_step, 5)
        :param batch_candi: (bsz, 5)
        :return: 
        '''

        if vars is None:
            vars = self.vars
        
        poi_emb_w, poi_type_emb_w, time_emb_w = vars[:3]
        att_w1, att_b1, att_w2, att_b2 = vars[3:7]
        mlp_w1, mlp_b1, mlp_w2, mlp_b2, mlp_w3, mlp_b3 = vars[7:]

        if self.with_cont_feat and scaler is not None:
            hist_feat = []
            candi_feat = []
            if 'dist' in scaler:  # datapoint[3:]: u-p dist, dtime, delta_dist
                mean_dist, std_dist = scaler['dist']
                mean_dtime, std_dtime = scaler['dtime']
                try:
                    hist_feat.append((batch_hist[:, :, 3].unsqueeze(-1).float() - mean_dist) / std_dist)
                except:
                    print("====debug====")
                    print(batch_hist.shape)
                    print(batch_hist[:, :, 3])
                    exit(2)
                hist_feat.append((batch_hist[:, :, 4].unsqueeze(-1).float() - mean_dtime) / std_dtime)
                candi_feat.append((batch_candi[:, 3].unsqueeze(-1).float() - mean_dist) / std_dist)
                candi_feat.append((batch_candi[:, 4].unsqueeze(-1).float() - mean_dtime) / std_dtime)

            hist_embed = torch.cat([
                F.embedding(input=batch_hist[:, :, 0], weight=poi_emb_w, padding_idx=0),
                F.embedding(input=batch_hist[:, :, 1], weight=poi_type_emb_w, padding_idx=0),
                F.embedding(input=batch_hist[:, :, 2], weight=time_emb_w),
                torch.cat(hist_feat, dim=-1)
            ], dim=-1)
            candi_embed = torch.cat([
                F.embedding(input=batch_candi[:, 0], weight=poi_emb_w, padding_idx=0),
                F.embedding(input=batch_candi[:, 1], weight=poi_type_emb_w, padding_idx=0),
                F.embedding(input=batch_candi[:, 2], weight=time_emb_w),
                torch.cat(candi_feat, dim=-1)
            ], dim=-1)
        else:
            hist_embed = torch.cat([
                F.embedding(input=batch_hist[:, :, 0], weight=poi_emb_w, padding_idx=0),
                F.embedding(input=batch_hist[:, :, 1], weight=poi_type_emb_w, padding_idx=0),
                F.embedding(input=batch_hist[:, :, 2], weight=time_emb_w),
            ], dim=-1)
            candi_embed = torch.cat([
                F.embedding(input=batch_candi[:, 0], weight=poi_emb_w, padding_idx=0),
                F.embedding(input=batch_candi[:, 1], weight=poi_type_emb_w, padding_idx=0),
                F.embedding(input=batch_candi[:, 2], weight=time_emb_w),
            ], dim=-1)
        mask = (batch_hist[:, :, 0] == 0)            

        hist = self.attention(att_w1, att_b1, att_w2, att_b2, candi_embed, hist_embed, mask)

        embeds = torch.cat([hist, candi_embed], dim=-1) 
        embeds = F.dropout(embeds, p=0.5)

        fc1 = F.linear(embeds, mlp_w1, mlp_b1)
        fc1 = F.relu(fc1)
        fc1 = F.dropout(fc1, p=0.5)

        fc2 = F.linear(fc1, mlp_w2, mlp_b2)
        fc2 = F.relu(fc2)
        fc2 = F.dropout(fc2, p=0.5)

        prediction = F.linear(fc2, mlp_w3, mlp_b3)
        return prediction

        
    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()
        

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.z
        :return:
        """
        return self.vars
        




