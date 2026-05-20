import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import torch
from ViT import *
# from Gated_Attention_MIL import GatedAttention
from nystrom_attention import NystromAttention

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x

class CrossAttention(nn.Module):
    def __init__(self, in_dim1=512, in_dim2=512, k_dim=512, v_dim=512, num_heads=4):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Sequential(nn.Linear(in_dim1, k_dim * num_heads, bias=False))
        self.proj_k2 = nn.Sequential(nn.Linear(in_dim2, k_dim * num_heads, bias=False))
        self.proj_v2 = nn.Sequential(nn.Linear(in_dim2, v_dim * num_heads, bias=False))
        self.proj_o = nn.Sequential(nn.Linear(v_dim * num_heads, in_dim1))

    def forward(self, query, key_value, mask=None):
        batch_size, seq_len1, in_dim1 = query.size()
        seq_len2 = key_value.size(1)
        #[1 N 512]
        q1 = self.proj_q1(query).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k2 = self.proj_k2(key_value).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v2 = self.proj_v2(key_value).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q1, k2) / self.k_dim ** 0.5

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        output = self.proj_o(output)

        return output, attn


# Code integrated from https://github.com/hrzhang1123/DTFD-MIL

class Attention2(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention2, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

    def forward(self, x, isNorm=True):
        ## x: N x L
        A = self.attention(x)  ## N x K
        A = torch.transpose(A, 1, 0)  # KxN
        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N
        return A  ### K x N


class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        ## x: B X N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U)  # NxK
        A = torch.transpose(A, -1, -2)  # KxN

        if isNorm:
            A = F.softmax(A, dim=-1)  # softmax over N
        return A  ### K x N


class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x


class Attention_with_Classifier(nn.Module):
    def __init__(self, L=512, D=128, K=1, num_cls=2, droprate=0):
        super(Attention_with_Classifier, self).__init__()
        self.attention = Attention_Gated(L, D, K)
        self.classifier = Classifier_1fc(L, num_cls, droprate)
    def forward(self, x): ## x: N x L
        AA = self.attention(x)  ## K x N
        afeat = torch.squeeze(torch.bmm(AA, x)) ## K x L
        pred = self.classifier(afeat) ## K x num_cls
        return pred, afeat


class residual_block(nn.Module):
    def __init__(self, nChn=512):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
            )
    def forward(self, x):
        tt = self.block(x)
        x = x + tt
        return x


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x


def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps



class DTFDMIL(nn.Module):
    def __init__(self, num_classes, total_instance = 500, num_group=4, feat_dim=512, distill_type='AFS', numLayer_Res=0, droprate=0.):
        super().__init__()

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.droprate = droprate
        self.total_instance = total_instance
        self.num_group = num_group
        self.distill_type = distill_type

        self.dimReduction = DimReduction(self.feat_dim, self.feat_dim, numLayer_Res=numLayer_Res)
        self.attention = Attention_Gated(self.feat_dim)
        self.subClassifier = Classifier_1fc(self.feat_dim, self.num_classes, droprate=self.droprate)
        self.attCls = Attention_with_Classifier(L=self.feat_dim, num_cls=self.num_classes, droprate=self.droprate)

    def forward(self, x, args):
        A = []
        feat_index = list(range(x.shape[1]))
        random.shuffle(feat_index)
        index_chunk_list = np.array_split(np.array(feat_index), self.num_group)
        index_chunk_list = [sst.tolist() for sst in index_chunk_list]

        pseudo_feat = []
        sub_preds = []
        for tindex in index_chunk_list:
            subFeat = torch.index_select(x, dim=1, index=torch.LongTensor(tindex).cuda(args.cuda_choice))

            tmidFeat = self.dimReduction(subFeat)
            tAA = self.attention(tmidFeat).squeeze(-2)
            A.append(tAA)
            tattFeats = torch.einsum('bns,bn->bns', tmidFeat, tAA)  ### n x fs
            af_inst_feat = torch.sum(tattFeats, dim=-2)  # B x 1 x fs
            tPredict = self.subClassifier(af_inst_feat)  # 1 x 2
            sub_preds.append(tPredict)
            if self.distill_type == 'AFS':
                pseudo_feat.append(torch.unsqueeze(af_inst_feat, dim=1))

        slide_pseudo_feat = torch.cat(pseudo_feat, dim=1)  # B x num_group x fs
        logits, slide_embeddings = self.attCls(slide_pseudo_feat)
        sub_preds = torch.cat(sub_preds, dim=0)

        return slide_embeddings,sub_preds,A

class GatedAttention(nn.Module):
    def __init__(self,n_classes):
        super(GatedAttention, self).__init__()
        # 全连接层的隐含单元
        self.L = 512 #512
        self.D = 128
        self.K = 1
        self.nclass = n_classes

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),  #（ in_dimension, out，dimension）
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, self.nclass),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N : KxN

        M = torch.mm(A, x)  # KxL

        return M , A # Y_prob Y_hat A

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv1d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv1d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv1d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x):
        x = x.transpose(1,2)
        x = (self.proj(x) + x + self.proj1(x) + self.proj2(x)).transpose(1,2)
        return x

class Conv_layer(nn.Module):
    def __init__(self, dim=512):
        super(Conv_layer, self).__init__()
        self.proj = nn.Conv1d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv1d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv1d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x):
        x = x.transpose(1,2)
        x = (self.proj(x) + x + self.proj1(x) + self.proj2(x)).transpose(1,2)
        return x

def generate_random_mask(mask_size, true_ratio):
    # 计算 True 的数量
    num_true = int(mask_size * true_ratio)

    # 创建一个包含指定数量 True 和 False 的张量
    mask = torch.cat((torch.ones(num_true, dtype=torch.bool),
                      torch.zeros(mask_size - num_true, dtype=torch.bool)))

    # 随机打乱掩码
    random_mask = mask[torch.randperm(mask_size)].cuda(0)

    # 仅仅返回为True的索引
    true_indices = torch.nonzero(random_mask).squeeze()

    return true_indices

class MultiLabel(nn.Module):
    def __init__(self, AFB_class, GMS_class, PAS_class, input_dim=512,
                 mlp_head=False, args=None):
        super(MultiLabel, self).__init__()

        self.embed_dim = input_dim
        self.cross_attention = CrossAttention()

        self.criterion = torch.nn.CrossEntropyLoss()
        self.loss_kl = torch.nn.KLDivLoss()
        self.loss_cosine = torch.nn.CosineEmbeddingLoss()

        self.proto = nn.Parameter(torch.zeros(1, 3, self.embed_dim))
        self.proto_multi_attention = TransLayer()

        self.fusion1 = nn.Sequential(
            nn.Linear(self.embed_dim * 3, self.embed_dim),
            nn.ReLU()
        )
        self.fusion2 = nn.Sequential(
            nn.Linear(self.embed_dim * 3, self.embed_dim),
            nn.ReLU()
        )
        self.fusion3 = nn.Sequential(
            nn.Linear(self.embed_dim * 3, self.embed_dim),
            nn.ReLU()
        )

        self.GatedAttention = GatedAttention(AFB_class)

        self.classifier1 = nn.Linear(self.embed_dim, AFB_class)
        self.classifier2 = nn.Linear(self.embed_dim, GMS_class)
        self.classifier3 = nn.Linear(self.embed_dim, PAS_class)



    def forward(self, X, text, AFB_labels, GMS_labels, PAS_labels, force_return_loss=None):
        X = X.unsqueeze(0)

        proto = self.proto_multi_attention(self.proto)
        #蒸馏损失
        logits_AFB_proto = self.classifier1(proto[:,0,:])  # [B, n_classes]
        Y_prob_AFB_proto = F.softmax(logits_AFB_proto, dim=-1)
        logits_GMS_proto = self.classifier2(proto[:,1,:])  # [B, n_classes]
        Y_prob_GMS_proto = F.softmax(logits_GMS_proto, dim=-1)
        logits_PAS_proto = self.classifier3(proto[:,2,:])  # [B, n_classes]
        Y_prob_PAS_proto = F.softmax(logits_PAS_proto, dim=-1)
        # print(X.size())

        X_cross, att = self.cross_attention(proto, X)  # [1,3,512]
        X_Gated, A = self.GatedAttention(X) #[1,512]

        #文本和图像cosine
        L_cos = self.loss_cosine(X_Gated ,text.squeeze().unsqueeze(0), torch.tensor([1]).cuda())

        X_AFB_cross = X_cross[:,0,:].unsqueeze(0)

        X_AFB_fusion = self.fusion1(torch.cat([X_Gated.unsqueeze(0),X_AFB_cross, text],dim=-1))
        logits_AFB = self.classifier1(X_AFB_fusion.squeeze().unsqueeze(0))  # [B, n_classes]
        Y_hat_AFB = torch.argmax(logits_AFB, dim=-1)
        Y_prob_AFB = F.softmax(logits_AFB, dim=-1)

        X_GMS_cross = X_cross[:, 1, :].unsqueeze(0)
        X_GMS_fusion = self.fusion2(torch.cat([X_Gated.unsqueeze(0), X_GMS_cross, text], dim=-1))
        logits_GMS = self.classifier2(X_GMS_fusion.squeeze().unsqueeze(0))  # [B, n_classes]
        Y_hat_GMS = torch.argmax(logits_GMS, dim=-1)
        Y_prob_GMS = F.softmax(logits_GMS, dim=-1)

        X_PAS_cross = X_cross[:, 2, :].unsqueeze(0)
        X_PAS_fusion = self.fusion3(torch.cat([X_Gated.unsqueeze(0), X_PAS_cross, text], dim=-1))
        logits_PAS = self.classifier3(X_PAS_fusion.squeeze().unsqueeze(0))  # [B, n_classes]
        Y_hat_PAS = torch.argmax(logits_PAS, dim=-1)
        Y_prob_PAS = F.softmax(logits_PAS, dim=-1)

        if self.training == True:
            # loss computation
            AFB_loss = self.criterion(Y_prob_AFB, AFB_labels)
            GMS_loss = self.criterion(Y_prob_GMS, GMS_labels)
            PAS_loss = self.criterion(Y_prob_PAS, PAS_labels)
            L_kl_AFB = self.loss_kl(torch.log(Y_prob_AFB),Y_prob_AFB_proto)
            L_kl_GMS = self.loss_kl(torch.log(Y_prob_GMS), Y_prob_GMS_proto)
            L_kl_PAS = self.loss_kl(torch.log(Y_prob_PAS), Y_prob_PAS_proto)

            results_dict = {'loss_AFB': AFB_loss, 'logits_AFB': logits_AFB, 'pred_AFB': Y_hat_AFB,
                            'labels_AFB': AFB_labels,
                            'loss_GMS': GMS_loss, 'logits_GMS': logits_GMS, 'pred_GMS': Y_hat_GMS,
                            'labels_GMS': GMS_labels,
                            'loss_PAS': PAS_loss, 'logits_PAS': logits_PAS, 'pred_PAS': Y_hat_PAS,
                            'labels_PAS': PAS_labels,

                            'L_kl_AFB': L_kl_AFB, 'L_kl_GMS': L_kl_GMS, 'L_kl_PAS': L_kl_PAS, 'L_cos': L_cos,
                            'Att_AFB': torch.mean(att[:, :, 0, :], dim=1, keepdim=True),
                            'Att_GMS': torch.mean(att[:, :, 1, :], dim=1, keepdim=True),
                            'Att_PAS': torch.mean(att[:, :, 2, :], dim=1, keepdim=True)
                , 'A': A
                            }

            return results_dict


        else:
            results_dict = { 'logits_AFB': logits_AFB, 'pred_AFB': Y_hat_AFB, 'labels_AFB': AFB_labels,
                          'logits_GMS': logits_GMS, 'pred_GMS': Y_hat_GMS, 'labels_GMS': GMS_labels,
                          'logits_PAS': logits_PAS, 'pred_PAS': Y_hat_PAS, 'labels_PAS': PAS_labels,

                        'Att_AFB': torch.mean(att[ :,:,0,:], dim=1, keepdim=True),   'Att_GMS': torch.mean(att[ :,:,1,:], dim=1, keepdim=True),  'Att_PAS': torch.mean(att[ :,:,2,:], dim=1, keepdim=True)
                        ,'A':A
                        }
            return results_dict
