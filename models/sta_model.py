# -*- coding: utf-8 -*-
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
import torchmetrics
import pathlib
import os
import json

from .modules.brep_encoder import BrepEncoder
from .modules.utils.macro import *
from .modules.domain_adv.domain_discriminator import DomainDiscriminator
from .modules.domain_adv.dann import DomainAdversarialLoss
from .brepseg_model import BrepSeg

class NonLinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 512, bias=False)
        self.bn2 = nn.BatchNorm1d(512)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(512, 256, bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.dp3 = nn.Dropout(p=dropout)
        self.linear4 = nn.Linear(256, num_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, inp, return_feat=False):
        feat = inp
        x = F.relu(self.bn1(self.linear1(inp)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = F.relu(self.bn3(self.linear3(x)))
        x = self.dp3(x)
        x = self.linear4(x)
        x = F.softmax(x, dim=-1)
        if return_feat:
            return x, feat
        else:
            return x


class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, output_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(output_dim)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, inp):
        x = F.relu(self.bn1(self.linear1(inp)))
        return x


class BinaryClassifier(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 2),
            nn.Softmax(dim=-1)
        )

        for m in self.main.children():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        for module in self.main.children():
            x = module(x)
        return x


class MultiClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.n = num_classes
        def f():
            return nn.Sequential(
                nn.Linear(in_dim, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
        for i in range(num_classes):
            self.__setattr__('discriminator_%04d' % i, f())

    def forward(self, x):
        outs = [self.__getattr__('discriminator_%04d' % i)(x) for i in range(self.n)]
        return torch.cat(outs, dim=-1)


def CrossEntropyLoss(label, predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=1e-12):
    N, C = label.size()
    N_, C_ = predict_prob.size()
    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    ce = -label * torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * ce * class_level_weight) / float(N)


def EntropyLoss(predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=1e-20):
    N, C = predict_prob.size()

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    entropy = -predict_prob * torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * entropy * class_level_weight) / float(N)


def BCELossForMultiClassification(label, predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=1e-12):
    N, C = label.size()
    N_, C_ = predict_prob.size()
    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    bce = -label * torch.log(predict_prob + epsilon) - (1.0 - label) * torch.log(1.0 - predict_prob + epsilon)
    return torch.sum(instance_level_weight * bce * class_level_weight) / float(N)


class STADomainAdapt(pl.LightningModule):
    """
    PyTorch Lightning module to train/test the model.
    """
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

        # garph encoder--------------------------------------------------------------------
        # self.brep_encoder = BrepEncoder(
        #     # < for graphormer
        #     num_degree=128,  # number of in degree types in the graph
        #     num_spatial=64,  # number of spatial types in the graph
        #     num_edge_dis=64,  # number of edge dis types in the graph
        #     edge_type="multi_hop",  # edge type in the graph "multi_hop"
        #     multi_hop_max_dist=16,  # max distance of multi-hop edges
        #     # >
        #     num_encoder_layers=args.n_layers_encode,  # num encoder layers
        #     embedding_dim=args.dim_node,  # encoder embedding dimension
        #     ffn_embedding_dim=args.d_model,  # encoder embedding dimension for FFN
        #     num_attention_heads=args.n_heads,  # num encoder attention heads
        #     dropout=args.dropout,  # dropout probability
        #     attention_dropout=args.attention_dropout,  # dropout probability for"attention weights"
        #     activation_dropout=args.act_dropout,  # dropout probability after"activation in FFN"
        #     layerdrop=0.1,
        #     encoder_normalize_before=True,  # apply layernorm before each encoder block
        #     pre_layernorm=True,
        #     # apply layernorm before self-attention and ffn. Without this, post layernorm will used
        #     apply_params_init=True,  # use custom param initialization for Graphormer
        #     activation_fn="gelu",  # activation function to use
        # )
        self.pre_train = args.pre_checkpoint
        pre_trained_model = BrepSeg.load_from_checkpoint(self.pre_train)
        self.brep_encoder = pre_trained_model.brep_encoder
        # garph encoder--------------------------------------------------------------------

        # node classifier------------------------------------------------------------------
        self.num_classes = args.num_classes
        self.classifier = NonLinearClassifier(2*args.dim_node, args.num_classes, args.dropout)
        # node classifier------------------------------------------------------------------

        # known-unknown classifier----------------------------------------------------------
        self.discriminator_t = BinaryClassifier(2*args.dim_node)

        # multi classes classifier----------------------------------------------------------
        self.discriminator_p = MultiClassifier(2*args.dim_node, args.num_classes)

        # domain discriminator--------------------------------------------------------------
        self.domain_discri = DomainDiscriminator(2 * args.dim_node, hidden_size=1024)
        self.discriminator = DomainAdversarialLoss(self.domain_discri)

        self.per_face_acc_source = []
        self.per_face_acc_target = []
        self.known_classes_acc = []
        self.unknown_classes_acc = []
        self.sta_acc = []

        self.automatic_optimization = False


    def training_step(self, batch, batch_idx):
        opt_step_1_1, opt_step_1_2, opt_step_2_1 = self.optimizers()

        # =========================pre-train the multi-binary classifier
        if self.current_epoch % 10 < 2:
            self.brep_encoder.eval()
            self.classifier.eval()
            self.discriminator_p.train()
            self.discriminator_t.eval()
            self.discriminator.eval()

            # graph encoder-----------------------------------------------------------
            node_emb, graph_emb = self.brep_encoder(batch, last_state_only=True)  # graph_emb [batch_size, dim]

            # separate source-target data----------------------------------------------
            node_emb = node_emb[0].permute(1, 0, 2)  # node_emb [batch_size, max_node_num+1, dim] with global node
            node_emb = node_emb[:, 1:, :]            # node_emb [batch_size, max_node_num, dim] without global node
            node_emb_s, node_emb_t = node_emb.chunk(2, dim=0)
            graph_emb_s, graph_emb_t = graph_emb.chunk(2, dim=0)  # graph_emb [batch_size, dim]
            padding_mask_s, padding_mask_t = batch["padding_mask"].chunk(2, dim=0)  # [batch_size, max_node_num]
            node_pos_s = torch.where(padding_mask_s == False)  # [(batch_size, node_index)]
            local_feat_s = node_emb_s[node_pos_s]              # [total_nodes, dim]

            # node feature extract------------------------------------------------------
            padding_mask_s_ = ~padding_mask_s
            num_nodes_per_graph_s = torch.sum(padding_mask_s_.long(), dim=-1)  # [batch_size]
            graph_emb_s_repeat = graph_emb_s.repeat_interleave(num_nodes_per_graph_s, dim=0).to(graph_emb.device)
            node_feat_s = torch.cat((local_feat_s, graph_emb_s_repeat), dim=1)  # node_feat_s [total_nodes, dim]

            # discriminator_p--------------------------------------------------------------
            p0 = self.discriminator_p(node_feat_s)  # p0 [total_nodes, num_classes]

            # multi_classifier loss-------------------------------------------------------------
            num_node_s = node_feat_s.size()[0]
            label_s = batch["label_feature"][:num_node_s].long() - 1
            label_s = F.one_hot(label_s, self.num_classes)
            loss_sp = BCELossForMultiClassification(label_s, p0)
            self.log("train_loss_Gc", loss_sp, on_step=False, on_epoch=True)

            loss = loss_sp
            self.log("train_loss", loss, on_step=False, on_epoch=True)
            torch.cuda.empty_cache()

            opt_step_1_1.zero_grad()
            self.manual_backward(loss)
            opt_step_1_1.step()

        # =========================pre-train the known/unknown discriminator
        elif self.current_epoch % 10 < 5:
            self.brep_encoder.eval()
            self.classifier.eval()
            self.discriminator_p.eval()
            self.discriminator_t.train()
            self.discriminator.eval()

            # graph encoder-----------------------------------------------------------
            node_emb, graph_emb = self.brep_encoder(batch, last_state_only=True)  # graph_emb [batch_size, dim]

            # separate source-target data----------------------------------------------
            node_emb = node_emb[0].permute(1, 0, 2)  # node_emb [batch_size, max_node_num+1, dim] with global node
            node_emb = node_emb[:, 1:, :]  # node_emb [batch_size, max_node_num, dim] without global node
            node_emb_s, node_emb_t = node_emb.chunk(2, dim=0)
            graph_emb_s, graph_emb_t = graph_emb.chunk(2, dim=0)  # graph_emb [batch_size, dim]
            padding_mask_s, padding_mask_t = batch["padding_mask"].chunk(2, dim=0)  # [batch_size, max_node_num]
            node_pos_s = torch.where(padding_mask_s == False)  # [(batch_size, node_index)]
            local_feat_s = node_emb_s[node_pos_s]  # [total_nodes, dim]
            node_pos_t = torch.where(padding_mask_t == False)  # [(batch_size, node_index)]
            local_feat_t = node_emb_t[node_pos_t]  # [total_nodes, dim]

            # node feature extract------------------------------------------------------
            padding_mask_s_ = ~padding_mask_s
            num_nodes_per_graph_s = torch.sum(padding_mask_s_.long(), dim=-1)  # [batch_size]
            graph_emb_s_repeat = graph_emb_s.repeat_interleave(num_nodes_per_graph_s, dim=0).to(graph_emb.device)
            node_feat_s = torch.cat((local_feat_s, graph_emb_s_repeat), dim=1)  # node_feat_s [total_nodes, dim]

            padding_mask_t_ = ~padding_mask_t
            num_nodes_per_graph_t = torch.sum(padding_mask_t_.long(), dim=-1)  # [batch_size]
            graph_emb_t_repeat = graph_emb_t.repeat_interleave(num_nodes_per_graph_t, dim=0).to(graph_emb.device)
            node_feat_t = torch.cat((local_feat_t, graph_emb_t_repeat), dim=1)  # node_feat_t [total_nodes, dim]

            # discriminator_p--------------------------------------------------------------
            # p0 = self.discriminator_p(node_feat_s)           # p0 [total_nodes, num_classes]
            p1 = self.discriminator_p(node_feat_t).detach()  # p1 [total_nodes, num_classes]
            p2 = torch.max(p1, dim=-1)[0]                    # p2 [total_nodes]

            # discriminator_t--------------------------------------------------------------
            num_known_sample = int(0.05*p2.size()[0])
            w = torch.sort(p2, dim=0)[1][-num_known_sample:]  # known node index
            h = torch.sort(p2, dim=0)[1][:num_known_sample]   # unknown node index
            node_feat_known = torch.index_select(node_feat_t, 0, w.view(num_known_sample))
            node_feat_unknown = torch.index_select(node_feat_t, 0, h.view(num_known_sample))
            pred_known = self.discriminator_t(node_feat_known)  # known
            pred_unknown = self.discriminator_t(node_feat_unknown)  # unknown

            # multi_classifier loss-------------------------------------------------------------
            # num_node_s = node_feat_s.size()[0]
            # label_s = batch["label_feature"][:num_node_s].long() - 1
            # label_s = F.one_hot(label_s, self.num_classes)
            # loss_sp = BCELossForMultiClassification(label_s, p0)
            # self.log("train_loss_Gc", loss_sp, on_step=False, on_epoch=True)

            # known/unknown discriminator loss------------------------------------------------
            label_known = torch.from_numpy(np.concatenate((np.ones((num_known_sample, 1)), np.zeros((num_known_sample, 1))), axis=-1).astype('float32')).to(pred_known.device)
            label_unknown = torch.from_numpy(np.concatenate((np.zeros((num_known_sample, 1)), np.ones((num_known_sample, 1))), axis=-1).astype('float32')).to(pred_unknown.device)
            loss_tt = CrossEntropyLoss(label_known, pred_known)
            loss_tt += CrossEntropyLoss(label_unknown, pred_unknown)
            self.log("train_loss_Gb", loss_tt, on_step=False, on_epoch=True)

            loss = loss_tt
            self.log("train_loss", loss, on_step=False, on_epoch=True)
            torch.cuda.empty_cache()
            opt_step_1_2.zero_grad()
            self.manual_backward(loss)
            opt_step_1_2.step()

        # =========================train domain discriminator with weight
        # =========================train classifier with unknown-label
        else:
            self.brep_encoder.train()
            self.classifier.train()
            self.discriminator_p.eval()
            self.discriminator_t.eval()
            self.discriminator.train()

            # graph encoder-----------------------------------------------------------
            node_emb, graph_emb = self.brep_encoder(batch, last_state_only=True)  # graph_emb [batch_size, dim]

            # separate source-target data----------------------------------------------
            node_emb = node_emb[0].permute(1, 0, 2)  # node_emb [batch_size, max_node_num+1, dim] with global node
            node_emb = node_emb[:, 1:, :]  # node_emb [batch_size, max_node_num, dim] without global node
            node_emb_s, node_emb_t = node_emb.chunk(2, dim=0)
            graph_emb_s, graph_emb_t = graph_emb.chunk(2, dim=0)  # graph_emb [batch_size, dim]
            padding_mask_s, padding_mask_t = batch["padding_mask"].chunk(2, dim=0)  # [batch_size, max_node_num]
            node_pos_s = torch.where(padding_mask_s == False)  # [(batch_size, node_index)]
            local_feat_s = node_emb_s[node_pos_s]  # [total_nodes, dim]
            node_pos_t = torch.where(padding_mask_t == False)  # [(batch_size, node_index)]
            local_feat_t = node_emb_t[node_pos_t]  # [total_nodes, dim]

            # node feature extract------------------------------------------------------
            padding_mask_s_ = ~padding_mask_s
            num_nodes_per_graph_s = torch.sum(padding_mask_s_.long(), dim=-1)  # [batch_size]
            graph_emb_s_repeat = graph_emb_s.repeat_interleave(num_nodes_per_graph_s, dim=0).to(graph_emb.device)
            node_feat_s = torch.cat((local_feat_s, graph_emb_s_repeat), dim=1)  # node_feat_s [total_nodes, dim]

            padding_mask_t_ = ~padding_mask_t
            num_nodes_per_graph_t = torch.sum(padding_mask_t_.long(), dim=-1)  # [batch_size]
            graph_emb_t_repeat = graph_emb_t.repeat_interleave(num_nodes_per_graph_t, dim=0).to(graph_emb.device)
            node_feat_t = torch.cat((local_feat_t, graph_emb_t_repeat), dim=1)  # node_feat_t [total_nodes, dim]

            # node classifier-------------------------------------------------------------
            node_seg_s = self.classifier(node_feat_s)  # [total_nodes, num_classes]
            node_seg_t = self.classifier(node_feat_t)  # [total_nodes, num_classes]

            # known-unknown classifier
            dp_target = self.discriminator_t(node_feat_t).detach()  # [total_nodes, 2]  as weight  {known, unknown}
            weight_dp = dp_target[:, 0]
            # weight_dp = torch.where(weight_dp < 0.2, 0.0, weight_dp)
            # weight_0 = self.discriminator_p(node_feat_t).detach()  # [total_nodes, num_classes]
            # weight_dp = torch.max(weight_0, dim=-1)[0]  # p2 [total_nodes]

            # source classify loss----------------------------------------------------------
            num_node_s = node_seg_s.size()[0]
            label_s = batch["label_feature"][:num_node_s].long() - 1
            label_s = F.one_hot(label_s, self.num_classes)
            loss_s = CrossEntropyLoss(label_s, node_seg_s)
            self.log("train_loss_Gy", loss_s, on_step=False, on_epoch=True)

            # domain_adv loss----------------------------------------------------------------
            num_node_t = node_feat_t.size()[0]
            max_num_node = max(num_node_s, num_node_t)
            feat_s = torch.zeros([max_num_node, node_feat_s.size()[-1]], device=node_feat_s.device, dtype=node_feat_s.dtype)
            feat_s[:num_node_s] = node_feat_s[:]
            feat_t = torch.zeros([max_num_node, node_feat_t.size()[-1]], device=node_feat_t.device, dtype=node_feat_t.dtype)
            feat_t[:num_node_t] = node_feat_t[:]
            weight_s = torch.zeros([max_num_node], device=node_feat_s.device, dtype=node_feat_s.dtype)
            weight_s[:num_node_s] = 1.0
            weight_t = torch.zeros([max_num_node], device=node_feat_t.device, dtype=node_feat_t.dtype)
            weight_t[:num_node_t] = weight_dp[:]
            loss_adv = self.discriminator(feat_s, feat_t, weight_s, weight_t)
            domain_acc = self.discriminator.domain_discriminator_accuracy
            self.log("train_loss_Gd", loss_adv, on_step=False, on_epoch=True)
            self.log("train_transfer_acc", domain_acc, on_step=False, on_epoch=True)

            # target entropy loss----------------------------------------------------------
            loss_entropy = EntropyLoss(node_seg_t, instance_level_weight=weight_dp.contiguous())
            self.log("train_loss_en", loss_adv, on_step=False, on_epoch=True)

            loss = loss_s + 0.3*loss_adv + 0.1*loss_entropy
            self.log("train_loss", loss, on_step=False, on_epoch=True)
            torch.cuda.empty_cache()

            opt_step_2_1.zero_grad()
            self.manual_backward(loss)
            opt_step_2_1.step()


    def validation_step(self, batch, batch_idx):
        self.brep_encoder.eval()
        self.classifier.eval()
        self.discriminator_p.eval()
        self.discriminator_t.eval()
        self.discriminator.eval()

        # graph encoder-----------------------------------------------------------
        node_emb, graph_emb = self.brep_encoder(batch, last_state_only=True)  # graph_emb [batch_size, dim]

        # separate source-target data----------------------------------------------
        node_emb = node_emb[0].permute(1, 0, 2)  # node_emb [batch_size, max_node_num+1, dim] with global node
        node_emb = node_emb[:, 1:, :]  # node_emb [batch_size, max_node_num, dim] without global node
        node_emb_s, node_emb_t = node_emb.chunk(2, dim=0)
        graph_emb_s, graph_emb_t = graph_emb.chunk(2, dim=0)  # graph_emb [batch_size, dim]
        padding_mask_s, padding_mask_t = batch["padding_mask"].chunk(2, dim=0)  # [batch_size, max_node_num]
        node_pos_s = torch.where(padding_mask_s == False)  # [(batch_size, node_index)]
        local_feat_s = node_emb_s[node_pos_s]  # [total_nodes, dim]
        node_pos_t = torch.where(padding_mask_t == False)  # [(batch_size, node_index)]
        local_feat_t = node_emb_t[node_pos_t]  # [total_nodes, dim]
        # node feature extract------------------------------------------------
        padding_mask_s_ = ~padding_mask_s
        num_nodes_per_graph_s = torch.sum(padding_mask_s_.long(), dim=-1)  # [batch_size]
        graph_emb_s_repeat = graph_emb_s.repeat_interleave(num_nodes_per_graph_s, dim=0).to(graph_emb.device)
        node_feat_s = torch.cat((local_feat_s, graph_emb_s_repeat), dim=1)  # node_feat_s [total_nodes, dim]
        padding_mask_t_ = ~padding_mask_t
        num_nodes_per_graph_t = torch.sum(padding_mask_t_.long(), dim=-1)  # [batch_size]
        graph_emb_t_repeat = graph_emb_t.repeat_interleave(num_nodes_per_graph_t, dim=0).to(graph_emb.device)
        node_feat_t = torch.cat((local_feat_t, graph_emb_t_repeat), dim=1)  # node_feat_t [total_nodes, dim]

        # node classifier--------------------------------------------------------------
        node_seg_s = self.classifier(node_feat_s)  # [total_nodes, num_classes]
        node_seg_t = self.classifier(node_feat_t)  # [total_nodes, num_classes]

        # source classify loss---------------------------------------------------------
        num_node_s = node_seg_s.size()[0]
        label_s = batch["label_feature"][:num_node_s].long() - 1
        label_s_ = F.one_hot(label_s, self.num_classes)
        loss_s = CrossEntropyLoss(label_s_, node_seg_s)
        self.log("eval_loss_s", loss_s, on_step=False, on_epoch=True)

        # source multi-classify loss--------------------------------------------------
        p0 = self.discriminator_p(node_feat_s)  # p0 [total_nodes, num_classes]
        loss_sp = BCELossForMultiClassification(label_s_, p0)
        self.log("eval_loss_Gc", loss_sp, on_step=False, on_epoch=True)

        # source pre_face_acc-----------------------------------
        pred_s = torch.argmax(node_seg_s, dim=-1)  # pres [total_nodes]
        pred_s_np = pred_s.long().detach().cpu().numpy()
        label_s_np = label_s.long().detach().cpu().numpy()
        per_face_comp1 = (pred_s_np == label_s_np).astype(np.int)
        self.per_face_acc_source.append(np.mean(per_face_comp1))

        # target pre_face_acc-----------------------------------
        pred_t = torch.argmax(node_seg_t, dim=-1)  # pres [total_nodes]
        pred_t_np = pred_t.long().detach().cpu().numpy()
        label_t = batch["label_feature"][num_node_s:]  # labels [total_nodes]
        label_unk = (self.num_classes + 1) * torch.ones_like(label_t)
        label_t = torch.where(label_t > self.num_classes, label_unk, label_t) - 1
        label_t_np = label_t.long().detach().cpu().numpy()
        per_face_comp2 = (pred_t_np == label_t_np).astype(np.int)
        self.per_face_acc_target.append(np.mean(per_face_comp2))

        # known classify acc-------------------------------------------------------
        known_pos = np.where(label_t_np < self.num_classes)
        known_pred_t_np = pred_t_np[known_pos]
        known_label_t_np = label_t_np[known_pos]
        per_face_comp3 = (known_pred_t_np == known_label_t_np).astype(np.int)
        self.known_classes_acc.append(np.mean(per_face_comp3))

        # unknown-known classify acc-------------------------------------------------------
        dp_target = self.discriminator_t(node_feat_t)  # [total_nodes, 2] {known, unknown}
        pred_tt = torch.argmax(dp_target, dim=-1)      # [total_nodes] {0,1}
        pred_tt_np = pred_tt.long().detach().cpu().numpy()
        label_ones = torch.ones_like(pred_tt)
        label_zeros = torch.zeros_like(pred_tt)
        label_tt = torch.where(label_t >= self.num_classes, label_ones, label_zeros)
        label_tt_np = label_tt.long().detach().cpu().numpy()
        per_face_comp4 = (pred_tt_np == label_tt_np).astype(np.int)
        self.sta_acc.append(np.mean(per_face_comp4))

        # unknown classify acc-----------------------------------------------------------------
        unknown_pos = np.where(label_t_np >= self.num_classes)
        unknown_pred_t_np = pred_tt_np[unknown_pos]
        unknown_label_t_np = label_tt_np[unknown_pos]
        per_face_comp5 = (unknown_pred_t_np == unknown_label_t_np).astype(np.int)
        self.unknown_classes_acc.append(np.mean(per_face_comp5))

        # Visualization of the features ------------------------------
        # if self.current_epoch % 10 >= 5:
        #     os.makedirs("/home/zhang/datasets_segmentation/latent_z/target_%s" % self.current_epoch, exist_ok=True)
        #     feture_np = node_feat_t.detach().cpu().numpy()
        #     json_root = {}
        #     json_root["node_feature"] = feture_np.tolist()
        #     json_root["gt_label"] = label_t_np.tolist()
        #     json_root["pred_label"] = pred_t_np.tolist()
        #     output_path = pathlib.Path("/home/zhang/datasets_segmentation/latent_z/target_%s" % self.current_epoch)
        #     file_name = "latent_z_%s.json" % (batch_idx)
        #     binfile_path = os.path.join(output_path, file_name)
        #     with open(binfile_path, 'w', encoding='utf-8') as fp:
        #         json.dump(json_root, fp, indent=4)
        #
        #     os.makedirs("/home/zhang/datasets_segmentation/latent_z/source_%s" % self.current_epoch,  exist_ok=True)
        #     feture_np = node_feat_s.detach().cpu().numpy()
        #     json_root = {}
        #     json_root["node_feature"] = feture_np.tolist()
        #     json_root["gt_label"] = label_s_np.tolist()
        #     json_root["pred_label"] = pred_s_np.tolist()
        #     output_path = pathlib.Path("/home/zhang/datasets_segmentation/latent_z/source_%s" % self.current_epoch)
        #     file_name = "latent_z_%s.json" % (batch_idx)
        #     binfile_path = os.path.join(output_path, file_name)
        #     with open(binfile_path, 'w', encoding='utf-8') as fp:
        #         json.dump(json_root, fp, indent=4)
        # Visualization of the features ------------------------------

        loss = loss_s + loss_sp
        # self.log("eval_loss", loss, on_step=False, on_epoch=True)
        self.log("eval_loss", 1.0/(np.mean(per_face_comp3)+np.mean(per_face_comp5)), on_step=False, on_epoch=True)
        torch.cuda.empty_cache()
        return loss

    def validation_epoch_end(self, val_step_outputs):
        self.log("per_face_accuracy_source", np.mean(self.per_face_acc_source))
        self.log("per_face_accuracy_target", np.mean(self.per_face_acc_target))
        self.log("known_classes_accuracy", np.mean(self.known_classes_acc))
        self.log("unknown_classes_accuracy", np.mean(self.unknown_classes_acc))
        self.log("separation_accuracy", np.mean(self.sta_acc))
        self.per_face_acc_source = []
        self.per_face_acc_target = []
        self.known_classes_acc = []
        self.unknown_classes_acc = []
        self.sta_acc = []


    def test_step(self, batch, batch_idx):
        self.brep_encoder.eval()
        self.classifier.eval()
        self.discriminator_p.eval()
        self.discriminator_t.eval()
        self.discriminator.eval()

        # graph encoder-----------------------------------------------------------
        node_emb, graph_emb = self.brep_encoder(batch, last_state_only=True)  # graph_emb [batch_size, dim]

        # separate source-target data----------------------------------------------
        node_emb = node_emb[0].permute(1, 0, 2)  # node_emb [batch_size, max_node_num+1, dim] with global node
        node_emb = node_emb[:, 1:, :]  # node_emb [batch_size, max_node_num, dim] without global node
        node_emb_s, node_emb_t = node_emb.chunk(2, dim=0)
        graph_emb_s, graph_emb_t = graph_emb.chunk(2, dim=0)
        padding_mask_s, padding_mask_t = batch["padding_mask"].chunk(2, dim=0)  # [batch_size, max_node_num]
        node_pos_s = torch.where(padding_mask_s == False)  # [(batch_size, node_index)]
        local_feat_s = node_emb_s[node_pos_s]  # [total_nodes, dim]
        node_pos_t = torch.where(padding_mask_t == False)  # [(batch_size, node_index)]
        local_feat_t = node_emb_t[node_pos_t]    # [total_nodes, dim]

        # local-global feature
        padding_mask_s_ = ~padding_mask_s
        num_nodes_per_graph_s = torch.sum(padding_mask_s_.long(), dim=-1)  # [batch_size]
        graph_emb_s_repeat = graph_emb_s.repeat_interleave(num_nodes_per_graph_s, dim=0).to(graph_emb.device)
        node_feat_s = torch.cat((local_feat_s, graph_emb_s_repeat), dim=1)  # node_feat_s [total_nodes, dim]

        padding_mask_t_ = ~padding_mask_t
        num_nodes_per_graph_t = torch.sum(padding_mask_t_.long(), dim=-1)  # [batch_size]
        graph_emb_t_repeat = graph_emb_t.repeat_interleave(num_nodes_per_graph_t, dim=0).to(graph_emb.device)
        node_feat_t = torch.cat((local_feat_t, graph_emb_t_repeat), dim=1)  # node_feat_t [total_nodes, dim]

        # node classifier--------------------------------------------------------------
        node_seg_s = self.classifier(node_feat_s)  # [total_nodes, num_classes + 1]
        node_seg_t = self.classifier(node_feat_t)  # [total_nodes, num_classes + 1]

        pred_t = torch.argmax(node_seg_t, dim=-1) + 1  # pres [total_nodes] {1,...,self.num_classes+1}
        num_node_s = node_seg_s.size()[0]
        label_t = batch["label_feature"][num_node_s:]  # labels [total_nodes]
        label_unk = (self.num_classes + 1) * torch.ones_like(label_t)
        label_t = torch.where(label_t > self.num_classes, label_unk, label_t)

        pred_t_np = pred_t.long().detach().cpu().numpy()
        label_t_np = label_t.long().detach().cpu().numpy()
        per_face_comp = (pred_t_np == label_t_np).astype(np.int)
        self.per_face_acc_target.append(np.mean(per_face_comp))

        known_pos = np.where(label_t_np < self.num_classes + 1)
        known_pred_t_np = pred_t_np[known_pos]
        known_label_t_np = label_t_np[known_pos]
        per_face_comp = (known_pred_t_np == known_label_t_np).astype(np.int)
        self.known_classes_acc.append(np.mean(per_face_comp))

        dp_target = self.discriminator_t(node_feat_t)  # [total_nodes, 2] {known, unknown}
        pred_tt = torch.argmax(dp_target, dim=-1)  # [total_nodes] {0,1}
        pred_tt_np = pred_tt.long().detach().cpu().numpy()
        unknown_pos = np.where(label_t_np >= self.num_classes)
        unknown_pred_t_np = pred_tt_np[unknown_pos]
        label_ones = torch.ones_like(pred_tt)
        label_ones_np = label_ones.long().detach().cpu().numpy()
        unknown_label_t_np = label_ones_np[unknown_pos]
        per_face_comp = (unknown_pred_t_np == unknown_label_t_np).astype(np.int)
        self.unknown_classes_acc.append(np.mean(per_face_comp))

        # 将结果转为txt文件--------------------------------------------------------------------------
        n_graph, max_n_node = padding_mask_t.size()[:2]
        face_feature = -1 * torch.ones([n_graph, max_n_node], device=self.device, dtype=torch.long)
        face_feature[node_pos_t] = pred_t[:]
        out_face_feature = face_feature.long().detach().cpu().numpy()  # [n_graph, max_n_node]
        for i in range(n_graph):
            # 计算每个graph的实际n_node
            end_index = max_n_node - np.sum((out_face_feature[i][:] == -1).astype(np.int))
            # masked出实际face feature
            pred_feature = out_face_feature[i][:end_index + 1]  # (n_node)

            output_path = pathlib.Path("/home/zhang/datasets_segmentation/val")
            file_name = "feature_" + str(batch["id"][n_graph+i].long().detach().cpu().numpy()) + ".txt"
            file_path = os.path.join(output_path, file_name)
            feature_file = open(file_path, mode="a")
            for j in range(end_index):
                feature_file.write(str(pred_feature[j]))
                feature_file.write("\n")
            feature_file.close()

        # Visualization of the features ------------------------------
        feture_np = node_feat_t.detach().cpu().numpy()
        json_root = {}
        json_root["node_feature"] = feture_np.tolist()
        json_root["gt_label"] = label_t_np.tolist()
        json_root["pred_label"] = pred_t_np.tolist()
        output_path = pathlib.Path("/home/zhang/datasets_segmentation/latent_z")
        file_name = "latent_z_%s.json" % (batch_idx)

        binfile_path = os.path.join(output_path, file_name)
        with open(binfile_path, 'w', encoding='utf-8') as fp:
            json.dump(json_root, fp, indent=4)
        # Visualization of the features ------------------------------

    def test_epoch_end(self, outputs):
        self.log("per_face_accuracy_target", np.mean(self.per_face_acc_target))
        self.log("known_classes_accuracy", np.mean(self.known_classes_acc))
        self.log("unknown_classes_accuracy", np.mean(self.unknown_classes_acc))
        print("per_face_accuracy_target:%s" % (np.mean(self.per_face_acc_target)))
        print("known_classes_accuracy:%s" % (np.mean(self.known_classes_acc)))
        print("unknown_classes_accuracy:%s" % (np.mean(self.unknown_classes_acc)))
        self.per_face_acc_target = []
        self.known_classes_acc = []
        self.unknown_classes_acc = []


    def configure_optimizers(self):
        # step-1-1
        opt_step_1_1 = torch.optim.AdamW(self.discriminator_p.parameters(), lr=0.001, betas=(0.99, 0.999))

        # step-1-2
        opt_step_1_2 = torch.optim.AdamW(self.discriminator_t.parameters(), lr=0.001, betas=(0.99, 0.999))

        # step-2-1
        opt_step_2_1 = torch.optim.AdamW(self.brep_encoder.parameters(), lr=0.00005, betas=(0.99, 0.999))
        opt_step_2_1.add_param_group({'params': self.classifier.parameters(), 'lr': 0.0005, 'betas': (0.99, 0.999)})
        opt_step_2_1.add_param_group({'params': self.discriminator.parameters(), 'lr': 0.0005, 'betas': (0.99, 0.999)})

        return opt_step_1_1, opt_step_1_2, opt_step_2_1


    # 逐渐增大学习率
    def optimizer_step(self,
                       epoch,
                       batch_idx,
                       optimizer,
                       optimizer_idx,
                       optimizer_closure,
                       on_tpu,
                       using_native_amp,
                       using_lbfgs,
                       ):
        # update params
        optimizer.step(closure=optimizer_closure)

        # manually warm up lr without a scheduler
        if self.trainer.global_step < 1000:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 1000.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * 0.001