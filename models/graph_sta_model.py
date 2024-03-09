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

#branch graph sta

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


class BinaryClassifier(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, 256),
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
                # nn.Linear(in_dim, 512),
                # nn.BatchNorm1d(512),
                # nn.LeakyReLU(0.2, inplace=True),
                # nn.Linear(512, 256),
                # nn.BatchNorm1d(256),
                # nn.LeakyReLU(0.2, inplace=True),
                # nn.Linear(256, 1),
                # nn.Sigmoid()

                nn.Linear(in_dim, 256),
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

def minmaxscale(x):
    min_value = torch.min(x)
    max_value = torch.max(x)
    return (x - min_value) / (max_value - min_value)


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


class GraphSTADomainAdapt(pl.LightningModule):
    """
    PyTorch Lightning module to train/test the model.
    """
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = args.num_classes
        self.close_set = args.close_set
        self.pre_train = args.pre_checkpoint

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
        pre_trained_model = BrepSeg.load_from_checkpoint(self.pre_train)
        self.brep_encoder = pre_trained_model.brep_encoder
        # garph encoder--------------------------------------------------------------------

        # node classifier------------------------------------------------------------------
        self.classifier = NonLinearClassifier(2*args.dim_node, args.num_classes+1, args.dropout)
        # node classifier------------------------------------------------------------------

        # multi classes classifier----------------------------------------------------------
        self.discriminator_multi = MultiClassifier(2*args.dim_node, args.num_classes)

        # known-unknown classifier----------------------------------------------------------
        self.discriminator_node = BinaryClassifier(args.dim_node)
        self.discriminator_graph = BinaryClassifier(args.dim_node)

        # node domain discriminator--------------------------------------------------------------
        self.domain_discri_node= DomainDiscriminator(args.dim_node, hidden_size=512)
        self.discriminator_domain_node = DomainAdversarialLoss(self.domain_discri_node)
        # graph domain discriminator--------------------------------------------------------------
        self.domain_discri_graph = DomainDiscriminator(args.dim_node, hidden_size=512)
        self.discriminator_domain_graph = DomainAdversarialLoss(self.domain_discri_graph)

        # for close set
        self.per_face_acc_source = []
        self.per_face_acc_target = []
        self.per_class_acc = []
        self.IoU = []

        # for open set
        self.all_class_acc = []
        self.OS_acc = []
        self.OS1_acc = []
        self.unk_acc = []

        # for test
        self.pred = []
        self.label = []
        self.pred_unk = []

        self.automatic_optimization = False


    def training_step(self, batch, batch_idx):
        opt_step_1_1, opt_step_1_2, opt_step_2_1 = self.optimizers()

        # =========================pre-train the multi-binary classifier
        if self.current_epoch % 10 < 10:
            self.brep_encoder.eval()
            self.classifier.eval()
            self.discriminator_multi.train()
            self.discriminator_node.eval()
            self.discriminator_graph.eval()
            self.discriminator_domain_node.eval()
            self.discriminator_domain_graph.eval()

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
            node_feat_s = torch.cat((local_feat_s, graph_emb_s_repeat), dim=1) # node_feat_s [total_nodes, dim]

            # discriminator_multi--------------------------------------------------------------
            pred_multi = self.discriminator_multi(node_feat_s)  # p0 [total_nodes, num_classes] {[0-1]}

            # multi_classifier loss-------------------------------------------------------------
            num_node_s = node_feat_s.size()[0]
            label_s = batch["label_feature"][:num_node_s].long() - 1
            label_s = F.one_hot(label_s, self.num_classes)
            loss_multi = BCELossForMultiClassification(label_s, pred_multi)
            self.log("train_loss_Gc", loss_multi, on_step=False, on_epoch=True)


            # debug =========================================================================================
            # node_pos_t = torch.where(padding_mask_t == False)  # [(batch_size, node_index)]
            # local_feat_t = node_emb_t[node_pos_t]  # [total_nodes, dim]
            # padding_mask_t_ = ~padding_mask_t
            # num_nodes_per_graph_t = torch.sum(padding_mask_t_.long(), dim=-1)  # [batch_size]
            # graph_emb_t_repeat = graph_emb_t.repeat_interleave(num_nodes_per_graph_t, dim=0).to(graph_emb.device)
            # node_feat_t = torch.cat((local_feat_t, graph_emb_t_repeat), dim=1)  # node_feat_t [total_nodes, dim]
            # # discriminator_multi/similarity weight---------------------------------------
            # pred_multi_t = self.discriminator_multi(node_feat_t).detach()  # [total_nodes, num_classes]
            #
            # num_node_s = node_feat_s.size()[0]
            # label_t = batch["label_feature"][num_node_s:].long()
            # pos_know = torch.where(label_t <= self.num_classes)
            # pos_unk = torch.where(label_t > self.num_classes)
            # # print("===================================")
            # # print(torch.mean(pred_multi_t_sum[pos_know]))
            # # print(torch.mean(torch.mean(pred_multi_t[pos_know], dim=-1)))
            # # print(torch.max(pred_multi_t[pos_know], dim=-1)[0])
            # # self.log("pred_known_sum", torch.mean(pred_multi_t_sum[pos_know]), on_step=True, on_epoch=False)
            # self.log("pred_known", torch.mean(torch.mean(pred_multi_t[pos_know], dim=-1)), on_step=True, on_epoch=False)
            # # self.log("pred_known_max", torch.mean(torch.max(pred_multi_t[pos_know], dim=-1)[0]), on_step=True, on_epoch=False)
            #
            # # print("---------------")
            # # print(torch.mean(pred_multi_t_sum[pos_unk]))
            # # print(torch.mean(torch.mean(pred_multi_t[pos_unk], dim=-1)))
            # # print(torch.max(pred_multi_t[pos_unk], dim=-1)[0])
            # # print(torch.mean(pred_multi_t[pos_unk], dim=-1))
            # # self.log("pred_unk_sum", torch.mean(pred_multi_t_sum[pos_unk]), on_step=True, on_epoch=False)
            # self.log("pred_unk", torch.mean(torch.mean(pred_multi_t[pos_unk], dim=-1)), on_step=True, on_epoch=False)
            # # self.log("pred_unk_max", torch.mean(torch.max(pred_multi_t[pos_unk], dim=-1)[0]), on_step=True, on_epoch=False)
            #
            # # weight_node = torch.mean(pred_multi_t, dim=-1)
            # # num_sample = int(0.05 * weight_node.size()[0])
            # # index_known = torch.sort(weight_node, dim=0)[1][:num_sample]  # known node index
            # # index_unknown = torch.sort(weight_node, dim=0)[1][-num_sample:]  # unknown node index
            # # label_t = torch.where(label_t > self.num_classes, self.num_classes+1, label_t)
            # # label_known = torch.index_select(label_t, 0, index_known.view(num_sample)).detach()
            # # label_unknown = torch.index_select(label_t, 0, index_unknown.view(num_sample)).detach()
            # debug ==========================================================================================


            opt_step_1_1.zero_grad()
            self.manual_backward(loss_multi)
            opt_step_1_1.step()
            torch.cuda.empty_cache()

        # =========================pre-train the known/unknown discriminator
        elif self.current_epoch % 10 < 0:
            self.brep_encoder.eval()
            self.classifier.eval()
            self.discriminator_multi.eval()
            self.discriminator_node.train()
            self.discriminator_graph.train()
            self.discriminator_domain_node.eval()
            self.discriminator_domain_graph.eval()

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
            padding_mask_t_ = ~padding_mask_t
            num_nodes_per_graph_t = torch.sum(padding_mask_t_.long(), dim=-1)  # [batch_size]
            graph_emb_t_repeat = graph_emb_t.repeat_interleave(num_nodes_per_graph_t, dim=0).to(graph_emb.device)
            node_feat_t = torch.cat((local_feat_t, graph_emb_t_repeat), dim=1) # node_feat_t [total_nodes, dim]

            # discriminator_multi/similarity weight---------------------------------------
            pred_multi = self.discriminator_multi(node_feat_t)    # [total_nodes, num_classes]
            weight_mean = torch.mean(pred_multi, dim=-1)          # [total_nodes]
            weight_max = torch.max(pred_multi, dim=-1)[0]         # [total_nodes]

            # discriminator_node-----------------------------------------------------------
            num_sample_known = int(0.15 * weight_max.size()[0])
            num_sample_unknown = int(0.025 * weight_mean.size()[0])
            index_known = torch.sort(weight_max.detach(), dim=0)[1][-num_sample_known:]        # known node index
            index_unknown = torch.sort(weight_mean.detach(), dim=0)[1][-num_sample_unknown:]   # unknown node index
            node_feat_known = torch.index_select(local_feat_t, 0, index_known.view(num_sample_known))
            node_feat_unknown = torch.index_select(local_feat_t, 0, index_unknown.view(num_sample_unknown))
            pred_known = self.discriminator_node(node_feat_known)      # known
            pred_unknown = self.discriminator_node(node_feat_unknown)  # unknown
            # known/unknown discriminator loss------------------------------------------------
            label_known = torch.from_numpy(np.concatenate((np.ones((num_sample_known, 1)), np.zeros((num_sample_known, 1))), axis=-1).astype('float32')).to(self.device)
            label_unknown = torch.from_numpy(np.concatenate((np.zeros((num_sample_unknown, 1)), np.ones((num_sample_unknown, 1))), axis=-1).astype('float32')).to(self.device)
            loss_node = CrossEntropyLoss(label_known, pred_known)
            loss_node += CrossEntropyLoss(label_unknown, pred_unknown)
            self.log("train_loss_Gn", loss_node, on_step=False, on_epoch=True)

            # discriminator_graph--------------------------------------------------------------
            n_graph, max_n_node = padding_mask_t.size()[:2]
            weight_graph = torch.zeros([n_graph, max_n_node], device=self.device, dtype=torch.float)
            weight_graph[node_pos_t] = weight_mean[:]   # [n_graph, max_n_node]
            sign_one = torch.sign(weight_graph).int()
            num_non_zeros = torch.count_nonzero(sign_one, dim=1)
            weight_graph = torch.mul(weight_graph.sum(dim=-1), 1.0 / num_non_zeros)  # [n_graph]

            # num_graph_sample = int(0.05 * graph_emb_t.size()[0])
            num_graph_sample = 2
            index_g_known = torch.sort(weight_graph.detach(), dim=0)[1][:num_graph_sample]      # known graph index
            index_g_unknown = torch.sort(weight_graph.detach(), dim=0)[1][-num_graph_sample:]   # unknown graph index
            graph_feat_known = torch.index_select(graph_emb_t, 0, index_g_known.view(num_graph_sample))
            graph_feat_unknown = torch.index_select(graph_emb_t, 0, index_g_unknown.view(num_graph_sample))
            pred_g_known = self.discriminator_graph(graph_feat_known)  # known
            pred_g_unknown = self.discriminator_graph(graph_feat_unknown)  # unknown

            # known/unknown discriminator loss------------------------------------------------
            label_g_known = torch.from_numpy(np.concatenate((np.ones((num_graph_sample, 1)), np.zeros((num_graph_sample, 1))), axis=-1).astype('float32')).to(self.device)
            label_g_unknown = torch.from_numpy(np.concatenate((np.zeros((num_graph_sample, 1)), np.ones((num_graph_sample, 1))), axis=-1).astype('float32')).to(self.device)
            loss_graph = CrossEntropyLoss(label_g_known, pred_g_known)
            loss_graph += CrossEntropyLoss(label_g_unknown, pred_g_unknown)
            self.log("train_loss_Gg", loss_graph, on_step=False, on_epoch=True)


            # debug =========================================================================================
            # num_node_s = local_feat_s.size()[0]
            # label_t = batch["label_feature"][num_node_s:].long()
            # pos_unk = torch.where(label_t > self.num_classes)
            # pos_known = torch.where(label_t <= self.num_classes)
            # weight = self.discriminator_node(node_feat_t)[:, 0]
            # self.log("weight_node_known", torch.mean(weight[pos_known]), on_step=True, on_epoch=False)
            # self.log("weight_node_unknown", torch.mean(weight[pos_unk]), on_step=True, on_epoch=False)
            #
            # label_known = torch.index_select(label_t, 0, index_known.view(num_sample)).detach()
            # label_unknown = torch.index_select(label_t, 0, index_unknown.view(num_sample)).detach()
            #
            # index_1 = torch.sort(weight_node.detach(), dim=0)[1][:num_sample]
            # index_2 = torch.sort(weight_node.detach(), dim=0)[1][num_sample:2*num_sample]
            # index_3 = torch.sort(weight_node.detach(), dim=0)[1][2*num_sample:3*num_sample]
            # index_4 = torch.sort(weight_node.detach(), dim=0)[1][3*num_sample:4*num_sample]
            # index_n1 = torch.sort(weight_node.detach(), dim=0)[1][-num_sample:]
            # index_n2 = torch.sort(weight_node.detach(), dim=0)[1][-2*num_sample:-num_sample]
            # index_n3 = torch.sort(weight_node.detach(), dim=0)[1][-3*num_sample:-2*num_sample]
            # index_n4 = torch.sort(weight_node.detach(), dim=0)[1][-4*num_sample:-3*num_sample]
            #
            # pred_1 = torch.index_select(weight, 0, index_1.view(num_sample))
            # pred_2 = torch.index_select(weight, 0, index_2.view(num_sample))
            # pred_3 = torch.index_select(weight, 0, index_3.view(num_sample))
            # pred_4 = torch.index_select(weight, 0, index_4.view(num_sample))
            # pred_n1 = torch.index_select(weight, 0, index_n1.view(num_sample))
            # pred_n2 = torch.index_select(weight, 0, index_n2.view(num_sample))
            # pred_n3 = torch.index_select(weight, 0, index_n3.view(num_sample))
            # pred_n4 = torch.index_select(weight, 0, index_n4.view(num_sample))
            # self.log("pred_1", torch.mean(pred_1), on_step=True, on_epoch=False)
            # self.log("pred_2", torch.mean(pred_2), on_step=True, on_epoch=False)
            # self.log("pred_3", torch.mean(pred_3), on_step=True, on_epoch=False)
            # self.log("pred_4", torch.mean(pred_4), on_step=True, on_epoch=False)
            # self.log("pred_n1", torch.mean(pred_n1), on_step=True, on_epoch=False)
            # self.log("pred_n2", torch.mean(pred_n2), on_step=True, on_epoch=False)
            # self.log("pred_n3", torch.mean(pred_n3), on_step=True, on_epoch=False)
            # self.log("pred_n4", torch.mean(pred_n4), on_step=True, on_epoch=False)
            # debug =========================================================================================


            loss = loss_node + loss_graph
            self.log("train_loss", loss, on_step=False, on_epoch=True)
            opt_step_1_2.zero_grad()
            self.manual_backward(loss)
            opt_step_1_2.step()
            torch.cuda.empty_cache()

        # =========================train domain discriminator with weight
        # =========================train classifier with unknown-label
        else:
            self.brep_encoder.train()
            self.classifier.train()
            self.discriminator_multi.eval()
            self.discriminator_node.eval()
            self.discriminator_graph.eval()
            self.discriminator_domain_node.train()
            self.discriminator_domain_graph.train()

            # graph encoder-----------------------------------------------------------
            node_emb, graph_emb = self.brep_encoder(batch, last_state_only=True)  # graph_emb [batch_size, dim]

            # separate source-target data----------------------------------------------
            node_emb = node_emb[0].permute(1, 0, 2)  # node_emb [batch_size, max_node_num+1, dim] with global node
            node_emb = node_emb[:, 1:, :]            # node_emb [batch_size, max_node_num, dim] without global node
            node_emb_s, node_emb_t = node_emb.chunk(2, dim=0)
            graph_emb_s, graph_emb_t = graph_emb.chunk(2, dim=0)  # graph_emb [batch_size, dim]
            padding_mask_s, padding_mask_t = batch["padding_mask"].chunk(2, dim=0)  # [batch_size, max_node_num]
            node_pos_s = torch.where(padding_mask_s == False)  # [(batch_size, node_index)]
            local_feat_s = node_emb_s[node_pos_s]  # [total_nodes, dim]
            node_pos_t = torch.where(padding_mask_t == False)  # [(batch_size, node_index)]
            local_feat_t = node_emb_t[node_pos_t]  # [total_nodes, dim]

            # node feature extract------------------------------------------------------
            padding_mask_s_ = ~padding_mask_s
            num_nodes_per_graph_s = torch.sum(padding_mask_s_.long(), dim=-1)   # [batch_size]
            graph_emb_s_repeat = graph_emb_s.repeat_interleave(num_nodes_per_graph_s, dim=0).to(graph_emb.device)
            node_feat_s = torch.cat((local_feat_s, graph_emb_s_repeat), dim=1)  # node_feat_s [total_nodes, dim]
            padding_mask_t_ = ~padding_mask_t
            num_nodes_per_graph_t = torch.sum(padding_mask_t_.long(), dim=-1)   # [batch_size]
            graph_emb_t_repeat = graph_emb_t.repeat_interleave(num_nodes_per_graph_t, dim=0).to(graph_emb.device)
            node_feat_t = torch.cat((local_feat_t, graph_emb_t_repeat), dim=1)  # node_feat_t [total_nodes, dim]

            # node classifier-------------------------------------------------------------
            node_seg_s = self.classifier(node_feat_s)  # [total_nodes, num_classes + 1]
            node_seg_t = self.classifier(node_feat_t)  # [total_nodes, num_classes + 1]

            # known-unknown classifier
            weight_node = self.discriminator_node(local_feat_t).detach()  # [total_nodes, 2]  as weight  {known, unknown}
            weight_node = weight_node[:, 0]    # [total_nodes]
            weight_node = torch.where(weight_node<0.1, 0.0, weight_node)
            weight_node = minmaxscale(weight_node)

            weight_graph = self.discriminator_graph(graph_emb_t).detach()  # [n_graph, 2]  as weight  {known, unknown}
            weight_graph = weight_graph[:, 0]  # [n_graph]
            weight_graph = torch.where(weight_graph<0.1, 0.0, weight_graph)
            weight_graph = minmaxscale(weight_graph)

            weight_graph_repeat = weight_graph.repeat_interleave(num_nodes_per_graph_t, dim=0).to(self.device)
            weight_all = 0.9 * weight_node + 0.1 * weight_graph_repeat
            if(self.close_set):
                weight_all[:] = 1.0
                weight_node[:] = 1.0
                weight_graph[:] = 1.0

            weight_all[:] = 1.0
            weight_node[:] = 1.0
            weight_graph[:] = 1.0

            # source classify loss----------------------------------------------------------
            num_node_s = node_seg_s.size()[0]
            label_s = batch["label_feature"][:num_node_s].long() - 1
            label_s = F.one_hot(label_s, self.num_classes+1)
            loss_s = CrossEntropyLoss(label_s, node_seg_s)
            self.log("train_loss_Gy", loss_s, on_step=False, on_epoch=True)

            # target classify loss for unknown class------------------------------------------
            if(not self.close_set):
                # select possible unknown_label sample
                num_sample = int(0.01 * node_feat_t.size()[0])
                unknown_index = torch.sort(weight_node.detach(), dim=0)[1][:num_sample]
                node_feat_unk = torch.index_select(node_feat_t, 0, unknown_index.view(num_sample))
                node_seg_t_unk = self.classifier(node_feat_unk)    # [num_unknown_sample, num_classes + 1]
                label_unknown = torch.from_numpy(np.concatenate((np.zeros((num_sample, self.num_classes)), np.ones((num_sample, 1))), axis=-1).astype('float32')).to(self.device)
                loss_unk = CrossEntropyLoss(label_unknown, node_seg_t_unk)
                self.log("train_loss_unk", loss_unk, on_step=False, on_epoch=True)

            # domain_adv loss node---------------------------------------------------------------
            num_node_t = node_feat_t.size()[0]
            max_num_node = max(num_node_s, num_node_t)
            feat_s = torch.zeros([max_num_node, local_feat_s.size()[-1]], device=self.device, dtype=local_feat_s.dtype)
            feat_s[:num_node_s] = local_feat_s[:]
            feat_t = torch.zeros([max_num_node, local_feat_t.size()[-1]], device=self.device, dtype=local_feat_t.dtype)
            feat_t[:num_node_t] = local_feat_t[:]
            weight_s = torch.zeros([max_num_node], device=self.device, dtype=local_feat_s.dtype)
            weight_s[:num_node_s] = 1.0
            weight_t = torch.zeros([max_num_node], device=self.device, dtype=local_feat_t.dtype)
            weight_t[:num_node_t] = weight_node[:]
            loss_adv_node = self.discriminator_domain_node(feat_s, feat_t, weight_s, weight_t)
            domain_acc_node = self.discriminator_domain_node.domain_discriminator_accuracy
            self.log("train_loss_Gdn", loss_adv_node, on_step=False, on_epoch=True)
            self.log("train_transfer_acc_n", domain_acc_node, on_step=False, on_epoch=True)

            # domain_adv loss graph----------------------------------------------------------------
            weight_graph_s = torch.ones([graph_emb_s.size()[0]], device=self.device, dtype=weight_graph.dtype)
            loss_adv_graph = self.discriminator_domain_graph(graph_emb_s, graph_emb_t, weight_graph_s, weight_graph)
            domain_acc_graph = self.discriminator_domain_graph.domain_discriminator_accuracy
            self.log("train_loss_Gdg", loss_adv_graph, on_step=False, on_epoch=True)
            self.log("train_transfer_acc_g", domain_acc_graph, on_step=False, on_epoch=True)

            # target entropy loss----------------------------------------------------------
            loss_entropy = EntropyLoss(node_seg_t, instance_level_weight=weight_node.contiguous())
            self.log("train_loss_en", loss_entropy, on_step=False, on_epoch=True)


            # debug =========================================================================================
            # label_t = batch["label_feature"][num_node_s:].long()
            # pos_unk = torch.where(label_t > self.num_classes)
            # pos_known = torch.where(label_t <= self.num_classes)
            # # print(torch.mean(weight_all[pos_unk]))
            # self.log("weight_all_unknown", torch.mean(weight_all[pos_unk]), on_step=True, on_epoch=False)
            # # print(torch.mean(weight_all[pos_known]))
            # self.log("weight_all_known", torch.mean(weight_all[pos_known]), on_step=True, on_epoch=False)
            # debug ==========================================================================================


            if(self.close_set):
                loss = loss_s + 0.3*loss_adv_node + 0.1*loss_adv_graph + 0.1*loss_entropy
            else:
                loss = loss_s + 0.3*loss_adv_node + 0.1*loss_adv_graph + 0.1*loss_entropy + 0.1*loss_unk
            self.log("train_loss", loss, on_step=False, on_epoch=True)

            opt_step_2_1.zero_grad()
            self.manual_backward(loss)
            opt_step_2_1.step()
            torch.cuda.empty_cache()


    def validation_step(self, batch, batch_idx):
        self.brep_encoder.eval()
        self.classifier.eval()
        self.discriminator_multi.eval()
        self.discriminator_node.eval()
        self.discriminator_graph.eval()
        self.discriminator_domain_node.eval()
        self.discriminator_domain_graph.eval()

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
        node_seg_s = self.classifier(node_feat_s)  # [total_nodes, num_classes + 1]
        node_seg_t = self.classifier(node_feat_t)  # [total_nodes, num_classes + 1]

        # source classify loss---------------------------------------------------------
        num_node_s = node_seg_s.size()[0]
        label_s = batch["label_feature"][:num_node_s].long() - 1
        label_s_ = F.one_hot(label_s, self.num_classes + 1)
        loss_s = CrossEntropyLoss(label_s_, node_seg_s)
        self.log("eval_loss_s", loss_s, on_step=False, on_epoch=True)

        # source pre_face_acc----------------------------------------------------------
        pred_s = torch.argmax(node_seg_s, dim=-1)  # pres [total_nodes]
        pred_s_np = pred_s.long().detach().cpu().numpy()
        label_s_np = label_s.long().detach().cpu().numpy()
        per_face_comp1 = (pred_s_np == label_s_np).astype(np.int)
        self.per_face_acc_source.append(np.mean(per_face_comp1))

        # =====close set transfer====
        # target pre_face_acc-----------------------------------------------------------
        pred_t = torch.argmax(node_seg_t, dim=-1)  # pres [total_nodes]
        pred_t_np = pred_t.long().detach().cpu().numpy()
        label_t = batch["label_feature"][num_node_s:]  # labels [total_nodes]
        label_unk = (self.num_classes + 1) * torch.ones_like(label_t)
        label_t = torch.where(label_t > self.num_classes, label_unk, label_t) - 1
        label_t_np = label_t.long().detach().cpu().numpy()
        per_face_comp2 = (pred_t_np == label_t_np).astype(np.int)
        self.per_face_acc_target.append(np.mean(per_face_comp2))

        # pre-class acc------------------------------------------------------------------
        per_class_acc = []
        for i in range(self.num_classes):
            class_pos = np.where(label_t_np == i)
            if len(class_pos[0]) > 0:
                class_i_preds = pred_t_np[class_pos]
                class_i_label = label_t_np[class_pos]
                per_face_comp_ = (class_i_preds == class_i_label).astype(np.int)
                per_class_acc.append(np.mean(per_face_comp_))
        self.per_class_acc.append(np.mean(per_class_acc))

        # IoU----------------------------------------------------------------------------
        per_class_iou = []
        for i in range(self.num_classes):
            label_pos = np.where(label_t_np == i)
            pred_pos = np.where(pred_t_np == i)
            if len(pred_pos[0]) > 0 and len(label_pos[0]) > 0:
                class_i_preds = pred_t_np[label_pos]
                class_i_label = label_t_np[label_pos]
                Intersection = (class_i_preds == class_i_label).astype(np.int)
                Union = (class_i_preds != class_i_label).astype(np.int)
                class_i_preds_ = pred_t_np[pred_pos]
                class_i_label_ = label_t_np[pred_pos]
                Union_ = (class_i_preds_ != class_i_label_).astype(np.int)
                per_class_iou.append(np.sum(Intersection) / (np.sum(Union) + np.sum(Intersection) + np.sum(Union_)))
        self.IoU.append(np.mean(per_class_iou))
        # =====close set transfer====

        # =====open set transfer====
        # all acc----------------------------------------------------------
        per_face_comp3 = (pred_t_np == label_t_np).astype(np.int)
        self.all_class_acc.append(np.mean(per_face_comp3))

        # OS---------------------------------------------------------------
        OS_acc = []
        for i in range(self.num_classes+1):
            class_pos = np.where(label_t_np == i)
            if len(class_pos[0]) > 0:
                class_i_preds = pred_t_np[class_pos]
                class_i_label = label_t_np[class_pos]
                per_face_comp_ = (class_i_preds == class_i_label).astype(np.int)
                OS_acc.append(np.mean(per_face_comp_))
        self.OS_acc.append(np.mean(OS_acc))

        # OS*--------------------------------------------------------------
        OS1_acc = []
        for i in range(self.num_classes):
            class_pos = np.where(label_t_np == i)
            if len(class_pos[0]) > 0:
                class_i_preds = pred_t_np[class_pos]
                class_i_label = label_t_np[class_pos]
                per_face_comp_ = (class_i_preds == class_i_label).astype(np.int)
                OS1_acc.append(np.mean(per_face_comp_))
        self.OS1_acc.append(np.mean(OS1_acc))

        # unk--------------------------------------------------------------
        unknown_pos = np.where(label_t_np >= self.num_classes)
        unknown_pred_t_np = pred_t_np[unknown_pos]
        unknown_label_t_np = label_t_np[unknown_pos]
        per_face_comp4 = (unknown_pred_t_np == unknown_label_t_np).astype(np.int)
        self.unk_acc.append(np.mean(per_face_comp4))
        # =====open set transfer====

        # Visualization of the features -----------------------------------
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

        # debug =========================================================================================
        pos_know = torch.where(label_t < self.num_classes)
        pos_unk = torch.where(label_t >= self.num_classes)

        pred_multi_t = self.discriminator_multi(node_feat_t)  # [total_nodes, num_classes]
        weight_0 = torch.mean(pred_multi_t, dim=-1)           # [total_nodes]
        pred_max = torch.max(pred_multi_t, dim=-1)[0]         # [total_nodes]
        self.log("pred_known", torch.mean(weight_0[pos_know]), on_step=False, on_epoch=True)
        self.log("pred_unk", torch.mean(weight_0[pos_unk]), on_step=False, on_epoch=True)
        self.log("max_known", torch.mean(pred_max[pos_know]), on_step=False, on_epoch=True)
        self.log("max_unk", torch.mean(pred_max[pos_unk]), on_step=False, on_epoch=True)

        num_sample = int(0.05 * weight_0.size()[0])
        index_mean_known = torch.sort(weight_0.detach(), dim=0)[1][:num_sample]  # known node index
        index_mean_unknown = torch.sort(weight_0.detach(), dim=0)[1][-num_sample:]  # unknown node index
        label_mean_known = torch.index_select(label_t, 0, index_mean_known.view(num_sample))
        label_mean_known = torch.where(label_mean_known < self.num_classes, 1.0, 0.0)
        label_mean_unknown = torch.index_select(label_t, 0, index_mean_unknown.view(num_sample))
        label_mean_unknown = torch.where(label_mean_unknown >= self.num_classes, 1.0, 0.0)
        self.log("label_mean_known_50", torch.mean(label_mean_known), on_step=False, on_epoch=True)
        self.log("label_mean_unknown_50", torch.mean(label_mean_unknown), on_step=False, on_epoch=True)
        index_max_known = torch.sort(pred_max.detach(), dim=0)[1][-num_sample:]
        index_max_unknown = torch.sort(pred_max.detach(), dim=0)[1][:num_sample]
        label_max_known = torch.index_select(label_t, 0, index_max_known.view(num_sample))
        label_max_known = torch.where(label_max_known < self.num_classes, 1.0, 0.0)
        label_max_unknown = torch.index_select(label_t, 0, index_max_unknown.view(num_sample))
        label_max_unknown = torch.where(label_max_unknown >= self.num_classes, 1.0, 0.0)
        self.log("label_max_known_50", torch.mean(label_max_known), on_step=False, on_epoch=True)
        self.log("label_max_unknown_50", torch.mean(label_max_unknown), on_step=False, on_epoch=True)

        num_sample = int(0.025 * weight_0.size()[0])
        index_mean_known = torch.sort(weight_0.detach(), dim=0)[1][:num_sample]  # known node index
        index_mean_unknown = torch.sort(weight_0.detach(), dim=0)[1][-num_sample:]  # unknown node index
        label_mean_known = torch.index_select(label_t, 0, index_mean_known.view(num_sample))
        label_mean_known = torch.where(label_mean_known < self.num_classes, 1.0, 0.0)
        label_mean_unknown = torch.index_select(label_t, 0, index_mean_unknown.view(num_sample))
        label_mean_unknown = torch.where(label_mean_unknown >= self.num_classes, 1.0, 0.0)
        self.log("label_mean_known_25", torch.mean(label_mean_known), on_step=False, on_epoch=True)
        self.log("label_mean_unknown_25", torch.mean(label_mean_unknown), on_step=False, on_epoch=True)
        index_max_known = torch.sort(pred_max.detach(), dim=0)[1][-num_sample:]
        index_max_unknown = torch.sort(pred_max.detach(), dim=0)[1][:num_sample]
        label_max_known = torch.index_select(label_t, 0, index_max_known.view(num_sample))
        label_max_known = torch.where(label_max_known < self.num_classes, 1.0, 0.0)
        label_max_unknown = torch.index_select(label_t, 0, index_max_unknown.view(num_sample))
        label_max_unknown = torch.where(label_max_unknown >= self.num_classes, 1.0, 0.0)
        self.log("label_max_known_25", torch.mean(label_max_known), on_step=False, on_epoch=True)
        self.log("label_max_unknown_25", torch.mean(label_max_unknown), on_step=False, on_epoch=True)

        num_sample = int(0.1 * weight_0.size()[0])
        index_mean_known = torch.sort(weight_0.detach(), dim=0)[1][:num_sample]  # known node index
        index_mean_unknown = torch.sort(weight_0.detach(), dim=0)[1][-num_sample:]  # unknown node index
        label_mean_known = torch.index_select(label_t, 0, index_mean_known.view(num_sample))
        label_mean_known = torch.where(label_mean_known < self.num_classes, 1.0, 0.0)
        label_mean_unknown = torch.index_select(label_t, 0, index_mean_unknown.view(num_sample))
        label_mean_unknown = torch.where(label_mean_unknown >= self.num_classes, 1.0, 0.0)
        self.log("label_mean_known_100", torch.mean(label_mean_known), on_step=False, on_epoch=True)
        self.log("label_mean_unknown_100", torch.mean(label_mean_unknown), on_step=False, on_epoch=True)
        index_max_known = torch.sort(pred_max.detach(), dim=0)[1][-num_sample:]
        index_max_unknown = torch.sort(pred_max.detach(), dim=0)[1][:num_sample]
        label_max_known = torch.index_select(label_t, 0, index_max_known.view(num_sample))
        label_max_known = torch.where(label_max_known < self.num_classes, 1.0, 0.0)
        label_max_unknown = torch.index_select(label_t, 0, index_max_unknown.view(num_sample))
        label_max_unknown = torch.where(label_max_unknown >= self.num_classes, 1.0, 0.0)
        self.log("label_max_known_100", torch.mean(label_max_known), on_step=False, on_epoch=True)
        self.log("label_max_unknown_100", torch.mean(label_max_unknown), on_step=False, on_epoch=True)

        num_sample = int(0.2 * weight_0.size()[0])
        index_mean_known = torch.sort(weight_0.detach(), dim=0)[1][:num_sample]  # known node index
        index_mean_unknown = torch.sort(weight_0.detach(), dim=0)[1][-num_sample:]  # unknown node index
        label_mean_known = torch.index_select(label_t, 0, index_mean_known.view(num_sample))
        label_mean_known = torch.where(label_mean_known < self.num_classes, 1.0, 0.0)
        label_mean_unknown = torch.index_select(label_t, 0, index_mean_unknown.view(num_sample))
        label_mean_unknown = torch.where(label_mean_unknown >= self.num_classes, 1.0, 0.0)
        self.log("label_mean_known_200", torch.mean(label_mean_known), on_step=False, on_epoch=True)
        self.log("label_mean_unknown_200", torch.mean(label_mean_unknown), on_step=False, on_epoch=True)
        index_max_known = torch.sort(pred_max.detach(), dim=0)[1][-num_sample:]
        index_max_unknown = torch.sort(pred_max.detach(), dim=0)[1][:num_sample]
        label_max_known = torch.index_select(label_t, 0, index_max_known.view(num_sample))
        label_max_known = torch.where(label_max_known < self.num_classes, 1.0, 0.0)
        label_max_unknown = torch.index_select(label_t, 0, index_max_unknown.view(num_sample))
        label_max_unknown = torch.where(label_max_unknown >= self.num_classes, 1.0, 0.0)
        self.log("label_max_known_200", torch.mean(label_max_known), on_step=False, on_epoch=True)
        self.log("label_max_unknown_200", torch.mean(label_max_unknown), on_step=False, on_epoch=True)


        weight_node = self.discriminator_node(local_feat_t)[:, 0]
        self.log("weight_node_known", torch.mean(weight_node[pos_know]), on_step=False, on_epoch=True)
        self.log("weight_node_unknown", torch.mean(weight_node[pos_unk]), on_step=False, on_epoch=True)

        num_sample = int(0.05 * weight_node.size()[0])
        index_1 = torch.sort(weight_0.detach(), dim=0)[1][:num_sample]
        index_2 = torch.sort(weight_0.detach(), dim=0)[1][num_sample:2 * num_sample]
        index_3 = torch.sort(weight_0.detach(), dim=0)[1][2 * num_sample:3 * num_sample]
        index_4 = torch.sort(weight_0.detach(), dim=0)[1][3 * num_sample:4 * num_sample]
        index_n1 = torch.sort(weight_0.detach(), dim=0)[1][-num_sample:]
        index_n2 = torch.sort(weight_0.detach(), dim=0)[1][-2 * num_sample:-num_sample]
        index_n3 = torch.sort(weight_0.detach(), dim=0)[1][-3 * num_sample:-2 * num_sample]
        index_n4 = torch.sort(weight_0.detach(), dim=0)[1][-4 * num_sample:-3 * num_sample]

        pred_1 = torch.index_select(weight_node, 0, index_1.view(num_sample))
        pred_2 = torch.index_select(weight_node, 0, index_2.view(num_sample))
        pred_3 = torch.index_select(weight_node, 0, index_3.view(num_sample))
        pred_4 = torch.index_select(weight_node, 0, index_4.view(num_sample))
        pred_n1 = torch.index_select(weight_node, 0, index_n1.view(num_sample))
        pred_n2 = torch.index_select(weight_node, 0, index_n2.view(num_sample))
        pred_n3 = torch.index_select(weight_node, 0, index_n3.view(num_sample))
        pred_n4 = torch.index_select(weight_node, 0, index_n4.view(num_sample))
        self.log("pred_1", torch.mean(pred_1), on_step=False, on_epoch=True)
        self.log("pred_2", torch.mean(pred_2), on_step=False, on_epoch=True)
        self.log("pred_3", torch.mean(pred_3), on_step=False, on_epoch=True)
        self.log("pred_4", torch.mean(pred_4), on_step=False, on_epoch=True)
        self.log("pred_n1", torch.mean(pred_n1), on_step=False, on_epoch=True)
        self.log("pred_n2", torch.mean(pred_n2), on_step=False, on_epoch=True)
        self.log("pred_n3", torch.mean(pred_n3), on_step=False, on_epoch=True)
        self.log("pred_n4", torch.mean(pred_n4), on_step=False, on_epoch=True)


        weight_graph = self.discriminator_graph(graph_emb_t)[:, 0]  # [n_graph]
        weight_graph_repeat = weight_graph.repeat_interleave(num_nodes_per_graph_t, dim=0).to(self.device)
        weight_all = 0.8 * weight_node + 0.2 * weight_graph_repeat
        self.log("weight_all_unknown", torch.mean(weight_all[pos_unk]), on_step=False, on_epoch=True)
        self.log("weight_all_known", torch.mean(weight_all[pos_know]), on_step=False, on_epoch=True)
        # debug ==========================================================================================

        loss = loss_s
        # self.log("eval_loss", loss, on_step=False, on_epoch=True)
        self.log("eval_loss", 1.0/np.mean(per_face_comp2), on_step=False, on_epoch=True)
        torch.cuda.empty_cache()
        return loss

    def validation_epoch_end(self, val_step_outputs):
        self.log("eval_per_face_accuracy_source", np.mean(self.per_face_acc_source))
        self.log("eval_per_face_accuracy_target", np.mean(self.per_face_acc_target))
        self.log("eval_per_class_accuracy", np.mean(self.per_class_acc))
        self.log("eval_IoU", np.mean(self.IoU))

        self.log("eval_ALL", np.mean(self.all_class_acc))
        self.log("eval_OS", np.mean(self.OS_acc))
        self.log("eval_OS*", np.mean(self.OS1_acc))
        self.log("eval_UNK", np.mean(self.unk_acc))

        self.per_face_acc_source = []
        self.per_face_acc_target = []
        self.per_class_acc = []
        self.IoU = []

        self.all_class_acc =[]
        self.OS_acc = []
        self.OS1_acc = []
        self.unk_acc = []


    def test_step(self, batch, batch_idx):
        self.brep_encoder.eval()
        self.classifier.eval()
        self.discriminator_multi.eval()
        self.discriminator_node.eval()
        self.discriminator_graph.eval()
        self.discriminator_domain_node.eval()
        self.discriminator_domain_graph.eval()

        # graph encoder-----------------------------------------------------------
        node_emb, graph_emb = self.brep_encoder(batch, last_state_only=True)  # graph_emb [batch_size, dim]

        # separate source-target data----------------------------------------------
        node_emb = node_emb[0].permute(1, 0, 2)  # node_emb [batch_size, max_node_num+1, dim] with global node
        node_emb = node_emb[:, 1:, :]            # node_emb [batch_size, max_node_num, dim] without global node
        node_emb_s, node_emb_t = node_emb.chunk(2, dim=0)
        graph_emb_s, graph_emb_t = graph_emb.chunk(2, dim=0)
        padding_mask_s, padding_mask_t = batch["padding_mask"].chunk(2, dim=0)  # [batch_size, max_node_num]
        node_pos_s = torch.where(padding_mask_s == False)  # [(batch_size, node_index)]
        local_feat_s = node_emb_s[node_pos_s]  # [total_nodes, dim]
        node_pos_t = torch.where(padding_mask_t == False)  # [(batch_size, node_index)]
        local_feat_t = node_emb_t[node_pos_t]  # [total_nodes, dim]

        # local-global feature
        padding_mask_s_ = ~padding_mask_s
        num_nodes_per_graph_s = torch.sum(padding_mask_s_.long(), dim=-1)   # [batch_size]
        graph_emb_s_repeat = graph_emb_s.repeat_interleave(num_nodes_per_graph_s, dim=0).to(graph_emb.device)
        node_feat_s = torch.cat((local_feat_s, graph_emb_s_repeat), dim=1)  # node_feat_s [total_nodes, dim]
        padding_mask_t_ = ~padding_mask_t
        num_nodes_per_graph_t = torch.sum(padding_mask_t_.long(), dim=-1)   # [batch_size]
        graph_emb_t_repeat = graph_emb_t.repeat_interleave(num_nodes_per_graph_t, dim=0).to(graph_emb.device)
        node_feat_t = torch.cat((local_feat_t, graph_emb_t_repeat), dim=1)  # node_feat_t [total_nodes, dim]

        # node classifier--------------------------------------------------------------
        node_seg_s = self.classifier(node_feat_s)  # [total_nodes, num_classes + 1]
        node_seg_t = self.classifier(node_feat_t)  # [total_nodes, num_classes + 1]
        num_node_s = node_seg_s.size()[0]

        # =====close set transfer====
        pred_t = torch.argmax(node_seg_t, dim=-1) + 1  # pres [total_nodes]
        pred_t_np = pred_t.long().detach().cpu().numpy()
        label_t = batch["label_feature"][num_node_s:]  # labels [total_nodes]
        label_unk = (self.num_classes + 1) * torch.ones_like(label_t)
        label_t = torch.where(label_t > self.num_classes, label_unk, label_t)
        label_t_np = label_t.long().detach().cpu().numpy()

        pred_s = torch.argmax(node_seg_s, dim=-1) + 1  # pres [total_nodes]
        pred_s_np = pred_s.long().detach().cpu().numpy()
        label_s = batch["label_feature"][:num_node_s]  # labels [total_nodes]
        label_s_np = label_s.long().detach().cpu().numpy()

        for i in range(len(pred_t_np)): self.pred.append(pred_t_np[i])
        for i in range(len(label_t_np)): self.label.append(label_t_np[i])

        # =====open set transfer====
        for i in range(len(pred_t_np)): self.pred_unk.append(pred_t_np[i])


        # debug =========================================================================================
        # # discriminator_multi/similarity weight---------------------------------------
        # pred_multi = self.discriminator_multi(node_feat_t).detach()  # [total_nodes, num_classes]
        # pred_multi_sum = torch.sum(pred_multi, dim=-1)
        # weight_node = torch.max(pred_multi, dim=-1)[0]  # [total_nodes]
        # pos_unk = torch.where(label_t == (self.num_classes + 1))
        # print(pred_multi[pos_unk])
        # print(pred_multi_sum[pos_unk])
        #
        # # discriminator_node-----------------------------------------------------------
        # num_known_sample = int(0.05 * weight_node.size()[0])
        # w = torch.sort(weight_node, dim=0)[1][-num_known_sample:]  # known node feature
        # h = torch.sort(weight_node, dim=0)[1][:num_known_sample]  # unknown node feature
        # node_feat_known = torch.index_select(pred_multi, 0, w.view(num_known_sample)).detach()
        # node_feat_unknown = torch.index_select(pred_multi, 0, h.view(num_known_sample)).detach()
        # debug ==========================================================================================


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
        # feture_np = node_feat_t.detach().cpu().numpy()
        # json_root = {}
        # json_root["node_feature"] = feture_np.tolist()
        # json_root["gt_label"] = label_t_np.tolist()
        # json_root["pred_label"] = pred_t_np.tolist()
        # output_path = pathlib.Path("/home/zhang/datasets_segmentation/latent_z")
        # file_name = "latent_z_%s.json" % (batch_idx)
        # binfile_path = os.path.join(output_path, file_name)
        # with open(binfile_path, 'w', encoding='utf-8') as fp:
        #     json.dump(json_root, fp, indent=4)
        # Visualization of the features ------------------------------

        # Visualization of the features -----------------------------------
        os.makedirs("/home/zhang/datasets_segmentation/latent_z/target", exist_ok=True)
        feture_np = node_feat_t.detach().cpu().numpy()
        json_root = {}
        json_root["node_feature"] = feture_np.tolist()
        json_root["gt_label"] = label_t_np.tolist()
        json_root["pred_label"] = pred_t_np.tolist()
        output_path = pathlib.Path("/home/zhang/datasets_segmentation/latent_z/target")
        file_name = "latent_z_%s.json" % (batch_idx)
        binfile_path = os.path.join(output_path, file_name)
        with open(binfile_path, 'w', encoding='utf-8') as fp:
            json.dump(json_root, fp, indent=4)

        os.makedirs("/home/zhang/datasets_segmentation/latent_z/source",  exist_ok=True)
        feture_np = node_feat_s.detach().cpu().numpy()
        json_root = {}
        json_root["node_feature"] = feture_np.tolist()
        json_root["gt_label"] = label_s_np.tolist()
        json_root["pred_label"] = pred_s_np.tolist()
        output_path = pathlib.Path("/home/zhang/datasets_segmentation/latent_z/source")
        file_name = "latent_z_%s.json" % (batch_idx)
        binfile_path = os.path.join(output_path, file_name)
        with open(binfile_path, 'w', encoding='utf-8') as fp:
            json.dump(json_root, fp, indent=4)
        # Visualization of the features ------------------------------

    def test_epoch_end(self, outputs):

        pred_t_np = np.array(self.pred)
        label_t_np = np.array(self.label)

        # per-face acc------------------------------------------------------------------
        per_face_comp1 = (pred_t_np == label_t_np).astype(np.int)
        self.log("eval_per_face_accuracy", np.mean(per_face_comp1))
        print("per_face_accuracy:%s" % np.mean(per_face_comp1))

        # per-class acc------------------------------------------------------------------
        per_class_acc = []
        for i in range(1, self.num_classes + 1):
            class_pos = np.where(label_t_np == i)
            if len(class_pos[0]) > 0:
                class_i_preds = pred_t_np[class_pos]
                class_i_label = label_t_np[class_pos]
                per_face_comp_ = (class_i_preds == class_i_label).astype(np.int)
                per_class_acc.append(np.mean(per_face_comp_))
        self.log("eval_per_class_accuracy", np.mean(per_class_acc))
        print("per_class_accuracy:%s" % np.mean(per_class_acc))

        # IoU----------------------------------------------------------------------------
        per_class_iou = []
        for i in range(1, self.num_classes + 1):
            label_pos = np.where(label_t_np == i)
            pred_pos = np.where(pred_t_np == i)
            if len(pred_pos[0]) > 0 and len(label_pos[0]) > 0:
                class_i_preds = pred_t_np[label_pos]
                class_i_label = label_t_np[label_pos]
                Intersection = (class_i_preds == class_i_label).astype(np.int)
                Union = (class_i_preds != class_i_label).astype(np.int)
                class_i_preds_ = pred_t_np[pred_pos]
                class_i_label_ = label_t_np[pred_pos]
                Union_ = (class_i_preds_ != class_i_label_).astype(np.int)
                per_class_iou.append(np.sum(Intersection) / (np.sum(Union) + np.sum(Intersection) + np.sum(Union_)))
        self.log("eval_IoU", np.mean(per_class_iou))
        print("IoU:%s" % np.mean(per_class_iou))

        pred_t_np = np.array(self.pred_unk)
        # all acc----------------------------------------------------------
        per_face_comp2 = (pred_t_np == label_t_np).astype(np.int)
        self.log("eval_ALL", np.mean(per_face_comp2))
        print("ALL:%s" % np.mean(per_face_comp2))

        # OS---------------------------------------------------------------
        OS_acc = []
        for i in range(1, self.num_classes + 2):
            class_pos = np.where(label_t_np == i)
            if len(class_pos[0]) > 0:
                class_i_preds = pred_t_np[class_pos]
                class_i_label = label_t_np[class_pos]
                per_face_comp_ = (class_i_preds == class_i_label).astype(np.int)
                OS_acc.append(np.mean(per_face_comp_))
        self.log("eval_OS", np.mean(OS_acc))
        print("OS:%s" % np.mean(OS_acc))

        # OS*--------------------------------------------------------------
        OS1_acc = []
        for i in range(1, self.num_classes + 1):
            class_pos = np.where(label_t_np == i)
            if len(class_pos[0]) > 0:
                class_i_preds = pred_t_np[class_pos]
                class_i_label = label_t_np[class_pos]
                per_face_comp_ = (class_i_preds == class_i_label).astype(np.int)
                OS1_acc.append(np.mean(per_face_comp_))
        self.log("eval_OS*", np.mean(OS1_acc))
        print("OS*:%s" % np.mean(OS1_acc))

        # unk--------------------------------------------------------------
        unknown_pos = np.where(label_t_np >= self.num_classes + 1)
        unknown_pred_t_np = pred_t_np[unknown_pos]
        unknown_label_t_np = label_t_np[unknown_pos]
        per_face_comp3 = (unknown_pred_t_np == unknown_label_t_np).astype(np.int)
        self.log("eval_UNK", np.mean(per_face_comp3))
        print("UNK:%s" % np.mean(per_face_comp3))


    def configure_optimizers(self):
        # step-1-1
        opt_step_1_1 = torch.optim.AdamW(self.discriminator_multi.parameters(), lr=0.001, betas=(0.99, 0.999))

        # step-1-2
        opt_step_1_2 = torch.optim.AdamW(self.discriminator_node.parameters(), lr=0.001, betas=(0.99, 0.999))
        opt_step_1_2.add_param_group({'params': self.discriminator_graph.parameters(), 'lr': 0.001, 'betas': (0.99, 0.999)})

        # step-2-1
        opt_step_2_1 = torch.optim.AdamW(self.brep_encoder.parameters(), lr=0.00005, betas=(0.99, 0.999))
        opt_step_2_1.add_param_group({'params': self.classifier.parameters(), 'lr': 0.0005, 'betas': (0.99, 0.999)})
        opt_step_2_1.add_param_group({'params': self.discriminator_domain_node.parameters(), 'lr': 0.0005, 'betas': (0.99, 0.999)})
        opt_step_2_1.add_param_group({'params': self.discriminator_domain_graph.parameters(), 'lr': 0.0005, 'betas': (0.99, 0.999)})

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
