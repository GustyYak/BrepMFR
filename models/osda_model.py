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
from .modules.domain_adv.grl import WarmStartGradientReverseLayer

def bce_loss(output, target):
    output_neg = 1 - output
    target_neg = 1 - target
    result = torch.mean(target * torch.log(output + 1e-6))
    result += torch.mean(target_neg * torch.log(output_neg + 1e-6))
    return -torch.mean(result)


class NonLinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        """
        A 3-layer MLP with linear outputs
        Args:
            input_dim (int): Dimension of the input tensor
            num_classes (int): Dimension of the output logits
            dropout (float, optional): Dropout used after each linear layer. Defaults to 0.3.
        """
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

        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, inp, return_feat=False, reverse=False):
        """
        Forward pass
        Args:
            inp (torch.tensor): Inputs features to be mapped to logits
                                (batch_size x input_dim)

        Returns:
            torch.tensor: Logits (batch_size x num_classes)
        """
        if reverse:
            x = self.grl(inp)
        else:
            x = inp
        feat = x

        x = F.relu(self.bn1(self.linear1(x)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = F.relu(self.bn3(self.linear3(x)))
        x = self.dp3(x)
        x = self.linear4(x)

        if return_feat:
            return x, feat
        else:
            return x


class OpenSetDomainAdapt(pl.LightningModule):
    """
    PyTorch Lightning module to train/test the model.
    """
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

        # garph encoder--------------------------------------------------------------------
        self.brep_encoder = BrepEncoder(
            # < for graphormer
            num_degree=128,  # number of in degree types in the graph
            num_spatial=64,  # number of spatial types in the graph
            num_edge_dis=64,  # number of edge dis types in the graph
            edge_type="multi_hop",  # edge type in the graph "multi_hop"
            multi_hop_max_dist=16,  # max distance of multi-hop edges
            # >
            num_encoder_layers=args.n_layers_encode,  # num encoder layers
            embedding_dim=args.dim_node,  # encoder embedding dimension
            ffn_embedding_dim=args.d_model,  # encoder embedding dimension for FFN
            num_attention_heads=args.n_heads,  # num encoder attention heads
            dropout=args.dropout,  # dropout probability
            attention_dropout=args.attention_dropout,  # dropout probability for"attention weights"
            activation_dropout=args.act_dropout,  # dropout probability after"activation in FFN"
            layerdrop=0.1,
            encoder_normalize_before=True,  # apply layernorm before each encoder block
            pre_layernorm=True,
            # apply layernorm before self-attention and ffn. Without this, post layernorm will used
            apply_params_init=True,  # use custom param initialization for Graphormer
            activation_fn="gelu",  # activation function to use
        )
        # garph encoder--------------------------------------------------------------------

        # node classifier------------------------------------------------------------------
        self.num_classes = args.num_classes
        self.classifier = NonLinearClassifier(2*args.dim_node, args.num_classes + 1, args.dropout)
        # node classifier------------------------------------------------------------------

        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=args.num_classes,
                                                    compute_on_step=False)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=args.num_classes,
                                                  compute_on_step=False)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=args.num_classes,
                                                   compute_on_step=False)
        self.per_face_acc_source = []
        self.per_face_acc_target = []
        self.known_classes_acc = []

    def training_step(self, batch, batch_idx):
        self.brep_encoder.train()
        self.classifier.train()

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
        node_pos_t = torch.where(padding_mask_t == False)  # [(batch_size, node_index)]
        local_feat_t = node_emb_t[node_pos_t]              # [total_nodes, dim]

        # local-global
        padding_mask_s_ = ~padding_mask_s
        num_nodes_per_graph_s = torch.sum(padding_mask_s_.long(), dim=-1)  # [batch_size]
        graph_emb_s_repeat = graph_emb_s.repeat_interleave(num_nodes_per_graph_s, dim=0).to(graph_emb.device)
        node_feat_s = torch.cat((local_feat_s, graph_emb_s_repeat), dim=1)  # node_feat_s [total_nodes, dim]

        padding_mask_t_ = ~padding_mask_t
        num_nodes_per_graph_t = torch.sum(padding_mask_t_.long(), dim=-1)      # [batch_size]
        graph_emb_t_repeat = graph_emb_t.repeat_interleave(num_nodes_per_graph_t, dim=0).to(graph_emb.device)
        node_feat_t = torch.cat((local_feat_t, graph_emb_t_repeat), dim=1)  # node_feat_t [total_nodes, dim]

        # node classifier------------------------------------------------------------------
        node_seg_s = self.classifier(node_feat_s)  # [total_nodes, num_classes + 1]
        node_seg_t = self.classifier(node_feat_t, reverse=True)  # [total_nodes, num_classes + 1]

        # source classify loss-------------------------------------------------------------
        num_node_s = node_seg_s.size()[0]
        label_s = batch["label_feature"][:num_node_s].long() - 1
        loss_s = F.cross_entropy(node_seg_s, label_s, reduction="mean")
        self.log("train_loss_s", loss_s, on_step=False, on_epoch=True)
        pred_s = torch.argmax(F.softmax(node_seg_s[:, :self.num_classes], dim=-1), dim=-1)  # pres [total_nodes]
        self.train_accuracy(pred_s, label_s)

        # target classify loss-------------------------------------------------------------
        pred_t = F.softmax(node_seg_t, dim=-1)
        prob1 = torch.sum(pred_t[:, :self.num_classes], 1).view(-1, 1)
        prob2 = pred_t[:, self.num_classes].contiguous().view(-1, 1)
        prob = torch.cat((prob1, prob2), dim=1)
        target_funk = torch.FloatTensor(prob.size()[0], 2).fill_(0.5).to(prob.device)
        loss_t = bce_loss(prob, target_funk)
        self.log("train_loss_t", loss_t, on_step=False, on_epoch=True)

        loss = loss_s + loss_t
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        torch.cuda.empty_cache()
        return loss

    def training_epoch_end(self, training_step_outputs):
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log("current_lr", current_lr, on_step=False, on_epoch=True)
        self.log("train_accuracy", self.train_accuracy.compute())

    def validation_step(self, batch, batch_idx):
        self.brep_encoder.eval()
        self.classifier.eval()

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

        # local-global
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
        loss_s = F.cross_entropy(node_seg_s, label_s, reduction="mean")
        self.log("eval_loss_s", loss_s, on_step=False, on_epoch=True)
        pred_s = torch.argmax(F.softmax(node_seg_s[:, :self.num_classes], dim=-1), dim=-1)  # pres [total_nodes]
        self.val_accuracy(pred_s, label_s)
        # pre_face_acc-----------------------------------
        pred_s_np = pred_s.long().detach().cpu().numpy()
        label_s_np = label_s.long().detach().cpu().numpy()
        per_face_comp = (pred_s_np == label_s_np).astype(np.int)
        self.per_face_acc_source.append(np.mean(per_face_comp))

        # target classify loss---------------------------------------------------------
        pred_t = F.softmax(node_seg_t, dim=-1)
        prob1 = torch.sum(pred_t[:, :self.num_classes], 1).view(-1, 1)
        prob2 = pred_t[:, self.num_classes].contiguous().view(-1, 1)
        prob = torch.cat((prob1, prob2), dim=1)
        target_funk = torch.FloatTensor(prob.size()[0], 2).fill_(0.5).to(prob.device)
        loss_t = bce_loss(prob, target_funk)
        self.log("eval_loss_t", loss_t, on_step=False, on_epoch=True)
        # pre_face_acc-----------------------------------
        pred_t = torch.argmax(pred_t, dim=-1)  # pres [total_nodes]
        num_node_s = node_seg_s.size()[0]
        label_t = batch["label_feature"][num_node_s:]  # labels [total_nodes]
        label_unk = (self.num_classes + 1) * torch.ones_like(label_t)
        label_t = torch.where(label_t > self.num_classes, label_unk, label_t) - 1
        pred_t_np = pred_t.long().detach().cpu().numpy()
        label_t_np = label_t.long().detach().cpu().numpy()
        per_face_comp = (pred_t_np == label_t_np).astype(np.int)
        self.per_face_acc_target.append(np.mean(per_face_comp))

        loss = loss_s + loss_t
        self.log("eval_loss", loss, on_step=False, on_epoch=True)
        torch.cuda.empty_cache()
        return loss

    def validation_epoch_end(self, val_step_outputs):
        # self.log("val_iou", self.val_iou.compute())
        self.log("eval_accuracy", self.val_accuracy.compute())
        self.log("per_face_accuracy_source", np.mean(self.per_face_acc_source))
        self.log("per_face_accuracy_target", np.mean(self.per_face_acc_target))
        self.per_face_acc_source = []
        self.per_face_acc_target = []

    def test_step(self, batch, batch_idx):
        self.brep_encoder.eval()
        self.classifier.eval()

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
        local_feat_t = node_emb_t[node_pos_t]  # [total_nodes, dim]

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

        pred_t = torch.argmax(F.softmax(node_seg_t, dim=-1), dim=-1) + 1  # pres [total_nodes] {1,...,self.num_classes+1}
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
            file_name = "feature_" + str(batch["id"][i].long().detach().cpu().numpy()) + ".txt"
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
        self.per_face_acc_target = []
        self.known_classes_acc = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.005, betas=(0.99, 0.999))
        # optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.001)

        # 学习策略
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                               threshold=0.00001, threshold_mode='rel',
                                                               min_lr=0.0001, cooldown=10, verbose=False)

        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1, "monitor": "eval_loss"}
                }

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
                pg["lr"] = lr_scale * 0.005
