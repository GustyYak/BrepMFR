import pathlib
import os
import numpy as np
import pytorch_lightning as pl
import torchmetrics
import torch
from torch import nn
import torch.nn.functional as F
from .modules.uvnet_encoders import *


class _NonLinearClassifier(nn.Module):
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
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, num_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, inp):
        """
        Forward pass

        Args:
            inp (torch.tensor): Inputs features to be mapped to logits
                                (batch_size x input_dim)

        Returns:
            torch.tensor: Logits (batch_size x num_classes)
        """
        x = F.relu(self.bn1(self.linear1(inp)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x


###############################################################################
# Classification model
###############################################################################


class UVNetClassifier(nn.Module):
    """
    UV-Net solid classification model
    """

    def __init__(
        self,
        num_classes,
        crv_emb_dim=64,
        srf_emb_dim=64,
        graph_emb_dim=128,
        dropout=0.3,
    ):
        """
        Initialize the UV-Net solid classification model
        
        Args:
            num_classes (int): Number of classes to output
            crv_emb_dim (int, optional): Embedding dimension for the 1D edge UV-grids. Defaults to 64.
            srf_emb_dim (int, optional): Embedding dimension for the 2D face UV-grids. Defaults to 64.
            graph_emb_dim (int, optional): Embedding dimension for the graph. Defaults to 128.
            dropout (float, optional): Dropout for the final non-linear classifier. Defaults to 0.3.
        """
        super().__init__()
        self.curv_encoder = UVNetCurveEncoder(
            in_channels=6, output_dims=crv_emb_dim
        )
        self.surf_encoder = UVNetSurfaceEncoder(
            in_channels=7, output_dims=srf_emb_dim
        )
        self.graph_encoder = UVNetGraphEncoder(
            srf_emb_dim, crv_emb_dim, graph_emb_dim,
        )
        self.clf = _NonLinearClassifier(graph_emb_dim, num_classes, dropout)

    def forward(self, batched_graph):
        """
        Forward pass

        Args:
            batched_graph (dgl.Graph): A batched DGL graph containing the face 2D UV-grids in node features
                                       (ndata['x']) and 1D edge UV-grids in the edge features (edata['x']).

        Returns:
            torch.tensor: Logits (batch_size x num_classes)
        """
        # Input features
        input_crv_feat = batched_graph.edata["x"]
        input_srf_feat = batched_graph.ndata["x"]
        # Compute hidden edge and face features
        hidden_crv_feat = self.curv_encoder(input_crv_feat)
        hidden_srf_feat = self.surf_encoder(input_srf_feat)
        # Message pass and compute per-face(node) and global embeddings
        # Per-face embeddings are ignored during solid classification
        _, graph_emb = self.graph_encoder(
            batched_graph, hidden_srf_feat, hidden_crv_feat
        )
        # Map to logits
        out = self.clf(graph_emb)
        return out


class Classification(pl.LightningModule):
    """
    PyTorch Lightning module to train/test the classifier.
    """

    def __init__(self, num_classes):
        """
        Args:
            num_classes (int): Number of per-solid classes in the dataset
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = UVNetClassifier(num_classes=num_classes)
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, batched_graph):
        logits = self.model(batched_graph)
        return logits

    def training_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        labels = batch["label"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        preds = F.softmax(logits, dim=-1)
        self.log("train_acc", self.train_acc(preds, labels), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        labels = batch["label"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        preds = F.softmax(logits, dim=-1)
        self.log("val_acc", self.val_acc(preds, labels), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        labels = batch["label"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        preds = F.softmax(logits, dim=-1)
        self.log("test_acc", self.test_acc(preds, labels), on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer


###############################################################################
# Segmentation model
###############################################################################


class UVNetSegmenter(nn.Module):
    """
    UV-Net solid face segmentation model
    """

    def __init__(
        self,
        num_classes,
        crv_emb_dim=64,
        srf_emb_dim=64,
        graph_emb_dim=128,
        dropout=0.3,
    ):
        """
        Initialize the UV-Net solid face segmentation model

        Args:
            num_classes (int): Number of classes to output per-face
            crv_in_channels (int, optional): Number of input channels for the 1D edge UV-grids
            crv_emb_dim (int, optional): Embedding dimension for the 1D edge UV-grids. Defaults to 64.
            srf_emb_dim (int, optional): Embedding dimension for the 2D face UV-grids. Defaults to 64.
            graph_emb_dim (int, optional): Embedding dimension for the graph. Defaults to 128.
            dropout (float, optional): Dropout for the final non-linear classifier. Defaults to 0.3.
        """
        super().__init__()
        # A 1D convolutional network to encode B-rep edge geometry represented as 1D UV-grids
        self.curv_encoder = UVNetCurveEncoder(
            in_channels=7, output_dims=crv_emb_dim
        )
        # A 2D convolutional network to encode B-rep face geometry represented as 2D UV-grids
        self.surf_encoder = UVNetSurfaceEncoder(
            in_channels=7, output_dims=srf_emb_dim
        )
        # A graph neural network that message passes face and edge features
        self.graph_encoder = UVNetGraphEncoder(
            srf_emb_dim, crv_emb_dim, graph_emb_dim, srf_emb_dim
        )
        # A non-linear classifier that maps face embeddings to face logits
        self.seg = _NonLinearClassifier(
            graph_emb_dim + srf_emb_dim, num_classes, dropout=dropout
        )

    def forward(self, batch_data):

        inputs = batch_data["graph"]
        input_srf_feat = inputs.ndata["x"].permute(0, 3, 1, 2)
        input_crv_feat = inputs.edata["x"].permute(0, 2, 1)

        hidden_crv_feat = self.curv_encoder(input_crv_feat)
        hidden_srf_feat = self.surf_encoder(input_srf_feat)
        node_emb, graph_emb = self.graph_encoder(inputs, hidden_srf_feat, hidden_crv_feat)

        padding_mask = batch_data["padding_mask"]  # [batch_size, max_node_num]
        padding_mask_ = ~padding_mask
        num_nodes_per_graph = torch.sum(padding_mask_.long(), dim=-1)  # [batch_size]
        graph_emb_repeat = graph_emb.repeat_interleave(num_nodes_per_graph, dim=0).to(graph_emb.device)
        local_global_feat = torch.cat((node_emb, graph_emb_repeat), dim=1)

        node_seg = self.seg(local_global_feat)  # [total_nodes, num_classes]

        # graph embedding--------------------------------------------------------------------------------
        graph_emb = torch.unsqueeze(graph_emb, dim=0)

        return node_seg, graph_emb


class Segmentation(pl.LightningModule):
    """
    PyTorch Lightning module to train/test the model.
    """

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = args.num_classes
        self.brep_encoder = UVNetSegmenter(args.num_classes, args.dim_node, args.dim_node, args.dim_node, args.dropout)

        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=args.num_classes,
                                                    compute_on_step=False)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=args.num_classes,
                                                  compute_on_step=False)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=args.num_classes,
                                                   compute_on_step=False)
        self.per_face_acc = []
        self.known_classes_acc = []
        self.per_class_acc = []
        self.IoU = []

    def training_step(self, batch, batch_idx):
        self.brep_encoder.train()
        logits, _ = self.brep_encoder(batch)  # logits [total_nodes, num_classes]
        labels = batch["label_feature"].long() - 1
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1)  # pres [total_nodes]
        # self.train_iou(preds, labels)
        self.train_accuracy(preds, labels)

        torch.cuda.empty_cache()
        return loss

    def training_epoch_end(self, training_step_outputs):
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log("current_lr", current_lr, on_step=False, on_epoch=True)
        # self.log("train_iou", self.train_iou.compute())
        self.log("train_accuracy", self.train_accuracy.compute())

    def validation_step(self, batch, batch_idx):
        self.brep_encoder.eval()
        logits, _ = self.brep_encoder(batch)  # logits [total_nodes, num_classes]

        labels = batch["label_feature"].long() - 1  # labels [total_nodes]
        loss = F.cross_entropy(logits, labels)

        preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1)  # pres [total_nodes]
        # self.val_iou(preds, labels)
        self.val_accuracy(preds, labels)

        preds_np = preds.long().detach().cpu().numpy()
        labels_np = labels.long().detach().cpu().numpy()
        per_face_comp = (preds_np == labels_np).astype(np.int)
        self.per_face_acc.append(np.mean(per_face_comp))

        self.log("val_loss", 1.0 / np.mean(per_face_comp), on_step=False, on_epoch=True)

        torch.cuda.empty_cache()
        return loss

    def validation_epoch_end(self, val_step_outputs):
        # self.log("val_iou", self.val_iou.compute())
        self.log("val_accuracy", self.val_accuracy.compute())
        self.log("per_face_accuracy", np.mean(self.per_face_acc))
        self.per_face_acc = []

    def test_step(self, batch, batch_idx):
        self.brep_encoder.eval()
        logits, _ = self.brep_encoder(batch)  # logits [total_nodes, num_classes]

        preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1) + 1  # pres [total_nodes]
        n_graph, max_n_node = batch["padding_mask"].size()[:2]
        node_pos = torch.where(batch["padding_mask"] == False)
        face_feature = -1 * torch.ones([n_graph, max_n_node], device=self.device, dtype=torch.long)
        face_feature[node_pos] = preds[:]

        # 将结果转为txt文件--------------------------------------------------------------------------
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

        # pre-face acc-----------------------------------------------------------------------------
        preds_np = preds.long().detach().cpu().numpy()
        labels = batch["label_feature"].long()  # labels [total_nodes]
        labels_np = labels.long().detach().cpu().numpy()

        known_pos = np.where(labels_np < self.num_classes + 1)
        preds_np = preds_np[known_pos]
        labels_np = labels_np[known_pos]

        per_face_comp = (preds_np == labels_np).astype(np.int)
        self.per_face_acc.append(np.mean(per_face_comp))

        # pre-class acc-----------------------------------------------------------------------------
        per_class_acc = []
        for i in range(1, self.num_classes + 1):
            class_pos = np.where(labels_np == i)
            if len(class_pos[0]) > 0:
                class_i_preds = preds_np[class_pos]
                class_i_label = labels_np[class_pos]
                per_face_comp = (class_i_preds == class_i_label).astype(np.int)
                per_class_acc.append(np.mean(per_face_comp))
        self.per_class_acc.append(np.mean(per_class_acc))

        # IoU---------------------------------------------------------------------------------------
        per_class_iou = []
        for i in range(1, self.num_classes + 1):
            label_pos = np.where(labels_np == i)
            pred_pos = np.where(preds_np == i)
            if len(pred_pos[0]) > 0 and len(label_pos[0]) > 0:
                class_i_preds = preds_np[label_pos]
                class_i_label = labels_np[label_pos]
                Intersection = (class_i_preds == class_i_label).astype(np.int)
                Union = (class_i_preds != class_i_label).astype(np.int)
                class_i_preds_ = preds_np[pred_pos]
                class_i_label_ = labels_np[pred_pos]
                Union_ = (class_i_preds_ != class_i_label_).astype(np.int)
                per_class_iou.append(np.sum(Intersection) / (np.sum(Union) + np.sum(Intersection) + np.sum(Union_)))
        self.IoU.append(np.mean(per_class_iou))

        # known_class_acc--------------------------------------------------------------------------
        known_pos = np.where(labels_np < self.num_classes + 1)
        known_preds_np = preds_np[known_pos]
        known_labels_np = labels_np[known_pos]
        known_classes_comp = (known_preds_np == known_labels_np).astype(np.int)
        self.known_classes_acc.append(np.mean(known_classes_comp))

    def test_epoch_end(self, outputs):
        self.log("per_face_accuracy", np.mean(self.per_face_acc))
        self.log("per_class_accuracy", np.mean(self.per_class_acc))
        self.log("IoU", np.mean(self.IoU))
        self.log("known_classes_accuracy", np.mean(self.known_classes_acc))

        print("num_classes: %s" % self.num_classes)
        print("per_face_accuracy: %s" % np.mean(self.per_face_acc))
        print("per_class_accuracy: %s" % np.mean(self.per_class_acc))
        print("IoU: %s" % np.mean(self.IoU))
        print("known_classes_accuracy %s" % np.mean(self.known_classes_acc))
        self.per_face_acc = []
        self.known_classes_acc = []
        self.per_class_acc = []
        self.IoU = []


    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.01)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.001)

        # 学习策略
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
        #                                                        threshold=0.00005, threshold_mode='rel',
        #                                                        min_lr=0.0000001, cooldown=5, verbose=False)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                               threshold=0.00001, threshold_mode='rel',
                                                               min_lr=0.0001, cooldown=10, verbose=False)

        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1, "monitor": "val_loss"}
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
