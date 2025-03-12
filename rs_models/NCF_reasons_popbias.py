# -*- coding: utf-8 -*-
# @Time   : 2020/6/27
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/8/22,
# @Author : Zihan Lin
# @Email  : linzihan.super@foxmain.com

r"""
NeuMF
################################################
Reference:
    Xiangnan He et al. "Neural Collaborative Filtering." in WWW 2017.
"""

import torch
import torch.nn as nn
from torch.nn.init import normal_

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.layers import MLPLayers
from recbole.utils import InputType
from recbole.model.general_recommender import BPR
from recbole.model.loss import BPRLoss, EmbLoss

import numpy as np
import pickle


class NeuMFReasonsPop(GeneralRecommender):
    r"""NeuMF is an neural network enhanced matrix factorization model.
    It replace the dot product to mlp for a more precise user-item interaction.

    Note:

        Our implementation only contains a rough pretraining function.

    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(NeuMFReasonsPop, self).__init__(config, dataset)

        # load populiary bias
        self.n_pop_bucket = config["pop_bucket"]
        self.n_reasons = len(set(dataset.field2token_id["reasons_id"]))

        # debias coefficient
        self.debias_coeffiecient = config["debias_coeffiecient"]

        # load parameters info
        self.mf_embedding_size = config["mf_embedding_size"]
        self.mlp_embedding_size = config["mlp_embedding_size"]
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.dropout_prob = config["dropout_prob"]
        self.mf_train = config["mf_train"]
        self.mlp_train = config["mlp_train"]
        self.use_pretrain = config["use_pretrain"]
        self.mf_pretrain_path = config["mf_pretrain_path"]
        self.mlp_pretrain_path = config["mlp_pretrain_path"]

        self.overlap = config["bucket_overlap"]
        self.pop_ratio = config["pop_ratio"]
        self.pop_embedding = nn.Embedding(
            self.n_pop_bucket, config["mlp_embedding_size"])
        dataset_name = config["dataset"]

        if "amazoncd" in dataset_name:
            tmp_file_path = "rs_models/popularity/cluster_results_MiniLM_UMAP20_openai_amazoncd_logfrequency.pkl"
        elif "movielens" in dataset_name:
            tmp_file_path = "rs_models/popularity/cluster_results_MiniLM_UMAP20_openai_movielens_logfrequency.pkl"
        elif "amazonbooks" in dataset_name:
            tmp_file_path = "rs_models/popularity/cluster_results_MiniLM_UMAP20_openai_amazonbooks_logfrequency.pkl"
        else:
            raise ValueError(f"dataset {dataset_name} not supported")
        with open(tmp_file_path, 'rb') as f:
            item_pops = pickle.load(f)
        raw_id2model_id = dataset.field2token_id["reasons_id"]

        model_id_item_pops = {}
        for key, value in item_pops.items():
            model_id_item_pops[raw_id2model_id[str(key)]] = value

        # print(len(model_id_item_pops))
        # print(self.n_reasons)
        # assert len(model_id_item_pops) == self.n_reasons - 1 # 有点问题，需要验证

        # 计算每个桶的宽度和重叠宽度
        bins = self.n_pop_bucket
        bin_width = 1 / bins
        overlap_width = bin_width * self.overlap

        # 创建桶的边界
        edges = np.linspace(0, 1, bins + 1)

        pop_item_weights = [np.zeros(bins)]
        # 计算桶的加权和
        for i in range(1, self.n_reasons):
            if i not in model_id_item_pops:
                weights = np.zeros(bins) + 1/bins
                pop_item_weights.append(weights)
                continue
            x = model_id_item_pops[i]
            weights = np.zeros(bins)
            for i in range(bins):
                # 计算每个桶的起始和结束点
                start = edges[i] - overlap_width / 2
                end = edges[i+1] + overlap_width / 2

                # 确保桶的边界在[0, 1]范围内
                start = max(0, start)
                end = min(1, end)

                # 如果x在当前桶的范围内
                if start <= x <= end:
                    # 计算x在当前桶内的位置占比
                    if x < (start + end) / 2:
                        # x在桶的左半部分
                        weights[i] = (x - start) / ((end - start) / 2)
                    else:
                        # x在桶的右半部分
                        weights[i] = (end - x) / ((end - start) / 2)
            pop_item_weights.append(weights)
        pop_item_weights = torch.FloatTensor(np.array(pop_item_weights))
        # pop_item_weights = nn.Parameter(pop_item_weights, requires_grad=False)
        self.pop_embedding = nn.Embedding.from_pretrained(
            pop_item_weights, freeze=True)
        # assert self.pop_embedding.weight.requires_grad == False
        assert self.pop_embedding.weight.shape[0] == self.n_reasons, self.pop_embedding.weight.shape
        # self.rawpop2embedding_layer = nn.Linear(
        #     in_features=self.n_pop_bucket, out_features=self.mlp_hidden_size)
        self.pop_mlp_layer = nn.Linear(
            self.n_pop_bucket, self.mlp_hidden_size[-1])
        assert self.pop_mlp_layer.weight.requires_grad == True
        self.pop_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)

        # load dataset info
        self.LABEL = config["LABEL_FIELD"]

        # self.start_reasons_ratio = config["start_reasons_ratio"]
        # self.end_reasons_ratio = config["end_reasons_ratio"]
        # self.start_epoch = config["start_epoch"]
        # self.end_epoch = config["end_epoch"]
        # self.rate = (self.end_reasons_ratio - self.start_reasons_ratio) / \
        #     max((self.end_epoch - self.start_epoch), 1)
        self.ui_ratio = config["ui_ratio"]
        self.reason_ratio = config["reason_ratio"]

        # define layers and loss
        self.user_mf_embedding = nn.Embedding(
            self.n_users, self.mf_embedding_size)
        self.item_mf_embedding = nn.Embedding(
            self.n_items, self.mf_embedding_size)
        self.user_mlp_embedding = nn.Embedding(
            self.n_users, self.mlp_embedding_size)
        self.item_mlp_embedding = nn.Embedding(
            self.n_items, self.mlp_embedding_size)

        self.reasons_mf_embedding = nn.Embedding(
            self.n_reasons, self.mf_embedding_size)
        self.reasons_mlp_embedding = nn.Embedding(
            self.n_reasons, self.mlp_embedding_size)

        self.mlp_layers = MLPLayers(
            [2 * self.mlp_embedding_size] +
            self.mlp_hidden_size, self.dropout_prob
        )
        self.reasons_mlp_layer = MLPLayers(
            [2 * self.mlp_embedding_size] +
            self.mlp_hidden_size, self.dropout_prob
        )
        self.mlp_layers.logger = None  # remove logger to use torch.save()
        self.reasons_mlp_layer.logger = None

        self.transform_mf_layer = nn.Linear(
            self.mf_embedding_size, self.mf_embedding_size)
        self.transform_mlp_layer = nn.Linear(
            self.mlp_embedding_size, self.mlp_embedding_size)

        if self.mf_train and self.mlp_train:
            self.predict_layer = nn.Linear(
                self.mf_embedding_size + self.mlp_hidden_size[-1], 1
            )
            self.reasons_predict_layer = nn.Linear(
                self.mf_embedding_size + self.mlp_hidden_size[-1], 1)
        elif self.mf_train:
            self.predict_layer = nn.Linear(self.mf_embedding_size, 1)
        elif self.mlp_train:
            self.predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        self.neg_user_l = config["neg_user_loss"]
        self.ur_i_l = config["ur_i_loss"]
        self.ir_u_l = config["ir_u_loss"]
        self.u_ir_l = config["u_ir_loss"]
        self.reg_loss = config["reg_loss"]

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()
        self.reason_loss = nn.BCEWithLogitsLoss()
        self.reg_loss_calculate = EmbLoss()

        self.pop_loss = nn.BCEWithLogitsLoss()

        # parameters initialization
        if self.use_pretrain:
            self.load_pretrain()
        else:
            self.apply(self._init_weights)

    def load_pretrain(self):
        r"""A simple implementation of loading pretrained parameters."""
        mf = torch.load(self.mf_pretrain_path, map_location="cpu")
        mlp = torch.load(self.mlp_pretrain_path, map_location="cpu")
        mf = mf if "state_dict" not in mf else mf["state_dict"]
        mlp = mlp if "state_dict" not in mlp else mlp["state_dict"]
        self.user_mf_embedding.weight.data.copy_(
            mf["user_mf_embedding.weight"])
        self.item_mf_embedding.weight.data.copy_(
            mf["item_mf_embedding.weight"])
        self.user_mlp_embedding.weight.data.copy_(
            mlp["user_mlp_embedding.weight"])
        self.item_mlp_embedding.weight.data.copy_(
            mlp["item_mlp_embedding.weight"])

        mlp_layers = list(self.mlp_layers.state_dict().keys())
        index = 0
        for layer in self.mlp_layers.mlp_layers:
            if isinstance(layer, nn.Linear):
                weight_key = "mlp_layers." + mlp_layers[index]
                bias_key = "mlp_layers." + mlp_layers[index + 1]
                assert (
                    layer.weight.shape == mlp[weight_key].shape
                ), f"mlp layer parameter shape mismatch"
                assert (
                    layer.bias.shape == mlp[bias_key].shape
                ), f"mlp layer parameter shape mismatch"
                layer.weight.data.copy_(mlp[weight_key])
                layer.bias.data.copy_(mlp[bias_key])
                index += 2

        predict_weight = torch.cat(
            [mf["predict_layer.weight"], mlp["predict_layer.weight"]], dim=1
        )
        predict_bias = mf["predict_layer.bias"] + mlp["predict_layer.bias"]

        self.predict_layer.weight.data.copy_(predict_weight)
        self.predict_layer.bias.data.copy_(0.5 * predict_bias)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    def forward(self, user, item):
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)
        user_mlp_e = self.user_mlp_embedding(user)
        item_mlp_e = self.item_mlp_embedding(item)

        if self.mf_train:
            # [batch_size, embedding_size]
            mf_output = torch.mul(user_mf_e, item_mf_e)
        if self.mlp_train:
            mlp_output = self.mlp_layers(
                torch.cat((user_mlp_e, item_mlp_e), -1)
            )  # [batch_size, layers[-1]]

        if self.mf_train and self.mlp_train:
            output = self.predict_layer(torch.cat((mf_output, mlp_output), -1))
        elif self.mf_train:
            output = self.predict_layer(mf_output)
        elif self.mlp_train:
            output = self.predict_layer(mlp_output)
        else:
            raise RuntimeError(
                "mf_train and mlp_train can not be False at the same time"
            )

        if self.reg_loss:
            return output.squeeze(-1), self.reg_loss_calculate(user_mf_e, item_mf_e, user_mlp_e, item_mlp_e)
        return output.squeeze(-1)

    def reasons_forward(self, reason_user, reason_item, reason_id):
        reason_user_mf_e = self.user_mf_embedding(reason_user)
        reason_item_mf_e = self.item_mf_embedding(reason_item)
        reason_reason_mf_e = self.reasons_mf_embedding(reason_id)
        reason_user_mlp_e = self.user_mlp_embedding(reason_user)
        reason_item_mlp_e = self.item_mlp_embedding(reason_item)
        reason_reason_mlp_e = self.reasons_mlp_embedding(reason_id)

        reason_mf_output = torch.mul(reason_user_mf_e, reason_item_mf_e)
        reason_reason_mf_e = self.transform_mf_layer(reason_reason_mf_e)
        reason_mf_output = torch.mul(reason_mf_output, reason_reason_mf_e)

        reason_mlp_output = self.reasons_mlp_layer(
            torch.cat((reason_user_mlp_e, reason_item_mlp_e), -1)
        )
        reason_reason_mlp_e = self.transform_mlp_layer(reason_reason_mlp_e)
        reason_mlp_output = torch.mul(
            reason_reason_mlp_e, reason_mlp_output)

        reason_output = self.reasons_predict_layer(
            torch.cat((reason_mf_output, reason_mlp_output), -1))

        if self.reg_loss:
            return reason_output.squeeze(-1), self.reg_loss_calculate(reason_user_mf_e, reason_item_mf_e, reason_reason_mf_e, reason_user_mlp_e, reason_item_mlp_e, reason_reason_mlp_e)
        return reason_output.squeeze(-1)

    def pop_forward(self, reason_id):
        pop_e = self.pop_embedding(reason_id)
        pop_e = self.pop_mlp_layer(pop_e)
        pop_output = self.pop_predict_layer(pop_e)
        return pop_output.squeeze(-1)

    def calculate_loss(self, interaction, epoch_idx):
        rs_data, reasons_data, neg_user_data = interaction[
            'rs_data'], interaction['reason_data'], interaction['neg_user_data']
        interaction = rs_data
        user = rs_data[self.USER_ID]
        item = rs_data[self.ITEM_ID]
        reason = rs_data["reasons_id"]
        label = rs_data[self.LABEL]

        reason_user = reasons_data[self.USER_ID]
        reason_item = reasons_data[self.ITEM_ID]
        reason_id = reasons_data["reasons_id"]
        reason_label = reasons_data[self.LABEL]

        neg_user = neg_user_data[self.USER_ID]
        neg_item = neg_user_data[self.ITEM_ID]
        neg_reason = neg_user_data["reasons_id"]
        neg_label = neg_user_data[self.LABEL]

        if self.reg_loss:
            output, reg_l = self.forward(
                user, item)  # U-I
            reason_output, tmp_reg_l = self.reasons_forward(
                reason_user, reason_item, reason_id)  # UI-R
            reg_l += tmp_reg_l
        else:
            output = self.forward(
                user, item)  # U-I
            reason_output = self.reasons_forward(
                reason_user, reason_item, reason_id)  # UI-R

        raw_loss = self.loss(output, label)
        reason_loss = self.loss(reason_output, reason_label)

        if self.ur_i_l:
            if self.reg_loss:
                UR_I_output, tmp_reg_l = self.reasons_forward(
                    user, item, reason)
                reg_l += tmp_reg_l
            else:
                UR_I_output = self.reasons_forward(user, item, reason)
            UR_I_loss = self.loss(UR_I_output, label)
            reason_loss += UR_I_loss

        if self.neg_user_l:
            if self.reg_loss:
                neg_user_output, tmp_reg_l = self.forward(
                    neg_user, neg_item)
                reg_l += tmp_reg_l
            else:
                neg_user_output = self.forward(neg_user, neg_item)
            neg_user_loss = self.loss(neg_user_output, neg_label)
            raw_loss += neg_user_loss

        if self.ir_u_l:
            if self.reg_loss:
                IR_U_output, tmp_reg_l = self.reasons_forward(
                    neg_user, neg_item, neg_reason)
                reg_l += tmp_reg_l
            else:
                IR_U_output = self.reasons_forward(
                    neg_user, neg_item, neg_reason)
            IR_U_loss = self.loss(IR_U_output, neg_label)
            reason_loss += IR_U_loss

        if self.u_ir_l:
            if self.reg_loss:
                U_IR_output, tmp_reg_l = self.reasons_forward(
                    user, neg_item, reason_id)
                reg_l += tmp_reg_l
            else:
                U_IR_output = self.reasons_forward(user, neg_item, reason_id)
            U_IR_loss = self.loss(U_IR_output, reason_label)
            reason_loss += U_IR_loss

        pop_output = self.pop_forward(reason)
        pop_loss = self.pop_loss(pop_output, label)

        out_loss = raw_loss * self.ui_ratio + self.reason_ratio * \
            reason_loss + self.pop_ratio * pop_loss
        if self.reg_loss:
            out_loss += 0.01*reg_l.sum()
        return out_loss

    def predict(self, interaction):
        raise NotImplementedError("should use full_sort_predict not predict")
        rs_data, reasons_data = interaction['rs_data'], interaction['reasons_data']
        interaction = rs_data
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        predict = self.sigmoid(self.forward(user, item))
        return predict

    def full_reasons_predict(self, interaction):
        self.reg_loss = False
        reason_user = interaction[self.USER_ID]
        reason_item = interaction[self.ITEM_ID]
        assert reason_user.shape[0] == 1

        # [10*embed_size]
        reason_user_mf_e = self.user_mf_embedding(
            reason_user)
        reason_item_mf_e = self.item_mf_embedding(reason_item)
        # [2300 x embed_size]
        reason_reason_mf_e = self.reasons_mf_embedding.weight

        reason_user_mf_e = reason_user_mf_e.expand(reason_reason_mf_e.shape)
        reason_item_mf_e = reason_item_mf_e.expand(reason_reason_mf_e.shape)

        reason_user_mlp_e = self.user_mlp_embedding(reason_user)
        reason_item_mlp_e = self.item_mlp_embedding(reason_item)
        reason_reason_mlp_e = self.reasons_mlp_embedding.weight

        reason_item_mlp_e = reason_item_mlp_e.expand(reason_reason_mlp_e.shape)
        reason_user_mlp_e = reason_user_mlp_e.expand(reason_reason_mlp_e.shape)

        reason_mf_output = torch.mul(reason_user_mf_e, reason_item_mf_e)
        reason_reason_mf_e = self.transform_mf_layer(reason_reason_mf_e)
        reason_mf_output = torch.mul(reason_mf_output, reason_reason_mf_e)

        reason_mlp_output = self.reasons_mlp_layer(
            torch.cat((reason_user_mlp_e, reason_item_mlp_e), -1)
        )
        reason_reason_mlp_e = self.transform_mlp_layer(reason_reason_mlp_e)
        reason_mlp_output = torch.mul(
            reason_reason_mlp_e, reason_mlp_output)

        reason_output = self.reasons_predict_layer(
            torch.cat((reason_mf_output, reason_mlp_output), -1))

        reason_pop_e = torch.LongTensor(
            range(self.n_reasons)).to(reason_user.device)
        pop_output = self.pop_forward(reason_pop_e)
        pop_output = pop_output.unsqueeze(-1)
        debias_output = reason_output - self.debias_coeffiecient * pop_output

        predict = self.sigmoid(debias_output)
        return predict

    def full_sort_predict(self, interaction):
        self.reg_loss = False
        interaction, neg_items = interaction
        reason_user = interaction[self.USER_ID]
        reason_item = interaction[self.ITEM_ID]
        reason_user = reason_user.expand(len(neg_items)+1)
        reason_item = torch.cat([reason_item, neg_items], dim=0)

        # [10*embed_size]
        reason_user_mf_e = self.user_mf_embedding(
            reason_user)
        reason_item_mf_e = self.item_mf_embedding(reason_item)
        # [2300 x embed_size]
        reason_reason_mf_e = self.reasons_mf_embedding.weight

        reason_user_mf_e = reason_user_mf_e.unsqueeze(
            1).expand(-1, reason_reason_mf_e.shape[0], -1)
        reason_item_mf_e = reason_item_mf_e.unsqueeze(
            1).expand(-1, reason_reason_mf_e.shape[0], -1)
        reason_reason_mf_e = reason_reason_mf_e.unsqueeze(0).expand(
            reason_user_mf_e.shape[0], -1, -1)

        reason_user_mlp_e = self.user_mlp_embedding(reason_user)
        reason_item_mlp_e = self.item_mlp_embedding(reason_item)
        reason_reason_mlp_e = self.reasons_mlp_embedding.weight

        reason_user_mlp_e = reason_user_mlp_e.unsqueeze(
            1).expand(-1, reason_reason_mlp_e.shape[0], -1)
        reason_item_mlp_e = reason_item_mlp_e.unsqueeze(
            1).expand(-1, reason_reason_mlp_e.shape[0], -1)
        reason_reason_mlp_e = reason_reason_mlp_e.unsqueeze(0).expand(
            reason_user_mlp_e.shape[0], -1, -1)

        # expand to same size as reason_reason_mf_e
        # reason_user_mf_e = reason_user_mf_e.expand(reason_reason_mf_e.shape)
        # reason_item_mf_e = reason_item_mf_e.expand(reason_reason_mf_e.shape)
        # reason_user_mlp_e = reason_user_mlp_e.expand(reason_reason_mlp_e.shape)
        # reason_item_mlp_e = reason_item_mlp_e.expand(reason_reason_mlp_e.shape)

        reason_mf_output = torch.mul(reason_user_mf_e, reason_item_mf_e)
        reason_reason_mf_e = self.transform_mf_layer(reason_reason_mf_e)
        reason_mf_output = torch.mul(reason_mf_output, reason_reason_mf_e)

        reason_mlp_output = self.reasons_mlp_layer(
            torch.cat((reason_user_mlp_e, reason_item_mlp_e), -1)
        )
        reason_reason_mlp_e = self.transform_mlp_layer(reason_reason_mlp_e)
        reason_mlp_output = torch.mul(
            reason_reason_mlp_e, reason_mlp_output)

        # [10 x 2300 x 1]
        reason_output = self.reasons_predict_layer(
            torch.cat((reason_mf_output, reason_mlp_output), -1))
        reason_output = torch.max(reason_output, dim=0, keepdim=False)[0]

        reason_pop_e = torch.LongTensor(
            range(self.n_reasons)).to(reason_user.device)
        pop_output = self.pop_forward(reason_pop_e)
        pop_output = pop_output.unsqueeze(-1)
        debias_output = reason_output - \
            self.debias_coeffiecient * pop_output

        predict = self.sigmoid(debias_output)
        return predict

    def dump_parameters(self):
        r"""A simple implementation of dumping model parameters for pretrain."""
        if self.mf_train and not self.mlp_train:
            save_path = self.mf_pretrain_path
            torch.save(self, save_path)
        elif self.mlp_train and not self.mf_train:
            save_path = self.mlp_pretrain_path
            torch.save(self, save_path)
