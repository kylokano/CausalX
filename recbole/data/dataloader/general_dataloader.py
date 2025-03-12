# @Time   : 2020/7/7
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2022/7/8, 2020/9/9, 2020/9/29, 2021/7/15, 2022/7/6
# @Author : Zhen Tian, Yupeng Hou, Yushuo Chen, Xingyu Pan, Gaowei Zhang
# @email  : chenyuwuxinn@gmail.com, houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, xy_pan@foxmail.com, zgw15630559577@163.com

"""
recbole.data.dataloader.general_dataloader
################################################
"""

import numpy as np
import torch
from logging import getLogger
from recbole.data.dataloader.abstract_dataloader import (
    AbstractDataLoader,
    NegSampleDataLoader,
)
from recbole.data.interaction import Interaction, cat_interactions
from recbole.utils import InputType, ModelType
from recbole.data.transform import construct_transform

import random
import copy

class TrainDataLoader(NegSampleDataLoader):
    """:class:`TrainDataLoader` is a dataloader for training.
    It can generate negative interaction when :attr:`training_neg_sample_num` is not zero.
    For the result of every batch, we permit that every positive interaction and its negative interaction
    must be in the same batch.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.logger = getLogger()
        self._set_neg_sample_args(
            config, dataset, config["MODEL_INPUT_TYPE"], config["train_neg_sample_args"]
        )
        self.sample_size = len(dataset)
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config["train_batch_size"]
        if self.neg_sample_args["distribution"] != "none":
            batch_num = max(batch_size // self.times, 1)
            new_batch_size = batch_num * self.times
            self.step = batch_num
            self.set_batch_size(new_batch_size)
        else:
            self.step = batch_size
            self.set_batch_size(batch_size)

    def update_config(self, config):
        self._set_neg_sample_args(
            config,
            self._dataset,
            config["MODEL_INPUT_TYPE"],
            config["train_neg_sample_args"],
        )
        super().update_config(config)

    def collate_fn(self, index):
        index = np.array(index)
        data = self._dataset[index]
        transformed_data = self.transform(self._dataset, data)
        return self._neg_sampling(transformed_data)


class ReasonsTrainDataLoader(NegSampleDataLoader):
    """:class:`TrainDataLoader` is a dataloader for training.
    It can generate negative interaction when :attr:`training_neg_sample_num` is not zero.
    For the result of every batch, we permit that every positive interaction and its negative interaction
    must be in the same batch.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.logger = getLogger()
        self.rid_field = "reasons_id"
        self._set_neg_sample_args(
            config, dataset, config["MODEL_INPUT_TYPE"], config["train_neg_sample_args"]
        )
        self.sample_size = len(dataset)
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config["train_batch_size"]
        if self.neg_sample_args["distribution"] != "none":
            batch_num = max(batch_size // self.times, 1)
            new_batch_size = batch_num * self.times
            self.step = batch_num
            self.set_batch_size(new_batch_size)
        else:
            self.step = batch_size
            self.set_batch_size(batch_size)

    def update_config(self, config):
        self._set_neg_sample_args(
            config,
            self._dataset,
            config["MODEL_INPUT_TYPE"],
            config["train_neg_sample_args"],
        )
        super().update_config(config)

    def collate_fn(self, index):
        index = np.array(index)
        data = self._dataset[index]
        transformed_data = self.transform(self._dataset, data)
        return self._neg_sampling(transformed_data)
    
    def _neg_sampling(self, inter_feat):
        if self.neg_sample_args.get("dynamic", False):
            return NotImplementedError
        elif (
            self.neg_sample_args["distribution"] != "none"
            and self.neg_sample_args["sample_num"] != "none"
        ):
            user_ids = inter_feat[self.uid_field].numpy()
            item_ids = inter_feat[self.iid_field].numpy()
            reasons_ids = inter_feat[self.rid_field].numpy()
            neg_item_ids = self._sampler.sample_by_user_ids(
                user_ids, item_ids, self.neg_sample_num
            )
            neg_user_ids = self._sampler.sample_user_by_item_ids(user_ids,item_ids,self.neg_sample_num)
            neg_reasons_ids = self._sampler.sample_reasons_by_user_ids(user_ids, item_ids, reasons_ids, self.neg_sample_num)
            return self.sampling_func(inter_feat, neg_item_ids, neg_user_ids, neg_reasons_ids)
        else:
            return inter_feat
        
    def _neg_sample_by_point_wise_sampling(self, inter_feat, neg_item_ids, neg_user_ids, neg_reasons_ids):
        pos_inter_num = len(inter_feat)
        new_data = inter_feat.repeat(self.times)
        new_data[self.iid_field][pos_inter_num:] = neg_item_ids
        new_data = self._dataset.join(new_data)
        labels = torch.zeros(pos_inter_num * self.times)
        labels[:pos_inter_num] = 1.0
        new_data.update(Interaction({self.label_field: labels}))
        
        new_neg_user_data = inter_feat.repeat(self.times)
        new_neg_user_data[self.uid_field][pos_inter_num:] = neg_user_ids
        new_neg_user_data = self._dataset.join(new_neg_user_data)
        labels = torch.zeros(pos_inter_num * self.times)
        labels[:pos_inter_num] = 1.0
        new_neg_user_data.update(Interaction({self.label_field: labels}))
        
        new_reasons_data = inter_feat.repeat(self.times)
        new_reasons_data[self.rid_field][pos_inter_num:] = neg_reasons_ids
        # new_data = self._dataset.join(new_data)
        reasons_labels = torch.zeros(pos_inter_num * self.times)
        reasons_labels[:pos_inter_num] = 1.0
        new_reasons_data.update(Interaction({self.label_field: reasons_labels}))
        
        return {"rs_data":new_data, "neg_user_data":new_neg_user_data, "reason_data":new_reasons_data}
    
    # todo
    def _neg_sample_by_pair_wise_sampling(self, inter_feat, neg_item_ids,  neg_user_ids, neg_reasons_ids):
        new_inter_feat = inter_feat.repeat(self.times)
        neg_item_feat = Interaction({self.iid_field: neg_item_ids})
        neg_item_feat = self._dataset.join(neg_item_feat)
        neg_item_feat.add_prefix(self.neg_prefix)
        new_inter_feat.update(neg_item_feat)
        
        neg_user_inter_feat = inter_feat.repeat(self.times)
        neg_user_feat = Interaction({self.uid_field: neg_user_ids})
        neg_user_feat = self._dataset.join(neg_user_feat)
        neg_user_feat.add_prefix(self.neg_prefix)
        neg_user_inter_feat.update(neg_user_feat)
        
        neg_reason_inter_feat = inter_feat.repeat(self.times)
        neg_reason_feat = Interaction({self.rid_field: neg_reasons_ids})
        neg_reason_feat = self._dataset.join(neg_reason_feat)
        neg_reason_feat.add_prefix(self.neg_prefix)
        neg_reason_inter_feat.update(neg_reason_feat)
        return {"rs_data":new_inter_feat, "neg_user_data":neg_user_inter_feat, "reason_data":neg_reason_inter_feat}

class NegSampleEvalDataLoader(NegSampleDataLoader):
    """:class:`NegSampleEvalDataLoader` is a dataloader for neg-sampling evaluation.
    It is similar to :class:`TrainDataLoader` which can generate negative items,
    and this dataloader also permits that all the interactions corresponding to each user are in the same batch
    and positive interactions are before negative interactions.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.logger = getLogger()
        phase = sampler.phase if sampler is not None else "test"
        self._set_neg_sample_args(
            config, dataset, InputType.POINTWISE, config[f"{phase}_neg_sample_args"]
        )
        if (
            self.neg_sample_args["distribution"] != "none"
            and self.neg_sample_args["sample_num"] != "none"
        ):
            user_num = dataset.user_num
            dataset.sort(by=dataset.uid_field, ascending=True)
            self.uid_list = []
            start, end = dict(), dict()
            for i, uid in enumerate(dataset.inter_feat[dataset.uid_field].numpy()):
                if uid not in start:
                    self.uid_list.append(uid)
                    start[uid] = i
                end[uid] = i
            self.uid2index = np.array([None] * user_num)
            self.uid2items_num = np.zeros(user_num, dtype=np.int64)
            for uid in self.uid_list:
                self.uid2index[uid] = slice(start[uid], end[uid] + 1)
                self.uid2items_num[uid] = end[uid] - start[uid] + 1
            self.uid_list = np.array(self.uid_list)
            self.sample_size = len(self.uid_list)
        else:
            self.sample_size = len(dataset)
        if shuffle:
            self.logger.warnning("NegSampleEvalDataLoader can't shuffle")
            shuffle = False
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config["eval_batch_size"]
        if (
            self.neg_sample_args["distribution"] != "none"
            and self.neg_sample_args["sample_num"] != "none"
        ):
            inters_num = sorted(self.uid2items_num * self.times, reverse=True)
            batch_num = 1
            new_batch_size = inters_num[0]
            for i in range(1, len(inters_num)):
                if new_batch_size + inters_num[i] > batch_size:
                    break
                batch_num = i + 1
                new_batch_size += inters_num[i]
            self.step = batch_num
            self.set_batch_size(new_batch_size)
        else:
            self.step = batch_size
            self.set_batch_size(batch_size)

    def update_config(self, config):
        phase = self._sampler.phase if self._sampler.phase is not None else "test"
        self._set_neg_sample_args(
            config,
            self._dataset,
            InputType.POINTWISE,
            config[f"{phase}_neg_sample_args"],
        )
        super().update_config(config)

    def collate_fn(self, index):
        index = np.array(index)
        if (
            self.neg_sample_args["distribution"] != "none"
            and self.neg_sample_args["sample_num"] != "none"
        ):
            uid_list = self.uid_list[index]
            data_list = []
            idx_list = []
            positive_u = []
            positive_i = torch.tensor([], dtype=torch.int64)

            for idx, uid in enumerate(uid_list):
                index = self.uid2index[uid]
                transformed_data = self.transform(self._dataset, self._dataset[index])
                data_list.append(self._neg_sampling(transformed_data))
                idx_list += [idx for i in range(self.uid2items_num[uid] * self.times)]
                positive_u += [idx for i in range(self.uid2items_num[uid])]
                positive_i = torch.cat(
                    (positive_i, self._dataset[index][self.iid_field]), 0
                )

            cur_data = cat_interactions(data_list)
            idx_list = torch.from_numpy(np.array(idx_list)).long()
            positive_u = torch.from_numpy(np.array(positive_u)).long()

            return cur_data, idx_list, positive_u, positive_i
        else:
            data = self._dataset[index]
            transformed_data = self.transform(self._dataset, data)
            cur_data = self._neg_sampling(transformed_data)
            return cur_data, None, None, None


class LimitedNumberNegSampleEvalDataLoader(NegSampleEvalDataLoader):
    """:class:`NegSampleEvalDataLoader` is a dataloader for neg-sampling evaluation.
    We modified it to constrain the data samples for the LLM.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """
    def __init__(self, config, dataset, sampler, shuffle=False):
        self.pos_eval_nums = config["eval_args"]["limited"]["pos_eval_nums"]
        self.neg_eval_nums = config["eval_args"]["limited"]["neg_eval_nums"]
        self.total_eval_nums = self.pos_eval_nums + self.neg_eval_nums
        if self.neg_eval_nums<self.pos_eval_nums:
            raise ValueError("neg_eval_nums should be larger than pos_eval_nums in LimitedNumberNegSampleEvalDataLoader")
        if self.neg_eval_nums%self.pos_eval_nums != 0:
            raise NotImplementedError("neg_eval_nums should be divisible by pos_eval_nums in LimitedNumberNegSampleEvalDataLoader")
        random.seed(config.seed)
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        self.neg_sample_num = self.neg_eval_nums//self.pos_eval_nums
        self.times = self.neg_sample_num + 1
        
    def _init_batch_size_and_step(self):
        batch_size = self.config["eval_batch_size"]
        inters_num = sorted(self.uid2items_num * self.times, reverse=True)
        batch_num = 1
        new_batch_size = min(inters_num[0], self.total_eval_nums)
        for i in range(1, len(inters_num)):
            if new_batch_size + min(inters_num[i], self.total_eval_nums) > batch_size:
                break
            batch_num = i + 1
            new_batch_size += min(inters_num[i], self.total_eval_nums)
        self.step = batch_num
        self.set_batch_size(new_batch_size)
    
    def collate_fn(self, index):
        index = np.array(index)
        if (
            self.neg_sample_args["distribution"] != "none"
            and self.neg_sample_args["sample_num"] != "none"
        ):
            uid_list = self.uid_list[index]
            data_list = []
            idx_list = []
            positive_u = []
            positive_i = torch.tensor([], dtype=torch.int64)

            for idx, uid in enumerate(uid_list):
                index = self.uid2index[uid]
                if index.stop - index.start > self.pos_eval_nums:
                    all_nums_index = range(index.start, index.stop, index.step if index.step else 1)
                    index = random.sample(all_nums_index, self.pos_eval_nums)
                transformed_data = self.transform(self._dataset, self._dataset[index])
                data_list.append(self._neg_sampling(transformed_data))
                idx_list += [idx for i in range(min(self.uid2items_num[uid], self.pos_eval_nums) * self.times)]
                positive_u += [idx for i in range(min(self.uid2items_num[uid], self.pos_eval_nums))]
                positive_i = torch.cat(
                    (positive_i, self._dataset[index][self.iid_field]), 0
                )

            cur_data = cat_interactions(data_list)
            idx_list = torch.from_numpy(np.array(idx_list)).long()
            positive_u = torch.from_numpy(np.array(positive_u)).long()

            return cur_data, idx_list, positive_u, positive_i
        else:
            data = self._dataset[index]
            transformed_data = self.transform(self._dataset, data)
            cur_data = self._neg_sampling(transformed_data)
            return cur_data, None, None, None


class FullSortEvalDataLoader(AbstractDataLoader):
    """:class:`FullSortEvalDataLoader` is a dataloader for full-sort evaluation. In order to speed up calculation,
    this dataloader would only return then user part of interactions, positive items and used items.
    It would not return negative items.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.logger = getLogger()
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field
        self.is_sequential = config["MODEL_TYPE"] == ModelType.SEQUENTIAL
        if not self.is_sequential:
            user_num = dataset.user_num
            self.uid_list = []
            self.uid2items_num = np.zeros(user_num, dtype=np.int64)
            self.uid2positive_item = np.array([None] * user_num)
            self.uid2history_item = np.array([None] * user_num)

            dataset.sort(by=self.uid_field, ascending=True)
            last_uid = None
            positive_item = set()
            uid2used_item = sampler.used_ids
            for uid, iid in zip(
                dataset.inter_feat[self.uid_field].numpy(),
                dataset.inter_feat[self.iid_field].numpy(),
            ):
                if uid != last_uid:
                    self._set_user_property(
                        last_uid, uid2used_item[last_uid], positive_item
                    )
                    last_uid = uid
                    self.uid_list.append(uid)
                    positive_item = set()
                positive_item.add(iid)
            self._set_user_property(last_uid, uid2used_item[last_uid], positive_item)
            self.uid_list = torch.tensor(self.uid_list, dtype=torch.int64)
            self.user_df = dataset.join(Interaction({self.uid_field: self.uid_list}))

        self.sample_size = len(self.user_df) if not self.is_sequential else len(dataset)
        if shuffle:
            self.logger.warnning("FullSortEvalDataLoader can't shuffle")
            shuffle = False
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _set_user_property(self, uid, used_item, positive_item):
        if uid is None:
            return
        history_item = used_item - positive_item
        self.uid2positive_item[uid] = torch.tensor(
            list(positive_item), dtype=torch.int64
        )
        self.uid2items_num[uid] = len(positive_item)
        self.uid2history_item[uid] = torch.tensor(list(history_item), dtype=torch.int64)

    def _init_batch_size_and_step(self):
        batch_size = self.config["eval_batch_size"]
        if not self.is_sequential:
            batch_num = max(batch_size // self._dataset.item_num, 1)
            new_batch_size = batch_num * self._dataset.item_num
            self.step = batch_num
            self.set_batch_size(new_batch_size)
        else:
            self.step = batch_size
            self.set_batch_size(batch_size)

    def update_config(self, config):
        super().update_config(config)

    def collate_fn(self, index):
        index = np.array(index)
        if not self.is_sequential:
            user_df = self.user_df[index]
            uid_list = list(user_df[self.uid_field])

            history_item = self.uid2history_item[uid_list]
            positive_item = self.uid2positive_item[uid_list]

            history_u = torch.cat(
                [
                    torch.full_like(hist_iid, i)
                    for i, hist_iid in enumerate(history_item)
                ]
            )
            history_i = torch.cat(list(history_item))

            positive_u = torch.cat(
                [torch.full_like(pos_iid, i) for i, pos_iid in enumerate(positive_item)]
            )
            positive_i = torch.cat(list(positive_item))

            return user_df, (history_u, history_i), positive_u, positive_i
        else:
            interaction = self._dataset[index]
            transformed_interaction = self.transform(self._dataset, interaction)
            inter_num = len(transformed_interaction)
            positive_u = torch.arange(inter_num)
            positive_i = transformed_interaction[self.iid_field]

            return transformed_interaction, None, positive_u, positive_i



# start_iter = False
class FullSortReasonsEvalDataLoader(AbstractDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        if dataset.inter_num == 0:
            return
        self.logger = getLogger()
        
        neg_eval_nums = config["eval_args"]["limited"]["neg_eval_nums"]
        pos_eval_nums = config["eval_args"]["limited"]["pos_eval_nums"]
        self.neg_sample_num = neg_eval_nums // pos_eval_nums
        self.times = 1 + self.neg_sample_num
        
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field
        self.rid_field = "reasons_id"
        self.is_sequential = config["MODEL_TYPE"] == ModelType.SEQUENTIAL
        if self.is_sequential:
            raise NotImplementedError("Sequential model is not supported in FullSortReasonsEvalDataLoader")
        self.ui_pairs2idx = sampler.ui_pair2idx
        pair_idx_list = []
        for uid, iid, rid in zip(
            dataset.inter_feat[self.uid_field].numpy(),
            dataset.inter_feat[self.iid_field].numpy(),
            dataset.inter_feat[self.rid_field].numpy(),
        ):
            uni_id = str(uid)+";"+str(iid)
            pair_idx_list.append(self.ui_pairs2idx[uni_id])
                
        ui_pair_num = len(sampler.reasons_used_id)
        self.reasons_num = sampler.reasons_num
        self.uiid_list = []
        self.uiid2items_num = np.zeros(ui_pair_num, dtype=np.int64)
        self.uiid2positive_item = np.array([None] * ui_pair_num)
        self.uiid2history_item = np.array([None] * ui_pair_num)
        
        # user_num = dataset.user_num
        # self.uid_list = []
        # self.uid2items_num = np.zeros(user_num, dtype=np.int64)
        # self.uid2positive_item = np.array([None] * user_num)
        # self.uid2history_item = np.array([None] * user_num)
        
        
        dataset.inter_feat.update(Interaction({"ui_pair_id": pair_idx_list}))
        dataset.sort(by="ui_pair_id", ascending=True)
        last_uiid = None
        positive_reasons = set()
        uiid2used_reasons = sampler.reasons_used_id
        
        # dataset.sort(by=self.uid_field, ascending=True)
        # last_uid = None
        # positive_item = set()
        # uid2used_item = sampler.used_ids
        
        for uid, iid, rid in zip(
            dataset.inter_feat[self.uid_field].numpy(),
            dataset.inter_feat[self.iid_field].numpy(),
            dataset.inter_feat[self.rid_field].numpy(),
        ):
            ui_pairs = str(uid)+";"+str(iid)
            uiid = self.ui_pairs2idx[ui_pairs]
            if uiid != last_uiid:
                self._set_user_property(
                    last_uiid, uiid2used_reasons[last_uiid], positive_reasons
                )
                last_uiid = uiid
                self.uiid_list.append(uiid)
                positive_reasons = set()
            positive_reasons.add(rid)
        self._set_user_property(
                    last_uiid, uiid2used_reasons[last_uiid], positive_reasons
                )
        # for uid, iid in zip(
        #     dataset.inter_feat[self.uid_field].numpy(),
        #     dataset.inter_feat[self.iid_field].numpy(),
        # ):
        #     if uid != last_uid:
        #         self._set_user_property(
        #             last_uid, uid2used_item[last_uid], positive_item
        #         )
        #         last_uid = uid
        #         self.uid_list.append(uid)
        #         positive_item = set()
        #     positive_item.add(iid)
        # self._set_user_property(last_uid, uid2used_item[last_uid], positive_item)
        
        
        
        uiid2raw_id = {}
        for ui_pair, ui_id in self.ui_pairs2idx.items():
            uiid2raw_id[ui_id] = ui_pair
        raw_uid, raw_iid = [], []
        for per_uiid in self.uiid_list:
            ui_pair = uiid2raw_id[per_uiid]
            per_uid, per_iid = map(int, ui_pair.split(";"))
            raw_uid.append(per_uid)
            raw_iid.append(per_iid)
            
        self.uiid_list = torch.tensor(self.uiid_list, dtype=torch.int64)
        raw_user_id = torch.tensor(raw_uid, dtype=torch.int64)
        raw_item_id = torch.tensor(raw_iid, dtype=torch.int64)
        self.user_df = dataset.join(Interaction({"ui_id": self.uiid_list, self.uid_field: raw_user_id, self.iid_field: raw_item_id}))

        self.sample_size = len(self.user_df) if not self.is_sequential else len(dataset)
        if shuffle:
            self.logger.warnning("FullSortEvalDataLoader can't shuffle")
            shuffle = False
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _set_user_property(self, uid, used_item, positive_item):
        if uid is None:
            return
        history_item = used_item - positive_item
        self.uiid2positive_item[uid] = torch.tensor(
            list(positive_item), dtype=torch.int64
        )
        self.uiid2items_num[uid] = len(positive_item)
        self.uiid2history_item[uid] = torch.tensor(list(history_item), dtype=torch.int64)

    def _init_batch_size_and_step(self):
        batch_size = self.config["eval_batch_size"]
        if not self.is_sequential:
            batch_num = max(batch_size // self.reasons_num, 1)
            new_batch_size = batch_num * self.reasons_num
            self.step = batch_num
            self.set_batch_size(new_batch_size)
        else:
            self.step = batch_size
            self.set_batch_size(batch_size)

    def update_config(self, config):
        super().update_config(config)

    def collate_fn(self, index):
        index = np.array(index)
        user_df = self.user_df[index]
        neg_item_ids = self._sampler.sample_by_user_ids(
            user_df[self.uid_field], user_df[self.iid_field], self.neg_sample_num
        )
        
        uid_list = list(user_df["ui_id"])

        history_item = self.uiid2history_item[uid_list]
        positive_item = self.uiid2positive_item[uid_list]

        history_u = torch.cat(
            [
                torch.full_like(hist_iid, i)
                for i, hist_iid in enumerate(history_item)
            ]
        )
        history_i = torch.cat(list(history_item))

        positive_u = torch.cat(
            [torch.full_like(pos_iid, i) for i, pos_iid in enumerate(positive_item)]
        )
        positive_i = torch.cat(list(positive_item))
        
        return [user_df, neg_item_ids], (history_u, history_i), positive_u, positive_i
