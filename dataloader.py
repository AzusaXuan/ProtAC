import sys
import random
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import esm
import torch.nn.functional as F
import os
import csv
import time
import h5py
import utils
import re
import numpy as np
from torch.utils.data.dataloader import default_collate
from itertools import takewhile

original_letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                          'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                          'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                          'N': 2, 'Y': 18, 'M': 12, '-': 20, 'X': 21, 'B': 22,
                          'Z': 23, 'O': 24, 'U': 25}  # '-': padding, 'X': unknown
esm_letter_to_num = {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9,
                     'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20,
                     'H': 21, 'W': 22, 'C': 23, 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30,
                     '<null_1>': 31, '<mask>': 32}


def id_collate(batch):
    new_batch = []
    ids = []
    seqs = []
    for _batch in batch:
        new_batch.append(_batch[:-2])
        ids.append(_batch[-2])
        seqs.append(_batch[-1])
    return default_collate(new_batch), ids, seqs


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    sampler = torch.utils.data.DistributedSampler(datasets, num_replicas=num_tasks, rank=global_rank,
                                                  shuffle=shuffles)
    return sampler


def create_loader(dataset, sampler, batch_size, num_workers, is_training):
    if is_training:
        shuffle = (sampler is None)
        drop_last = False
    else:
        shuffle = False
        drop_last = False

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        # local_rank = local_rank,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        shuffle=shuffle,
        collate_fn=None,
        drop_last=drop_last,
        prefetch_factor=16,
    )

    return loader


def create_caption_loader(dataset, sampler, batch_size, num_workers):
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        # local_rank = local_rank,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        shuffle=False,
        collate_fn=id_collate,
        drop_last=True,
        prefetch_factor=16,
    )

    return loader

class Dataset_pretrain_new(data.Dataset):
    def __init__(self, dataset, seq_len, num_annotation, epoch, captioned_seq_sav_dir, args):
        super(Dataset_pretrain_new, self).__init__()
        self.seq_len = seq_len
        self.n_annotation = num_annotation
        self.dataset = dataset
        self.seqs = 'seqs'
        self.kws = 'kws'

        if epoch == 0:
            self.en_path = f'/hpc2hdd/home/lzhang819/zixuanjiang/datasets/{self.dataset}/{self.dataset}_new_split_sample_merged.hdf5'  # merged; untokenized
        else:
            self.en_path = captioned_seq_sav_dir
        with h5py.File(self.en_path, 'r') as f:
            self.en_train_count = f[self.seqs].shape[0]

        self.sw_path = "/hpc2hdd/home/lzhang819/zixuanjiang/datasets/swissprot/swissprot_double.hdf5"
        with h5py.File(self.sw_path, 'r') as f:
            self.sw_train_count = f[self.seqs].shape[0]

        self.total_ind = self.en_train_count + self.sw_train_count
        print("en_path", self.en_path, "sw_path", self.sw_path)
        print("en_train_count", self.en_train_count, "sw_train_count", self.sw_train_count)

    def __len__(self):
        return self.total_ind

    def __getitem__(self, idx):
        kws = np.zeros((self.n_annotation), dtype=bool)
        if idx < self.en_train_count:
            with h5py.File(self.en_path, 'r') as en_datasets:
                seqs = en_datasets[self.seqs][idx]
                kws_indices = en_datasets[self.kws][idx]
        else:
            idx = idx - self.en_train_count
            with h5py.File(self.sw_path, 'r') as sw_datasets:
                seqs = sw_datasets[self.seqs][idx]
                kws_indices = sw_datasets[self.kws][idx]

        if np.any(kws_indices):
            kws[kws_indices] = True
        try:
            seq_lens = list(seqs).index(20)
        except ValueError:
            seq_lens = self.seq_len

        return torch.tensor(seqs, dtype=torch.int), torch.tensor(kws, dtype=torch.float), torch.tensor(seq_lens, dtype=torch.float)


class Dataset_pretrain_untokenized(data.Dataset):
    def __init__(self, dataset, seq_len, num_annotation, epoch, captioned_seq_sav_dir, args, alphabet,
                 truncation_seq_length):
        super(Dataset_pretrain_untokenized, self).__init__()
        self.seq_len = seq_len
        self.n_annotation = num_annotation
        self.dataset = dataset
        self.seqs = 'seqs'
        self.kws = 'kws'
        self.alphabet = alphabet
        self.truncation_seq_length = truncation_seq_length

        if epoch == 0:
            self.en_path = f'/hpc2hdd/home/lzhang819/zixuanjiang/datasets/{self.dataset}/{self.dataset}_new_split_sample_untokenized.hdf5'  # merged; untokenized
        else:
            self.en_path = captioned_seq_sav_dir
        with h5py.File(self.en_path, 'r') as f:
            self.en_train_count = f[self.seqs].shape[0]

        self.sw_path = "/hpc2hdd/home/lzhang819/zixuanjiang/datasets/swissprot/swissprot_train_untokenized_new.hdf5"
        with h5py.File(self.sw_path, 'r') as f:
            self.sw_train_count = f[self.seqs].shape[0]

        self.total_ind = self.en_train_count + self.sw_train_count
        print("en_path", self.en_path, "sw_path", self.sw_path)
        print("en_train_count", self.en_train_count, "sw_train_count", self.sw_train_count)

    def __len__(self):
        return self.total_ind

    def __getitem__(self, idx):
        kws = np.zeros((self.n_annotation), dtype=bool)
        if idx < self.en_train_count:
            with h5py.File(self.en_path, 'r') as en_datasets:
                seqs = en_datasets[self.seqs][idx]
                kws_indices = en_datasets[self.kws][idx]
        else:
            idx = idx - self.en_train_count
            with h5py.File(self.sw_path, 'r') as sw_datasets:
                seqs = sw_datasets[self.seqs][idx]
                kws_indices = sw_datasets[self.kws][idx]

        if np.any(kws_indices):
            kws[kws_indices] = True
        seq_lens = len(seqs)

        seq_encoded_list = self.alphabet.encode(seqs.decode("utf-8"))
        if self.truncation_seq_length:
            seq_encoded_list = seq_encoded_list[:self.truncation_seq_length]
        tokens = torch.empty(
            (
                self.seq_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)

        if self.alphabet.prepend_bos:
            tokens[0] = self.alphabet.cls_idx
        seq = torch.tensor(seq_encoded_list, dtype=torch.int64)

        tokens[
        int(self.alphabet.prepend_bos): len(seq_encoded_list)
                                        + int(self.alphabet.prepend_bos)
        ] = seq
        if self.alphabet.append_eos:
            tokens[len(seq_encoded_list) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx

        return torch.tensor(tokens, dtype=torch.int), torch.tensor(kws, dtype=torch.float), torch.tensor(seq_lens,
                                                                                                         dtype=torch.int)


class Dataset_caption(data.Dataset):
    def __init__(self, dataset="uniref50_2018", seq_len=512, num_annotation=7533, epoch=0, captioned_seq_sav_dir=None):
        super(Dataset_caption, self).__init__()
        self.seq_len = seq_len
        self.n_annotation = num_annotation
        self.dataset = dataset
        self.inds = 'inds'
        self.seqs = 'seqs'
        self.kws = 'kws'

        if epoch == 0:
            self.en_path = f'/hpc2hdd/home/lzhang819/zixuanjiang/datasets/{self.dataset}/{self.dataset}_new_split_sample_merged.hdf5'
        else:
            self.en_path = captioned_seq_sav_dir
        with h5py.File(self.en_path, 'r') as f:
            self.en_train_count = f[self.seqs].shape[0]

        self.total_ind = self.en_train_count
        print("en_path", self.en_path)
        print("en_train_count", self.en_train_count)

    def __len__(self):
        return self.total_ind

    def __getitem__(self, idx):
        kws = np.zeros((self.n_annotation), dtype=bool)
        with h5py.File(self.en_path, 'r') as en_datasets:
            inds = en_datasets[self.inds][idx]
            seqs = en_datasets[self.seqs][idx]
            kws_indices = en_datasets[self.kws][idx]

        if np.any(kws_indices):
            kws[kws_indices] = True

        return torch.tensor(inds, dtype=torch.int), torch.tensor(seqs, dtype=torch.int), torch.tensor(kws, dtype=torch.float)


class Dataset_swiss_random(data.Dataset):

    def __init__(self, num_annotation, split="train"):
        super(Dataset_swiss_random, self).__init__()

        self.path = f"/hpc2hdd/home/lzhang819/zixuanjiang/datasets/swissprot/swissprot_double.hdf5"
        # self.cnt = len(os.listdir(self.path))
        # print("swiss cnt: ", self.cnt)
        self.seqs = 'seqs'
        self.kws = 'kws'
        self.n_annotation = num_annotation
        with h5py.File(self.path, 'r') as f:
            self.cnt = f[self.seqs].shape[0]
        print("sw cnt: ", self.cnt)

    def __len__(self):
        return self.cnt

    def __getitem__(self, idx):
        kws = np.zeros((self.n_annotation), dtype=bool)

        with h5py.File(self.path, 'r') as sw_datasets:
            seqs = sw_datasets[self.seqs][idx]
            kws_indices = sw_datasets[self.kws][idx]

        if np.any(kws_indices):
            kws[kws_indices] = True
        try:
            seq_lens = list(seqs).index(20)
        except ValueError:
            seq_lens = len(seqs)
        return torch.tensor(seqs, dtype=torch.int), torch.tensor(kws, dtype=torch.float), torch.tensor(seq_lens, dtype=torch.float)


class Dataset_finetune_untokenized(data.Dataset):
    def __init__(self, alphabet, truncation_seq_length, seq_len=512, num_annotation=7533):
        super(Dataset_finetune_untokenized, self).__init__()
        self.seqs = 'seqs'
        self.kws = 'kws'
        self.alphabet = alphabet
        self.truncation_seq_length = truncation_seq_length
        self.seq_len = seq_len
        self.n_annotation = num_annotation
        self.sw_path = f"/hpc2hdd/home/zjiang597/datasets/swissprot/swissprot_train_untokenized_new.hdf5"
        with h5py.File(self.sw_path, 'r') as f:
            self.sw_train_count = f[self.seqs].shape[0]
        print("sw_train_count", self.sw_train_count)

    def __len__(self):
        return self.sw_train_count

    def __getitem__(self, idx):
        kws = np.zeros((self.n_annotation), dtype=bool)

        with h5py.File(self.sw_path, 'r') as sw_datasets:
            seqs = sw_datasets[self.seqs][idx]
            kws_indices = sw_datasets[self.kws][idx]

        if np.any(kws_indices):
            kws[kws_indices] = True
        seq_lens = len(seqs)
        seq_encoded_list = self.alphabet.encode(seqs.decode("utf-8"))
        if self.truncation_seq_length:
            seq_encoded_list = seq_encoded_list[:self.truncation_seq_length]
        tokens = torch.empty(
            (
                self.seq_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)

        if self.alphabet.prepend_bos:
            tokens[0] = self.alphabet.cls_idx
        seq = torch.tensor(seq_encoded_list, dtype=torch.int64)
        tokens[
        int(self.alphabet.prepend_bos): len(seq_encoded_list)
                                        + int(self.alphabet.prepend_bos)
        ] = seq
        if self.alphabet.append_eos:
            tokens[len(seq_encoded_list) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx
        return torch.tensor(tokens, dtype=torch.int), torch.tensor(kws, dtype=torch.float), torch.tensor(seq_lens, dtype=torch.float)


class Dataset_swiss_random_test(data.Dataset):

    def __init__(self):
        super(Dataset_swiss_random_test, self).__init__()

        self.path = f"/hpc2hdd/home/lzhang819/zixuanjiang/datasets/swissprot/test"
        self.cnt = len(os.listdir(self.path))

    def __len__(self):
        return self.cnt

    def __getitem__(self, idx):
        batch = self._from_name_to_protein(idx)
        return batch

    def _from_name_to_protein(self, idx):
        entry = np.load(os.path.join(self.path, f"{idx}.npy"), allow_pickle=True)
        seq, annotation_masks = entry
        try:
            seq_lens = list(seq).index(20)
        except ValueError:
            seq_lens = len(seq)
        return torch.tensor(seq, dtype=torch.int), torch.tensor(annotation_masks, dtype=torch.float), torch.tensor(seq_lens, dtype=torch.float)


class Dataset_swiss_random_test_untokenized(data.Dataset):

    def __init__(self, alphabet, truncation_seq_length, seq_len=512, num_annotation=7533):
        super(Dataset_swiss_random_test_untokenized, self).__init__()
        self.seqs = 'seqs'
        self.kws = 'kws'
        self.alphabet = alphabet
        self.truncation_seq_length = truncation_seq_length
        self.seq_len = seq_len
        self.n_annotation = num_annotation
        self.sw_path = f"/hpc2hdd/home/lzhang819/zixuanjiang/datasets/swissprot/swissprot_test_untokenized_new.hdf5"
        with h5py.File(self.sw_path, 'r') as f:
            self.sw_test_count = f[self.seqs].shape[0]
        print("sw_test_count", self.sw_test_count)

    def __len__(self):
        return self.sw_test_count

    def __getitem__(self, idx):
        kws = np.zeros((self.n_annotation), dtype=bool)

        with h5py.File(self.sw_path, 'r') as sw_datasets:
            seqs = sw_datasets[self.seqs][idx]
            kws_indices = sw_datasets[self.kws][idx]

        if np.any(kws_indices):
            kws[kws_indices] = True
        seq_lens = len(seqs)
        seq_encoded_list = self.alphabet.encode(seqs.decode("utf-8"))
        if self.truncation_seq_length:
            seq_encoded_list = seq_encoded_list[:self.truncation_seq_length]
        tokens = torch.empty(
            (
                self.seq_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)

        if self.alphabet.prepend_bos:
            tokens[0] = self.alphabet.cls_idx
        seq = torch.tensor(seq_encoded_list, dtype=torch.int64)
        tokens[
        int(self.alphabet.prepend_bos): len(seq_encoded_list)
                                        + int(self.alphabet.prepend_bos)
        ] = seq
        if self.alphabet.append_eos:
            tokens[len(seq_encoded_list) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx
        return torch.tensor(tokens, dtype=torch.int), torch.tensor(kws, dtype=torch.float), torch.tensor(seq_lens,
                                                                                                         dtype=torch.int)


class Dataset_swiss_testset_kw(data.Dataset):

    def __init__(self, alphabet, truncation_seq_length, split="train", seq_len=512):
        super(Dataset_swiss_testset_kw, self).__init__()
        self.alphabet = alphabet
        self.truncation_seq_length = truncation_seq_length
        self.seq_len = seq_len
        self.path = f"/hpc2hdd/home/zjiang597/datasets/swiss_test_kw_773_11/{split}"
        self.cnt = len(os.listdir(self.path))
        self.original_letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                                       'N': 2, 'Y': 18, 'M': 12, '-': 20, 'X': 21, 'B': 22,
                                       'Z': 23, 'O': 24, 'U': 25}  # '-': padding, 'X': unknown
        self.orginal_num_to_letter = {v: k for k, v in self.original_letter_to_num.items()}
        self.original_padding_ind = 20
        self.esm_letter_to_num = {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7,
                                  'S': 8, 'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16,
                                  'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24, 'B': 25,
                                  'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32}
        print(f"{split} set: ", self.cnt)

    def __len__(self):
        return self.cnt

    def __getitem__(self, idx):
        batch = self._from_name_to_protein(idx)
        return batch

    def _from_name_to_protein(self, idx):
        entry = np.load(os.path.join(self.path, f"{idx}.npy"), allow_pickle=True)
        seqs, annotation_masks = entry
        new_seq = ''.join(
            [self.orginal_num_to_letter[ind] for ind in takewhile(lambda x: x != self.original_padding_ind, seqs)])
        # print(new_seq)
        seq_encoded_list = self.alphabet.encode(new_seq)
        if self.truncation_seq_length:
            seq_encoded_list = seq_encoded_list[:self.truncation_seq_length]
        tokens = torch.empty(
            (
                self.seq_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)

        if self.alphabet.prepend_bos:
            tokens[0] = self.alphabet.cls_idx
        seq = torch.tensor(seq_encoded_list, dtype=torch.int64)
        tokens[
        int(self.alphabet.prepend_bos): len(seq_encoded_list)
                                        + int(self.alphabet.prepend_bos)
        ] = seq
        if self.alphabet.append_eos:
            tokens[len(seq_encoded_list) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx
        # sys.exit()
        return torch.tensor(tokens, dtype=torch.int), torch.tensor(annotation_masks, dtype=torch.float),

class Dataset_swiss_testset_kw_pb(data.Dataset):

    def __init__(self, truncation_seq_length, split="train", seq_len=512):
        super(Dataset_swiss_testset_kw_pb, self).__init__()
        self.truncation_seq_length = truncation_seq_length
        self.seq_len = seq_len
        self.path = f"/hpc2hdd/home/lzhang819/zixuanjiang/datasets/swiss_test_kw_773_11/{split}_new"
        self.cnt = len(os.listdir(self.path))
        self.original_letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                                       'N': 2, 'Y': 18, 'M': 12, '-': 20, 'X': 21, 'B': 22,
                                       'Z': 23, 'O': 24, 'U': 25}  # '-': padding, 'X': unknown
        self.orginal_num_to_letter = {v: k for k, v in self.original_letter_to_num.items()}
        self.original_padding_ind = 20
        self.esm_letter_to_num = {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7,
                                  'S': 8, 'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16,
                                  'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24, 'B': 25,
                                  'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32}
        print(f"{split} set: ", self.cnt)

    def __len__(self):
        return self.cnt

    def __getitem__(self, idx):
        batch = self._from_name_to_protein(idx)
        return batch

    def _from_name_to_protein(self, idx):
        entry = np.load(os.path.join(self.path, f"{idx}.npy"), allow_pickle=True)
        seqs, annotation_masks = entry
        return torch.tensor(seqs, dtype=torch.int), torch.tensor(annotation_masks, dtype=torch.float),

class Dataset_uniref50_caption_new(data.Dataset):

    def __init__(self, dataset="uniref50_2018", seq_len=512, num_annotation=7533, epoch=0, captioned_seq_sav_dir=None,
                 args=None, truncation_seq_length=512):
        super(Dataset_uniref50_caption_new, self).__init__()
        self.dataset = dataset
        self.seqs = 'seqs'
        self.kws = 'kws'
        self.inds = 'inds'
        self.prot_ids = 'prot_ids'
        self.truncation_seq_length = truncation_seq_length
        self.seq_len = seq_len
        self.n_annotation = num_annotation
        if epoch == 0:
            self.en_path = f'/hpc2hdd/home/lzhang819/zixuanjiang/datasets/{self.dataset}/{self.dataset}_new_split_sample_untokenized.hdf5'  # merged; untokenized
        else:
            self.en_path = captioned_seq_sav_dir
        with h5py.File(self.en_path, 'r') as f:
            self.uniref_count = f[self.seqs].shape[0]
        print("UR50_count", self.uniref_count)

    def __len__(self):
        return self.uniref_count

    def __getitem__(self, idx):
        kws = np.zeros((self.n_annotation), dtype=bool)

        with h5py.File(self.en_path, 'r') as ur_datasets:
            seqs = ur_datasets[self.seqs][idx]
            kws_indices = ur_datasets[self.kws][idx]
            inds = ur_datasets[self.inds][idx]
            prot_ids = ur_datasets[self.prot_ids][idx]

        if np.any(kws_indices):
            kws[kws_indices] = True
        seq_lens = len(seqs)
        return torch.tensor(kws, dtype=torch.float), torch.tensor(seq_lens, dtype=torch.int), torch.tensor(
            inds, dtype=torch.int), torch.tensor(idx, dtype=torch.int), prot_ids, seqs


if __name__ == '__main__':
    model_esm2, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model_esm2.eval()  # disables dropout for deterministic results

    pretrain_dataset = Dataset_uniref50_caption_new(alphabet=alphabet)
    world_size = utils.get_world_size()
    rank = utils.get_rank()
    print(world_size, rank)
    # sys.exit()
    pretrain_sampler = create_sampler(pretrain_dataset, True, world_size, rank)
    batch_size = 32

    pretrain_dataloader = create_caption_loader(pretrain_dataset, pretrain_sampler, batch_size, 8)
    truncation_seq_length = 512
    for i, batch in enumerate(pretrain_dataloader):
        tensor_batch, prot_ids, seq_str_list = batch
        input_anno, input_seqlen, inds, idx = tensor_batch
        print(prot_ids, seq_str_list)

        seq_encoded_list = [alphabet.encode(seq_str) for seq_str in seq_str_list]
        if truncation_seq_length:
            seq_encoded_list = [seq_str[:truncation_seq_length] for seq_str in seq_encoded_list]
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        tokens = torch.empty(
            (
                batch_size,
                max_len + int(alphabet.prepend_bos) + int(alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(alphabet.padding_idx)

        for i, seq_encoded in enumerate(seq_encoded_list):
            if alphabet.prepend_bos:
                tokens[i, 0] = alphabet.cls_idx
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[
            i,
            int(alphabet.prepend_bos): len(seq_encoded)
                                       + int(alphabet.prepend_bos),
            ] = seq
            if alphabet.append_eos:
                tokens[i, len(seq_encoded) + int(alphabet.prepend_bos)] = alphabet.eos_idx
        print(tokens, tokens.shape)
        sys.exit()
        print(input_seqlen)
        print(input_seq.shape, input_seqlen.shape)
        with torch.no_grad():
            results = model_esm2(input_seq, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]
        seq_logits = results["logits"].argmax(-1)
        range_tensor = torch.arange(0, seq_logits.shape[1]).expand(seq_logits.shape[0], -1)

        # 检查每个元素的索引是否小于b中对应的值
        # 这将创建一个布尔掩码，其中True表示索引小于b[i]的元素
        mask = range_tensor <= input_seqlen.unsqueeze(1)
        mask[:, 0] = False
        # 应用掩码到a，然后计算非零元素的数量
        diff = seq_logits - input_seq
        cnt = torch.count_nonzero(diff * mask)
        print(cnt / input_seqlen.sum())
        sys.exit()
