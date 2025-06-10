import esm
import argparse
import os
import pandas as pd
import random
import time
import datetime
import json
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from timm.utils import AverageMeter
import h5py
import gc
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy, BinaryRecall, BinaryPrecision, AUROC
from models.protac_pretrain_protein_v5 import protac_pretrain_v5_kw, protac_pretrain_seq_anno_mlm_esm2_new
import utils
from utils import warmup_lr_schedule, step_lr_schedule, warmup_lr_schedule_k, step_lr_schedule_k
import subprocess
from itertools import chain
import wandb


from tqdm import tqdm
from dataloader import Dataset_pretrain_untokenized, Dataset_swiss_random_test_untokenized, Dataset_swiss_testset_kw, create_loader, create_sampler, Dataset_finetune_untokenized, create_caption_loader
from dataloader import Dataset_uniref50_caption_new
import numpy as np
import torch._dynamo as _dynamo

_dynamo.config.suppress_errors = True



def exists(val):
    return val is not None

def train(model, optimizer, epoch, dataloader, args, test_dataloader):
    model.train()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    log_cnt = 0
    log_interval = 0.05
    save_interval = 0.2
    sav_cnt = 0.2
    end_time = time.time()
    last_idx = len(dataloader) - 1
    total_batch = len(dataloader)
    if args.mode == "finetune":
        average = eval(model, args.actual_epoch, test_dataloader, args)
        if args.rank == 0 and args.use_wandb:
            wandb.log(average)
        model.train()
    for i, batch in enumerate(dataloader):

        data_time_m.update(time.time() - end_time)
        if epoch == 0:
            warmup_lr_schedule_k(optimizer, i, len(dataloader) * 0.2, args.warmup_lr, args.init_lr,
                                 args.batch_size / 32)

        optimizer.zero_grad()
        input_seq, input_anno, input_seqlen = batch
        input_seq = input_seq.to(args.gpu)
        input_anno = input_anno.float().to(args.gpu)
        input_seqlen = input_seqlen.to(args.gpu)

        # Forward pass
        outputs = model(input_seq, input_anno, input_seqlen, "train")
        outputs = tuple(torch.tensor(item).to(args.gpu) if not torch.is_tensor(item) else item for item in outputs)


        loss_ita, loss_itm, loss_lm_anno, loss_lm_seq, anno_diff, itm_accuracy, update_count = outputs
        anno_accuracy, anno_precision, anno_recall, anno_fmax, anno_auc, seq_difference = anno_diff
        loss = loss_ita * args.itc_loss_weight + loss_itm * args.itm_loss_weight + loss_lm_anno * args.anno_lm_loss_weight + loss_lm_seq * args.seq_lm_loss_weight
        # Backward and optimize
        loss.backward()
        optimizer.step()

        batch_time_m.update(time.time() - end_time)
        end_time = time.time()

        if args.rank == 0 and args.use_wandb:
            additional_metrics = {
                'Total_loss': loss.item(),
                'ITA_loss': loss_ita.item(),
                'ITM_loss': loss_itm.item(),
                'LM_anno_loss': loss_lm_anno.item(),
                'seq_anno_loss': loss_ita.item(),
                'anno_accuracy': anno_accuracy.item(),
                'anno_precision': anno_precision.item(),
                'anno_recall': anno_recall.item(),
                'anno_fmax': anno_fmax.item(),
                'anno_auc': anno_auc.item(),
                'itm_accuracy': itm_accuracy.item(),
                'LM_seq_loss': loss_lm_seq.item(),
                'seq_difference': seq_difference.item(),
                'update_count': update_count.item(),
            }
            wandb.log(additional_metrics)

        if i / total_batch > log_cnt and args.mode == "train":
            if args.rank == 0:
                print("Train: {} [{:>4d}/{} ({:>3.0f}%)]  "
                      "Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  "
                      "({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  "
                      "Data: {data_time.val:.3f} ({data_time.avg:.3f})".format(
                    epoch,
                    i,
                    len(dataloader),
                    100.0 * i / last_idx,
                    batch_time=batch_time_m,
                    rate=input_anno.shape[0] * args.world_size / batch_time_m.val,
                    rate_avg=input_anno.shape[0] * args.world_size / batch_time_m.avg,
                    data_time=data_time_m, ))
            log_cnt += log_interval
            average = eval(model, args.actual_epoch, test_dataloader, args)
            if args.rank == 0 and args.use_wandb:
                wandb.log(average)
            model.train()

        if i / total_batch > sav_cnt and args.rank == 0 and args.mode == "train":
            print(f'Saving step {i} checkpoint for epoch {args.actual_epoch}')
            save_obj = {
                'model': model.module.state_dict(),
                'epoch': args.actual_epoch + 1,
            }
            pretrain_checkpoint_i = os.path.join(args.output_dir,
                                                 f'{args.version}_clean_before_ft_e_%02d_per_%.2f_unclean.pth' % (
                                                     args.actual_epoch,
                                                     sav_cnt))
            torch.save(save_obj, pretrain_checkpoint_i)
            sav_cnt += save_interval
            sys.exit()

    if args.mode == "finetune":
        average = eval(model, args.actual_epoch, test_dataloader, args)
        if args.rank == 0 and args.use_wandb:
            wandb.log(average)
        if args.rank == 0 and (epoch + 1) % 10 == 0 and (epoch + 1) != args.finetune_epoch:
            print(f'Saving checkpoint for ft epoch {epoch + 1}')
            save_obj = {
                'model': model.module.state_dict(),
                'epoch': epoch + 1,
            }
            finetune_checkpoint_i = os.path.join(args.output_dir,
                                                 f'{args.version}_clean_after_ft_e_%02d_ft_%02d_unclean.pth' % (
                                                     args.actual_epoch,
                                                     epoch + 1))
            torch.save(save_obj, finetune_checkpoint_i)
        model.train()


def kw_train(model, optimizer, epoch, dataloader, args):
    model.train()
    for i, batch in enumerate(dataloader):

        if epoch == 0:
            warmup_lr_schedule_k(optimizer, i, len(dataloader), args.warmup_lr, args.init_lr, args.batch_size / 32)

        optimizer.zero_grad()
        input_seq, input_anno = batch
        input_seq = input_seq.to(args.gpu)
        input_anno = input_anno.float().to(args.gpu)
        # Forward pass
        outputs = model(input_seq, input_anno)
        outputs = tuple(torch.tensor(item).to(args.gpu) if not torch.is_tensor(item) else item for item in outputs)
        loss_lm_anno, anno_diff = outputs

        anno_accuracy, anno_precision, anno_recall, anno_fmax, anno_auc = anno_diff
        loss = loss_lm_anno * args.anno_lm_loss_weight

        # Backward and optimize
        loss.backward()
        optimizer.step()

        if args.rank == 0 and args.use_wandb:
            additional_metrics = {
                'Total_loss': loss.item(),
                'LM_anno_loss': loss_lm_anno.item(),
                'anno_accuracy': anno_accuracy.item(),
                'anno_precision': anno_precision.item(),
                'anno_recall': anno_recall.item(),
                'anno_fmax': anno_fmax.item(),
                'anno_auc': anno_auc.item(),
            }

            wandb.log(additional_metrics)


def eval(model, epoch, dataloader, args):
    print("start testing!")
    metrics = {
        'TEST_ITA_loss': 0.0,
        'TEST_ITM_loss': 0.0,
        'TEST_LM_anno_loss': 0.0,
        'TEST_LM_seq_loss': 0.0,
        # 'TEST_LM_seq_loss_mlm': 0.0,
        'TEST_Total_loss': 0.0,
        'TEST_seq_anno_loss': 0.0,
        'TEST_anno_accuracy': 0.0,
        'TEST_anno_precision': 0.0,
        'TEST_anno_recall': 0.0,
        'TEST_anno_fmax': 0.0,
        'TEST_anno_auc': 0.0,
        'TEST_itm_accuracy': 0.0,
        'TEST_seq_difference': 0.0,
        # 'TEST_seq_difference_mlm': 0.0,
        'TEST_total_batches': 0
    }

    model.eval()
    with torch.no_grad():
        with tqdm(dataloader, dynamic_ncols=True, leave=False) as tqdmDataLoader:
            for i, batch in enumerate(tqdmDataLoader):

                input_seq, input_anno, input_seqlen = batch
                input_seq = input_seq.to(args.gpu).squeeze(1)
                input_anno = input_anno.float().to(args.gpu)
                input_seqlen = input_seqlen.to(args.gpu)

                outputs = model(input_seq,
                                input_anno,
                                input_seqlen,
                                "test")
                outputs = tuple(
                    torch.tensor(item).to(args.gpu) if not torch.is_tensor(item) else item for item in outputs)
                loss_ita, loss_itm, loss_lm_anno, loss_lm_seq, anno_diff, itm_accuracy, _ = outputs
                anno_accuracy, anno_precision, anno_recall, anno_fmax, anno_auc, seq_difference = anno_diff
                loss = loss_ita * args.itc_loss_weight + loss_itm * args.itm_loss_weight + loss_lm_anno * args.anno_lm_loss_weight + loss_lm_seq * args.anno_lm_loss_weight
                metrics['TEST_LM_seq_loss'] += loss_lm_seq.item()
                metrics['TEST_seq_difference'] += seq_difference.item()

                metrics['TEST_ITA_loss'] += loss_ita.item()
                metrics['TEST_total_batches'] += 1
                metrics['TEST_seq_anno_loss'] = metrics['TEST_ITA_loss']
                metrics['TEST_ITM_loss'] += loss_itm.item()
                metrics['TEST_LM_anno_loss'] += loss_lm_anno.item()
                metrics['TEST_Total_loss'] += loss.item()
                metrics['TEST_anno_accuracy'] += anno_accuracy.item()
                metrics['TEST_anno_precision'] += anno_precision.item()
                metrics['TEST_anno_recall'] += anno_recall.item()
                metrics['TEST_anno_fmax'] += anno_fmax.item()
                metrics['TEST_anno_auc'] += anno_auc.item()
                metrics['TEST_itm_accuracy'] += itm_accuracy.item()
    dist.barrier()

    averages = {key: (value / metrics['TEST_total_batches'] if key != 'TEST_total_batches' else value) for key, value in metrics.items()}

    return averages


def kw_eval(model, epoch, dataloader, args):
    print("start testing!")
    metrics = {
        'TEST_LM_anno_loss': 0.0,
        'TEST_Total_loss': 0.0,
        'TEST_anno_accuracy': 0.0,
        'TEST_anno_precision': 0.0,
        'TEST_anno_recall': 0.0,
        'TEST_anno_fmax': 0.0,
        'TEST_anno_auc': 0.0,
        'TEST_total_batches': 0
    }

    model.eval()
    with torch.no_grad():
        with tqdm(dataloader, dynamic_ncols=True, leave=False) as tqdmDataLoader:
            for i, batch in enumerate(tqdmDataLoader):
                input_seq, input_anno = batch
                input_seq = input_seq.to(args.gpu).squeeze(1)
                input_anno = input_anno.float().to(args.gpu)

                loss_lm_anno, anno_diff = model(input_seq,
                                                input_anno)

                metrics['TEST_total_batches'] += 1

                # Determine the loss based on the version

                loss = loss_lm_anno * args.anno_lm_loss_weight
                anno_accuracy, anno_precision, anno_recall, anno_fmax, anno_auc = anno_diff
                metrics['TEST_LM_anno_loss'] += loss_lm_anno.item()
                metrics['TEST_Total_loss'] += loss.item()
                metrics['TEST_anno_accuracy'] += anno_accuracy.item()
                metrics['TEST_anno_precision'] += anno_precision.item()
                metrics['TEST_anno_recall'] += anno_recall.item()
                metrics['TEST_anno_fmax'] += anno_fmax
                metrics['TEST_anno_auc'] += anno_auc.item()

    averages = {key: value / metrics['TEST_total_batches'] for key, value in metrics.items() if key != 'total_batches'}
    # print(averages)
    # Log metrics if appropriate
    if args.rank == 0 and args.use_wandb:
        wandb.log(averages)


def save_to_disk(filename, data, append=False):
    mode = 'ab' if append else 'wb'
    with open(filename, mode) as f:
        # np.save(f, np.array(data).astype(np.int32))
        f.write(np.array(data).astype(str)[0])


def get_module_by_name(module, name):
    # Check the child modules at the current level
    for child_name, child_mod in module.named_children():
        if name in child_name:
            return child_mod

    # If not found at the current level, recursively check the next level
    for child_name, child_mod in module.named_children():
        result = get_module_by_name(child_mod, name)
        if result:
            return result
    return None

@torch.compile
def data_to_caption_new(model, dataloader, epoch, args, save_dir, alphabet):

    # global SELEC_BATCH_COUNT
    ## Load seq encoder
    seq_encoder = get_module_by_name(model, "seq_encoder")
    anno_encoder = get_module_by_name(model, "anno_encoder")
    anno_decoder = get_module_by_name(model, "anno_decoder")
    to_annotation_logits = get_module_by_name(model, "to_annotation_logits")
    itm_head = get_module_by_name(model, "itm_head")

    model.eval()

    caption_list = []
    chunk_counter = 0
    output_file = f"{save_dir}_{args.rank}_raw.hdf5"
    caption_list_file = f"{args.captioned_ind_sav_dir}/caption_inds_epoch{epoch}_{args.rank}.hdf5"
    if os.path.isfile(caption_list_file):
        os.remove(caption_list_file)
    if os.path.isfile(output_file):
        os.remove(output_file)
    truncation_seq_length = args.seq_len
    with torch.no_grad():
        new_ind = 0
        total_data_len = 0  # Initialize total data length
        with tqdm(dataloader, dynamic_ncols=True, leave=False) as tqdmDataLoader:
            for index, batch in enumerate(tqdmDataLoader):
                tensor_batch, prot_ids, seq_str_list = batch
                input_anno, input_seqlen, inds, idx = tensor_batch
                seq_encoded_list = [alphabet.encode(seq_str.decode("utf-8")) for seq_str in seq_str_list]
                if truncation_seq_length:
                    seq_encoded_list = [seq_str[:truncation_seq_length] for seq_str in seq_encoded_list]
                max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
                input_seq = torch.empty(
                    (
                        args.sample_batch_size,
                        max_len + int(alphabet.prepend_bos) + int(alphabet.append_eos),
                    ),
                    dtype=torch.int64,
                )
                input_seq.fill_(alphabet.padding_idx)

                for i, seq_encoded in enumerate(seq_encoded_list):
                    if alphabet.prepend_bos:
                        input_seq[i, 0] = alphabet.cls_idx
                    seq = torch.tensor(seq_encoded, dtype=torch.int64)
                    input_seq[
                    i,
                    int(alphabet.prepend_bos): len(seq_encoded)
                                               + int(alphabet.prepend_bos),
                    ] = seq
                    if alphabet.append_eos:
                        input_seq[i, len(seq_encoded) + int(alphabet.prepend_bos)] = alphabet.eos_idx
                # print(tokens, tokens.shape)
                input_seq = input_seq.to(args.gpu)
                input_anno = input_anno.float().to(args.gpu)
                inds = inds.to(args.gpu)
                results = seq_encoder(input_seq, repr_layers=[args.esm_layer], return_contacts=False)
                seq_hidden_embeds = results["representations"][args.esm_layer]
                decoder_output = anno_decoder(torch.zeros_like(input_anno), seq_embed=seq_hidden_embeds)
                predict_anno = to_annotation_logits(decoder_output)
                anno_hat = torch.sigmoid(predict_anno).to(input_anno.device)
                output_pos_true = anno_encoder(input_anno, seq_embed=seq_hidden_embeds)
                output_pos_hat = anno_encoder(anno_hat.round(), seq_embed=seq_hidden_embeds)
                vl_output_true = itm_head(output_pos_true)  # [bs, 2]
                vl_output_hat = itm_head(output_pos_hat)  # [bs, 2]

                mask = (vl_output_hat[:, 1] > vl_output_true[:, 1]) & (vl_output_hat[:, 1] > vl_output_hat[:, 0])
                mask_indices = torch.nonzero(mask).squeeze()
                new_inds = inds[mask_indices]

                try:
                    iter(new_inds)  # Check if new_inds is iterable
                except TypeError:
                    caption_list.append(new_inds.detach().cpu().numpy().astype(np.int32))
                    len_new_inds = 1
                else:
                    # Iterable, extend the list
                    caption_list.extend(new_inds.tolist() if hasattr(new_inds, 'tolist') else new_inds)
                    len_new_inds = len(new_inds)

                new_ind += len_new_inds
                # Choose between anno_hat and input_anno based on the mask
                new_kw = torch.where(mask.unsqueeze(1), anno_hat.round(), input_anno)
                inds_np = inds.detach().cpu().numpy().astype(np.int32)
                new_kw_np = new_kw.detach().cpu().numpy().astype(np.bool_)  # new_kw: binary, 0,1

                # get kws
                non_zero = new_kw_np
                col_indices = np.argwhere(non_zero)
                # print("col inds: ", col_indices.shape)
                true_indices_per_row_chunk = np.split(col_indices[:, 1], np.cumsum(np.sum(non_zero, axis=1))[:-1])
                # print("true_indices_per_row_chunk: ", len(true_indices_per_row_chunk))
                dataset_id = int(chunk_counter * args.world_size) + args.rank
                ind_name = f'inds_{dataset_id}'
                seq_name = f'seqs_{dataset_id}'
                kw_name = f'kws_{dataset_id}'
                prot_name = f'prot_ids_{dataset_id}'

                with h5py.File(output_file, 'a') as f:
                    f.create_dataset(ind_name, data=inds_np)
                    str_dt = h5py.special_dtype(vlen=str)
                    name_list = f.create_dataset(prot_name, (len(seq_str_list),), dtype=str_dt)
                    seq_list = f.create_dataset(seq_name, (len(seq_str_list),), dtype=str_dt)

                    name_list[:] = prot_ids
                    seq_list[:] = seq_str_list

                    dt = h5py.special_dtype(vlen=np.dtype('int32'))
                    kws_dataset = f.create_dataset(kw_name, (input_anno.shape[0],), dtype=dt)

                    kws_dataset[0:input_anno.shape[0]] = true_indices_per_row_chunk

                    f.flush()
                # print("new ind: ", new_ind)
                if caption_list:
                    with h5py.File(caption_list_file, 'a') as f:
                        # Check if the dataset already exists
                        if 'caption_inds' not in f:
                            # If the dataset does not exist, create an empty dataset with unlimited growth along the first dimension
                            dset = f.create_dataset('caption_inds', shape=(0,), maxshape=(None,), chunks=True)
                        else:
                            # If the dataset already exists, get it directly
                            dset = f['caption_inds']

                        # Append data in the loop
                        new_data = np.array(caption_list)
                        # Calculate the new dataset size
                        new_size = dset.shape[0] + new_data.shape[0]
                        # Resize the dataset to fit the new data
                        dset.resize(new_size, axis=0)
                        # Write new data to the file
                        dset[-new_data.shape[0]:] = new_data
                # Reset the buffer and increment the chunk counter
                caption_list = []  # Clear the buffer
                chunk_counter += 1
                total_data_len += input_anno.shape[0]  # Update total data length

                print(f"Processed {total_data_len} data points. Captioned {new_ind} seqs.")
        # sys.exit()


def main(args):
    args.distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1


    if args.distributed:
        utils.init_distributed_mode(args)
    # device = torch.device(args.gpu)

    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.gpu = "cuda:%d" % args.local_rank
    args.world_size = utils.get_world_size()
    args.rank = utils.get_rank()

    seed = args.seed
    np.random.seed(seed)
    torch.initial_seed()
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    if args.rank == 0 and args.use_wandb:
        wandb.init(project=args.wandb_project_name,
                   name=args.mode + "_e_" + str(args.actual_epoch) + "_" + args.version,
                   entity='protac',
                   )

    total_epoch = args.actual_epoch

    args.init_checkpoint_path = f"model_{args.version}_init.pth"
    args.init_checkpoint_path_kw = f"model_{args.version}_init_kw.pth"
    pretrain_checkpoint = os.path.join(args.output_dir,
                                       f'{args.version}_clean_before_ft_e_%02d.pth' % (
                                           total_epoch))
    pretrain_sub_checkpoint = os.path.join(
        args.output_dir,
        f'{args.version}_clean_before_ft_e_{total_epoch:02d}_sub_{{}}.pth'
    )
    if os.path.isfile(pretrain_checkpoint) and args.mode == "train":
        print(f"{pretrain_checkpoint} exist !!!!")
        raise RuntimeError

    finetune_checkpoint = os.path.join(args.output_dir,
                                       f'{args.version}_clean_after_ft_e_%02d.pth' % (
                                           total_epoch))
    if os.path.isfile(finetune_checkpoint) and args.mode == "finetune":
        print(f"{finetune_checkpoint} exist !!!!")
        raise RuntimeError

    fewshot_checkpoint = os.path.join(args.output_dir,
                                      f'{args.version}_kw_pred_e_%02d.pth' % (
                                          total_epoch))
    if args.version == "esm2_8m":
        seq_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    elif args.version == "esm2_35m":
        seq_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    elif args.version == "esm2_150m":
        seq_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    elif args.version == "esm2_650m":
        seq_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    elif args.version == "pb":
        seq_model, alphabet = None

    test_dataset = Dataset_swiss_random_test_untokenized(alphabet, args.seq_len)
    test_sampler = create_sampler(test_dataset, False, args.world_size, args.rank)
    test_dataloader = create_loader(test_dataset, test_sampler, args.test_batch_size, 8, False)

    if args.mode == "train":

        pretrain_dataset = Dataset_pretrain_untokenized(
            dataset=args.dataset, seq_len=args.seq_len,
            num_annotation=args.num_annotation, epoch=args.actual_epoch, # for unclean version, args.actual_epoch
            captioned_seq_sav_dir=os.path.join(args.captioned_seq_sav_dir,
                                            f'{args.version}_epoch{total_epoch}.hdf5'),
            args=args,
            alphabet=alphabet,
            truncation_seq_length=args.seq_len)

        pretrain_sampler = create_sampler(pretrain_dataset, True, args.world_size, args.rank)
        pretrain_dataloader = create_loader(pretrain_dataset, pretrain_sampler, args.batch_size, 8, True)
        if total_epoch > 0 and args.checkpoint == None:
            args.checkpoint = os.path.join(args.output_dir,
                                           f'{args.version}_clean_before_ft_e_%02d.pth' % (
                                               total_epoch - 1))

        model, model_without_ddp, start_epoch, optimizer_pretrain, optimizer_finetune = create_model(args,
                                                                                                     seq_model,
                                                                                                     alphabet,
                                                                                                     args.checkpoint, )

        #########           PRETRAINING              ###########
        print('number of training samples: %d' % len(pretrain_dataset))
        print(f"Start pretraining epoch {total_epoch}")

        for epoch in range(0, args.pretrain_epoch):
            if args.distributed:
                pretrain_sampler.set_epoch(epoch + args.actual_epoch * args.pretrain_epoch)

            pretrain_one_epoch(model, optimizer_pretrain, pretrain_dataloader, args, epoch, test_dataloader)

            if args.rank == 0:
                print(f'Saving checkpoint for pretraining sub_epoch{epoch} in total_epoch {total_epoch}')
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer_pretrain': optimizer_pretrain.state_dict(),
                    'optimizer_finetune': optimizer_finetune.state_dict(),
                    'epoch': total_epoch + 1,
                }
                if epoch != args.pretrain_epoch - 1:
                    torch.save(save_obj, pretrain_sub_checkpoint.format(epoch))
                else:
                    torch.save(save_obj, pretrain_checkpoint)

        print("pretraining finished")

    if args.mode == "finetune":
        train_dataset = Dataset_finetune_untokenized(alphabet, args.seq_len)
        train_sampler = create_sampler(train_dataset, True, args.world_size, args.rank)
        train_dataloader = \
            create_loader(train_dataset, train_sampler, batch_size=args.batch_size, num_workers=8, is_training=True)
        if args.ft_first and args.actual_epoch==0:
            args.checkpoint = None
        else:
            args.checkpoint = pretrain_checkpoint if args.checkpoint is None else args.checkpoint
        model, model_without_ddp, _, optimizer_pretrain, optimizer_finetune = create_model(args,
                                                                                           seq_model,
                                                                                           alphabet,
                                                                                           args.checkpoint, )

        for epoch in range(0, args.finetune_epoch):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            finetune_one_epoch(model, optimizer_finetune, train_dataloader, args, epoch, test_dataloader)

        if args.rank == 0:
            print(f'Saving checkpoint for epoch {total_epoch}')
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer_pretrain': optimizer_pretrain.state_dict(),
                'optimizer_finetune': optimizer_finetune.state_dict(),
                'epoch': total_epoch + 1,
            }

            torch.save(save_obj, finetune_checkpoint)

        print("finetuning finished")

    if args.mode == "caption":
        args.checkpoint = finetune_checkpoint if args.checkpoint is None else args.checkpoint
        model, model_without_ddp, _, optimizer_pretrain, optimizer_finetune = create_model(args,
                                                                                           seq_model,
                                                                                           alphabet,
                                                                                           args.checkpoint, )

        caption(model, args, total_epoch, alphabet)

    if args.mode == "eval":
        args.checkpoint = finetune_checkpoint if args.checkpoint is None else args.checkpoint
        model, model_without_ddp, _, optimizer_pretrain, optimizer_finetune = create_model(args,
                                                                                           seq_model,
                                                                                           alphabet,
                                                                                           args.checkpoint, )
        average = eval(model, args.actual_epoch, test_dataloader, args)
        print(average)

    if args.mode == "kw_pred":
        train_dataset = Dataset_swiss_testset_kw(alphabet, args.seq_len, split="train")
        train_sampler = create_sampler(train_dataset, True, args.world_size, args.rank)
        train_dataloader = \
            create_loader(train_dataset, train_sampler, batch_size=args.batch_size, num_workers=8, is_training=True)

        test_dataset = Dataset_swiss_testset_kw(alphabet, args.seq_len, split="test")
        test_sampler = create_sampler(test_dataset, True, args.world_size, args.rank)
        test_dataloader = \
            create_loader(test_dataset, test_sampler, batch_size=args.test_batch_size, num_workers=8, is_training=False)

        args.checkpoint = finetune_checkpoint if args.checkpoint is None else args.checkpoint
        model, model_without_ddp, optimizer_pretrain = create_model_kw(args, seq_model, alphabet, args.checkpoint)

        for epoch in range(0, args.fewshot_epoch):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            fewshot_one_epoch(model, optimizer_pretrain, train_dataloader, args, epoch)
            kw_eval(model, epoch, test_dataloader, args)

        if args.rank == 0:
            print(f'Saving checkpoint for epoch {total_epoch}')
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer_pretrain': optimizer_pretrain.state_dict(),
            }

            torch.save(save_obj, fewshot_checkpoint)

        print("kw pred finished")

    if args.mode == "caption_sw":
        args.checkpoint = pretrain_checkpoint if args.checkpoint is None else args.checkpoint
        model, model_without_ddp, _, optimizer_pretrain, optimizer_finetune = create_model(args,
                                                                                           seq_model,
                                                                                           alphabet,
                                                                                           args.checkpoint, )

        caption_sw(model, args, total_epoch, alphabet)


def pretrain_one_epoch(model, optimizer_pretrain, pretrain_dataloader, args, epoch, test_dataloader):
    if args.rank == 0 and args.use_wandb:
        wandb.log({'Pretrain Epoch': epoch})

    epoch_start = time.time()
    print(f'Starting Epoch: {epoch + 1}')
    step_lr_schedule_k(optimizer_pretrain, epoch, args.init_lr, args.min_lr, args.lr_decay_rate,
                       args.batch_size / 32)

    # epoch =0 to ensure the same learning setting
    train(model, optimizer_pretrain, epoch, pretrain_dataloader, args, test_dataloader)
    if args.distributed:
        dist.barrier()

    total_time = time.time() - epoch_start
    print(f'Pretraining time {total_time / 60.0} mins')


def finetune_one_epoch(model, optimizer_finetune, finetune_dataloader, args, epoch, test_dataloader):
    if args.rank == 0 and args.use_wandb:
        wandb.log({'Epoch': epoch})

    epoch_start = time.time()
    print(f'Finetuning Epoch: {epoch}')
    step_lr_schedule_k(optimizer_finetune, epoch, args.init_lr, args.min_lr, args.lr_decay_rate,
                       args.batch_size / 32)
    train(model, optimizer_finetune, epoch, finetune_dataloader, args, test_dataloader)
    if args.distributed: dist.barrier()

    total_time = time.time() - epoch_start
    print(f'Finetuning time {total_time / 60.0} mins')


def fewshot_one_epoch(model, optimizer_finetune, finetune_dataloader, args, epoch):
    if args.rank == 0 and args.use_wandb:
        wandb.log({'Epoch': epoch})

    epoch_start = time.time()
    print(f'Finetuning Epoch: {epoch}')
    step_lr_schedule_k(optimizer_finetune, epoch, args.init_lr, args.min_lr, args.lr_decay_rate,
                       args.batch_size / 32)
    kw_train(model, optimizer_finetune, epoch, finetune_dataloader, args)
    if args.distributed: dist.barrier()

    total_time = time.time() - epoch_start
    print(f'Finetuning time {total_time / 60.0} mins')


def caption(model, args, epoch, alphabet):
    pre_dir = os.path.join(args.captioned_seq_sav_dir,
                           f"{args.version}_epoch{epoch}.hdf5")
    save_dir = os.path.join(args.captioned_seq_sav_dir,
                            f"{args.version}_epoch{epoch + 1}")
    inds_dir = os.path.join(args.captioned_ind_sav_dir,
                            f"caption_inds_epoch{epoch}")
    caption_dataset = Dataset_uniref50_caption_new(captioned_seq_sav_dir=pre_dir, epoch=epoch, )
    caption_samplers = create_sampler(caption_dataset, False, args.world_size, args.rank)
    caption_dataloader = create_caption_loader(caption_dataset, caption_samplers, args.sample_batch_size, 8)
    print('number of selected samples: %d' % len(caption_dataset))

    if args.distributed:
        caption_samplers.set_epoch(0)

    print("Start caption")
    start_time = time.time()
    data_to_caption_new(model, caption_dataloader, epoch, args, save_dir, alphabet)
    if args.distributed:
        dist.barrier()

    total_time = time.time() - start_time
    print(f'caption time {total_time / 60.0} mins')

    delete_model(model)  # to save memory
    print("caption raw data done!")

    print("merge captioned dataset!")

    try:
        subprocess.run([
            "python", "merge_hdf5_new.py",
            "--input_prefix", save_dir,
            "--output_path", f"{save_dir}.hdf5",
            "--rank", str(args.world_size)
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"The command '{e.cmd}' failed with return code {e.returncode}")

    print("merge captioned inds!")

    try:
        subprocess.run([
            "python", "merge_caption_inds.py",
            "--input_prefix", inds_dir,
            "--output_path", f"{inds_dir}.hdf5",
            "--rank", str(args.world_size)
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"The command '{e.cmd}' failed with return code {e.returncode}")

    print("caption done!")

def caption_sw(model, args, epoch, alphabet):
    with open(os.path.join('path/to/go_dict_7533.csv')) as inp:
        reader = csv.reader(inp)
        original_annotation_index_to_common_annotation_index = {rows[0]: rows[1] for rows in reader}
        index_to_annotation = {val: key for key, val in original_annotation_index_to_common_annotation_index.items()}
    pre_dir = 'path/to/swissprot_caption.csv'
    df = pd.read_csv(pre_dir)
    print("Start caption")
    start_time = time.time()

    seq_encoder = get_module_by_name(model, "seq_encoder")
    anno_encoder = get_module_by_name(model, "anno_encoder")
    anno_decoder = get_module_by_name(model, "anno_decoder")
    to_annotation_logits = get_module_by_name(model, "to_annotation_logits")
    itm_head = get_module_by_name(model, "itm_head")

    model.eval()

    caption_list = []
    chunk_counter = 0
    truncation_seq_length = args.seq_len

    uni_anno_accuracy = 0.0
    uni_anno_precision = 0.0
    uni_anno_recall = 0.0
    uni_anno_fmax = 0.0
    uni_anno_auc = 0.0
    uni_cnt = 0

    sw_anno_accuracy = 0.0
    sw_anno_precision = 0.0
    sw_anno_recall = 0.0
    sw_anno_fmax = 0.0
    sw_anno_auc = 0.0
    sw_cnt = 0

    with open(f'./caption_sw/esm_35m/sw_cleaned_vs_updated_go_tiny_epoch{args.actual_epoch}.csv', 'w', newline='') as csvfile:
        # Create a CSV writer object
        fieldnames = ['entry', 'raw GO', 'cleaned go num','cleaned GO', 'captioned', 'updated GO']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # Write column headers
        cnt = 0
        writer.writeheader()
        with torch.no_grad():
            for index, row in tqdm(df.iterrows()):
                seqs = row["seq"]
                go_list = row["uniref50 GO"]
                
                annotation_masks = np.zeros(args.num_annotation)
                go_list = go_list.strip('[')
                go_list = go_list.strip(']')
                go_list = go_list.split(', ')
                for annotation in go_list:
                    annotation = annotation.strip("'")
                    if str(annotation) in set(original_annotation_index_to_common_annotation_index):
                        annotation_masks[
                            int(original_annotation_index_to_common_annotation_index[annotation])] = 1
                uni50_anno = torch.tensor(annotation_masks).float().to(args.gpu)

                sw_go_list = row["swissprot GO"]
                sw_annotation_masks = np.zeros(args.num_annotation)
                sw_go_list = sw_go_list.strip('[')
                sw_go_list = sw_go_list.strip(']')
                sw_go_list = sw_go_list.split(', ')
                for annotation in sw_go_list:
                    annotation = annotation.strip("'")
                    if str(annotation) in set(original_annotation_index_to_common_annotation_index):
                        sw_annotation_masks[
                            int(original_annotation_index_to_common_annotation_index[annotation])] = 1
                sw_anno = torch.tensor(sw_annotation_masks).float().to(args.gpu)
                if not np.array_equal(annotation_masks, sw_annotation_masks):
                    uni_cnt+=1

                seq_lens = len(seqs)
                seq_encoded_list = alphabet.encode(seqs)
                if truncation_seq_length:
                    seq_encoded_list = seq_encoded_list[:truncation_seq_length]
                tokens = torch.empty(
                    (
                        args.seq_len + int(alphabet.prepend_bos) + int(alphabet.append_eos),
                    ),
                    dtype=torch.int64,
                )
                tokens.fill_(alphabet.padding_idx)

                if alphabet.prepend_bos:
                    tokens[0] = alphabet.cls_idx
                seq = torch.tensor(seq_encoded_list, dtype=torch.int64)
                tokens[
                int(alphabet.prepend_bos): len(seq_encoded_list)
                                           + int(alphabet.prepend_bos)
                ] = seq
                if alphabet.append_eos:
                    tokens[len(seq_encoded_list) + int(alphabet.prepend_bos)] = alphabet.eos_idx
                input_seq = torch.tensor(tokens, dtype=torch.int).to(args.gpu)
                input_anno = torch.tensor(annotation_masks).float().to(args.gpu)

                results = seq_encoder(input_seq.unsqueeze(0), repr_layers=[args.esm_layer], return_contacts=False)
                seq_hidden_embeds = results["representations"][args.esm_layer]
                decoder_output = anno_decoder(torch.zeros_like(input_anno).unsqueeze(0), seq_embed=seq_hidden_embeds)
                predict_anno = to_annotation_logits(decoder_output)
                anno_hat = torch.sigmoid(predict_anno).to(input_anno.device)

                uni50_anno_accuracy = BinaryAccuracy(threshold=0.5)(anno_hat.squeeze(0).to('cpu'), uni50_anno.to('cpu'))
                uni50_anno_precision = BinaryPrecision(threshold=0.5)(anno_hat.squeeze(0).to('cpu'), uni50_anno.to('cpu'))
                uni50_anno_recall = BinaryRecall(threshold=0.5)(anno_hat.squeeze(0).to('cpu'), uni50_anno.to('cpu'))
                uni50_anno_fmax = 0.0
                for cut in (c / 20 for c in range(21)):
                    uni50_anno_fmax = max(uni50_anno_fmax, BinaryF1Score(threshold=cut)(anno_hat.squeeze(0).to('cpu'), uni50_anno.to('cpu')))
                uni50_anno_auc = AUROC(task="binary")(anno_hat.squeeze(0).to('cpu'), uni50_anno.to('cpu'))

                uni_anno_accuracy += uni50_anno_accuracy
                uni_anno_precision += uni50_anno_precision
                uni_anno_recall += uni50_anno_recall
                uni_anno_fmax += uni50_anno_fmax
                uni_anno_auc += uni50_anno_auc

                anno_accuracy = BinaryAccuracy(threshold=0.5)(anno_hat.squeeze(0).to('cpu'), sw_anno.to('cpu'))
                anno_precision = BinaryPrecision(threshold=0.5)(anno_hat.squeeze(0).to('cpu'), sw_anno.to('cpu'))
                anno_recall = BinaryRecall(threshold=0.5)(anno_hat.squeeze(0).to('cpu'), sw_anno.to('cpu'))
                anno_fmax = 0.0
                for cut in (c / 20 for c in range(21)):
                    anno_fmax = max(anno_fmax, BinaryF1Score(threshold=cut)(anno_hat.squeeze(0).to('cpu'), sw_anno.to('cpu')))
                anno_auc = AUROC(task="binary")(anno_hat.squeeze(0).to('cpu'), sw_anno.to('cpu'))

                sw_anno_accuracy += anno_accuracy
                sw_anno_precision += anno_precision
                sw_anno_recall += anno_recall
                sw_anno_fmax += anno_fmax
                sw_anno_auc += anno_auc
                
                sw_cnt += 1

                output_pos_true = anno_encoder(input_anno.unsqueeze(0), seq_embed=seq_hidden_embeds)
                output_pos_hat = anno_encoder(anno_hat.round(), seq_embed=seq_hidden_embeds)

                vl_output_true = itm_head(output_pos_true)  # [bs, 2]
                vl_output_hat = itm_head(output_pos_hat)  # [bs, 2]

                mask = (vl_output_hat[:, 1] > vl_output_true[:, 1]) & (vl_output_hat[:, 1] > vl_output_hat[:, 0])
                if mask:
                    cnt += 1
                    captioned = "true"
                else:
                    captioned = "false"
                new_kw_list = np.nonzero(anno_hat.round().squeeze(0).detach().cpu().numpy())[0]
                new_anno_list = []
                for annotation in new_kw_list:
                    new_anno_list.append(index_to_annotation[str(annotation)])

                row_dict = {'entry': row["entry"],
                        'raw GO': row["uniref50 GO"],
                        'cleaned go num':len(new_anno_list),
                        'cleaned GO': new_anno_list,
                        'captioned': captioned,
                        'updated GO': row["swissprot GO"]}
                writer.writerow(row_dict)
            sw_anno_accuracy /= sw_cnt
            sw_anno_precision /= sw_cnt
            sw_anno_recall /= sw_cnt
            sw_anno_fmax /= sw_cnt
            sw_anno_auc /= sw_cnt
            row_dict = {'sw_anno_accuracy': sw_anno_accuracy,
                        'sw_anno_precision': sw_anno_precision,
                        'sw_anno_recall': sw_anno_recall,
                        'sw_anno_fmax': sw_anno_fmax,
                        'sw_anno_auc': sw_anno_auc,
                        'sw_cnt': sw_cnt}
            print(row_dict)

            uni_anno_accuracy /= sw_cnt
            uni_anno_precision /= sw_cnt
            uni_anno_recall /= sw_cnt
            uni_anno_fmax /= sw_cnt
            uni_anno_auc /= sw_cnt
            uni_row_dict = {'uni_anno_accuracy': uni_anno_accuracy,
                        'uni_anno_precision': uni_anno_precision,
                        'uni_anno_recall': uni_anno_recall,
                        'uni_anno_fmax': uni_anno_fmax,
                        'uni_anno_auc': uni_anno_auc,
                        'uni_cnt': uni_cnt}
            print(uni_row_dict)
            # writer.writerow(row_dict)
    print(cnt)
    print("caption done!")


def create_model(args, seq_model, alphabet, checkpoint=None):
    print("Creating model")

    model = protac_pretrain_seq_anno_mlm_esm2_new(seq_model=seq_model, alphabet=alphabet, args=args)
    print('Successfully loading protac_pretrain_seq_anno_mlm_esm2_new model')

    total_params = sum(p.numel() for p in model.parameters())
    print(args.version + f" total number of parameters:{total_params / (1024 * 1024)}M.")
    model = model.cuda()

    if args.actual_epoch == 0:
        print("freeze seq model!")
        Chain = chain(model.seq_encoder.parameters())
        for param in Chain:
            param.requires_grad = False

    optimizer_pretrain = torch.optim.AdamW(params=model.parameters(), lr=args.init_lr,
                                           weight_decay=args.weight_decay)
    optimizer_finetune = torch.optim.AdamW(params=model.parameters(), lr=args.init_lr,
                                           weight_decay=args.weight_decay)

    if checkpoint == None:
        if os.path.exists(args.init_checkpoint_path):
            print("random init!")
            print(args.init_checkpoint_path)
            checkpoint = torch.load(args.init_checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint, strict=True)
        else:
            torch.save(model.state_dict(), args.init_checkpoint_path)
        start_epoch = 0
    else:
        print(f"load ckpt: {checkpoint}!")
        checkpoint = torch.load(checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        start_epoch = checkpoint['epoch']

        state_dict = on_load_checkpoint(state_dict)
        model.load_state_dict(state_dict, strict=True)

    model_without_ddp = model
    model = torch.compile(model)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          find_unused_parameters=True)
        model_without_ddp = model.module._orig_mod
        dist.barrier()
    else:
        model_without_ddp = model._orig_mod

    return model, model_without_ddp, start_epoch, optimizer_pretrain, optimizer_finetune

def create_model_kw(args, seq_model, alphabet, checkpoint=None):
    print("Creating model")
    model = protac_pretrain_v5_kw(seq_model=seq_model, alphabet=alphabet, args=args)
    print('Successfully loading Protein-BLIP-V5-kw model')

    total_params = sum(p.numel() for p in model.parameters())
    print(args.version + f" total number of parameters:{total_params / (1024 * 1024)}M.")
    model = model.cuda()

    optimizer_pretrain = torch.optim.AdamW(params=model.parameters(), lr=args.init_lr,
                                           weight_decay=args.weight_decay)

    keys_to_extract = ['seq_encoder', 'anno_decoder', 'to_annotation_logits']

    print(f"load ckpt: {checkpoint}!")
    if checkpoint == None:
        if os.path.exists(args.init_checkpoint_path_kw):
            print("random init!")
            print(args.init_checkpoint_path_kw)
            checkpoint = torch.load(args.init_checkpoint_path_kw, map_location='cpu')
            model.load_state_dict(checkpoint, strict=True)
        else:
            torch.save(model.state_dict(), args.init_checkpoint_path_kw)
        start_epoch = 0
    else:
        checkpoint = torch.load(checkpoint, map_location='cpu')
        # state_dict = checkpoint
        state_dict = checkpoint['model']

        state_dict = on_load_checkpoint(state_dict)
        extracted_weights = {}

        for key in keys_to_extract:
            extracted_weights[key] = {}
            for k, v in state_dict.items():
                if k.startswith(key):
                    extracted_weights[key][k[len(key) + 1:]] = v
        for key, weights in extracted_weights.items():
            if hasattr(model, key):
                print(f'load {key}')
                getattr(model, key).load_state_dict(weights, strict=True)

    Chain = chain(model.seq_encoder.parameters(), model.anno_decoder.parameters(),
                  model.to_annotation_logits.parameters())
    for param in Chain:
        param.requires_grad = False
    model_without_ddp = model
    model = torch.compile(model)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          find_unused_parameters=True)
        model_without_ddp = model.module._orig_mod

    if args.distributed: dist.barrier()
    return model, model_without_ddp, optimizer_pretrain


def delete_model(model):
    """
    Cleanup and release the resources of a model, especially when using GPUs.

    Parameters:
        model (torch.nn.Module): The PyTorch model to be cleaned up and released.
    """
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module._orig_mod  # model after compiled

    model.cpu()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def on_load_checkpoint(state_dict):
    keys_list = list(state_dict.keys())
    for key in keys_list:
        if 'orig_mod.' in key:
            deal_key = key.replace('_orig_mod.', '')
            state_dict[deal_key] = state_dict[key]
            del state_dict[key]
    return state_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## dataset settings
    parser.add_argument('--dataset', default="uniref50_2018", type=str)  # or eval or test or downstream

    ## model settings
    parser.add_argument('--version', default="esm2_35m", type=str,
                        choices=['esm2_8m', 'esm2_35m', 'esm2_150m', 'esm2_650m', 'pb'],
                        help='[esm2_8m, esm2_35m, esm2_150m, esm2_650m, pb]')
    parser.add_argument('--from_scratch', default=False, type=bool)

    parser.add_argument('--alpha', default=0.4, type=float)
    parser.add_argument('--anno_lm_loss_weight', default=1., type=float)
    parser.add_argument('--seq_lm_loss_weight', default=1., type=float)
    parser.add_argument('--itc_loss_weight', default=1., type=float)
    parser.add_argument('--itm_loss_weight', default=1., type=float)

    ## structure settings
    parser.add_argument('--esm_layer', default=12, type=int)
    parser.add_argument('--seq_len', default=512, type=int)
    parser.add_argument('--num_annotation', default=7533, type=int)
    parser.add_argument('--attn_dim_head', default=64, type=int)
    parser.add_argument('--attn_heads', default=8, type=int)  # 35m, 150m: 8
    parser.add_argument('--depth', default=12, type=int)  # 35m, 150m: 12
    parser.add_argument('--dim', default=128, type=int)
    parser.add_argument('--dim_global', default=512, type=int)
    parser.add_argument('--wide_conv_dilation', default=5, type=int)
    parser.add_argument('--wide_conv_kernel', default=9, type=int)
    parser.add_argument('--glu_conv', default=False, type=bool)
    parser.add_argument('--local_self_attn', default=True, type=bool)
    parser.add_argument('--narrow_conv_kernel', default=9, type=int)
    parser.add_argument('--num_global_tokens', default=1, type=int)
    parser.add_argument('--seq_sample_token_prob', default=0.15, type=float)
    parser.add_argument('--seq_mask_token_prob', default=0.8, type=float)
    parser.add_argument('--random_replace_token_prob', default=0.1, type=float)
    parser.add_argument('--remove_annotation_prob', default=0.25, type=float)
    parser.add_argument('--add_annotation_prob', default=0.0001, type=float)
    parser.add_argument('--remove_all_annotations_prob', default=1.0, type=float)  # before: 0.5

    ## training settings
    parser.add_argument('--ft_first', default=False, type=int)
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--mode', default="clean", type=str)  # or eval or test or downstream
    parser.add_argument('--epoch', default=1, type=int)
    parser.add_argument('--actual_epoch', default=0, type=int)
    parser.add_argument('--pretrain_epoch', default=1, type=int)
    parser.add_argument('--fewshot_epoch', default=500, type=int) # 200
    parser.add_argument('--finetune_epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--sample_batch_size', default=512, type=int)
    parser.add_argument('--test_batch_size', default=32, type=int)

    parser.add_argument('--warmup_steps', default=3000, type=float)
    parser.add_argument('--warmup_lr', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--init_lr', default=2e-5, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--lr_decay_rate', default=0.9, type=float)

    ### dir settings
    parser.add_argument('--output_dir',
                        default='path/to/output_dir')  
    parser.add_argument('--captioned_seq_sav_dir',
                        default='path/to/captioned_seq_sav_dir')
    parser.add_argument('--captioned_ind_sav_dir',
                        default='path/to/captioned_ind_sav_dir')
    parser.add_argument('--checkpoint',
                        default=None)

    ## system settings
    parser.add_argument('--use_wandb', default=False, type=bool)
    parser.add_argument('--wandb_project_name', default='proteinblip-cleaning-esm-2024-new', type=str,
                        help='wandb project name')  # problip-downstream
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--device_id', default=[0, 1], type=list)
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.captioned_seq_sav_dir).mkdir(parents=True, exist_ok=True)
    Path(args.captioned_ind_sav_dir).mkdir(parents=True, exist_ok=True)
    main(args)
