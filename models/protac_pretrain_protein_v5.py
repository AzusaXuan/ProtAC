from torchvision.ops.focal_loss import sigmoid_focal_loss as focal_loss
import math
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from models.protac_create_module import create_proteinBERT
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy, BinaryRecall, BinaryPrecision, AUROC

class ProtAC_Pretrain_V5_kw(nn.Module):
    def __init__(self,
                 seq_model=None,
                 alphabet=None,
                 args=None
                 ):

        super().__init__()
        self.esm_layer = args.esm_layer
        self.seq_encoder, self.alphabet = seq_model, alphabet
        args.dim = self.seq_encoder.embed_dim
        self.seq_bert_dim = self.seq_encoder.embed_dim
        self.anno_bert_dim = args.dim_global
        self.num_tokens = len(self.alphabet.all_toks)
        self.n_annotation = args.num_annotation
        ## When Encoding, the protein seq and annotation mask probablity are 0!
        # create the decoder
        self.anno_decoder = create_proteinBERT("anno_bert_lm", args)
        self.to_annotation_logits = nn.Linear(args.dim_global, self.n_annotation)
        self.anno_to_kw_linear = nn.Linear(self.n_annotation, 773)
        # clip
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, protein_seq, kw):
        bs = protein_seq.size(0)
        results = self.seq_encoder(protein_seq, repr_layers=[self.esm_layer], return_contacts=True)
        seq_hidden_embeds = results["representations"][self.esm_layer]
        # seq_logits = results["logits"]
        ##================= LM ========================##
        kw_zero = torch.zeros([bs, self.n_annotation]).to(kw.device)
        decoder_targets = kw.clone()

        decoder_output = self.anno_decoder(kw_zero, seq_embed=seq_hidden_embeds)

        predict_anno = self.to_annotation_logits(decoder_output)
        predict_kw = self.anno_to_kw_linear(predict_anno)

        loss_lm_anno = focal_loss(predict_kw,
                                  decoder_targets.to(device=predict_kw.device),
                                  alpha=0.75,
                                  gamma=2,
                                  reduction='mean')

        anno_hat = torch.clamp(predict_kw, 0, 1).to(kw.device)
        anno_accuracy = BinaryAccuracy(threshold=0.5)(anno_hat.to('cpu'), kw.to('cpu'))
        anno_precision = BinaryPrecision(threshold=0.5)(anno_hat.to('cpu'), kw.to('cpu'))
        anno_recall = BinaryRecall(threshold=0.5)(anno_hat.to('cpu'), kw.to('cpu'))
        anno_fmax = 0.0
        for cut in (c / 20 for c in range(21)):
            anno_fmax = max(anno_fmax, BinaryF1Score(threshold=cut)(anno_hat.to('cpu'), kw.to('cpu')))

        anno_auc = AUROC(task="binary")(anno_hat.to('cpu'), kw.to('cpu'))

        return loss_lm_anno, [anno_accuracy, anno_precision, anno_recall, anno_fmax, anno_auc]


def protac_pretrain_v5_kw(**kwargs):
    model = ProtAC_Pretrain_V5_kw(**kwargs)
    return model


class ProtAC_Pretrain_V5_kw_pb(nn.Module):
    def __init__(self,
                 args=None
                 ):

        super().__init__()
        self.seq_encoder = create_proteinBERT("seq_bert_itc", args)
        self.seq_bert_dim = args.dim
        self.anno_bert_dim = args.dim_global
        self.n_annotation = args.num_annotation
        ## When Encoding, the protein seq and annotation mask probablity are 0!
        # create the decoder
        self.anno_decoder = create_proteinBERT("anno_bert_lm", args)
        self.to_annotation_logits = nn.Linear(args.dim_global, self.n_annotation)
        self.anno_to_kw_linear = nn.Linear(self.n_annotation, 773)
        # clip
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, protein_seq, kw):
        bs = protein_seq.size(0)
        seq_hidden_embeds = self.seq_encoder(protein_seq)  ## return (batch_size, seq_len=512, seq_embed_dim=512)
        ##================= LM ========================##
        kw_zero = torch.zeros([bs, self.n_annotation]).to(kw.device)
        decoder_targets = kw.clone()

        decoder_output = self.anno_decoder(kw_zero, seq_embed=seq_hidden_embeds)

        predict_anno = self.to_annotation_logits(decoder_output)
        predict_kw = self.anno_to_kw_linear(predict_anno)

        loss_lm_anno = focal_loss(predict_kw,
                                  decoder_targets.to(device=predict_kw.device),
                                  alpha=0.75,
                                  gamma=2,
                                  reduction='mean')

        anno_hat = torch.clamp(predict_kw, 0, 1).to(kw.device)
        anno_accuracy = BinaryAccuracy(threshold=0.5)(anno_hat.to('cpu'), kw.to('cpu'))
        anno_precision = BinaryPrecision(threshold=0.5)(anno_hat.to('cpu'), kw.to('cpu'))
        anno_recall = BinaryRecall(threshold=0.5)(anno_hat.to('cpu'), kw.to('cpu'))
        anno_fmax = 0.0
        for cut in (c / 20 for c in range(21)):
            anno_fmax = max(anno_fmax, BinaryF1Score(threshold=cut)(anno_hat.to('cpu'), kw.to('cpu')))

        anno_auc = AUROC(task="binary")(anno_hat.to('cpu'), kw.to('cpu'))

        return loss_lm_anno, [anno_accuracy, anno_precision, anno_recall, anno_fmax, anno_auc]


def protac_pretrain_v5_kw_pb(**kwargs):
    model = ProtAC_Pretrain_V5_kw_pb(**kwargs)
    return model


class ProtAC_Pretrain_Seq_Anno_MLM(nn.Module):
    def __init__(self,
                 args=None
                 ):

        super().__init__()

        self.seq_bert_dim = args.dim
        self.anno_bert_dim = args.dim_global
        self.num_tokens = args.num_tokens
        self.n_annotation = args.num_annotation
        self.exclude_token_ids = args.exclude_token_ids
        self.random_replace_token_prob = args.random_replace_token_prob
        self.remove_all_annotations_prob = args.remove_all_annotations_prob
        self.remove_annotation_prob = args.remove_annotation_prob
        self.add_annotation_prob = args.add_annotation_prob

        self.seq_encoder = create_proteinBERT("seq_bert_itc", args)
        self.seq_globatten = nn.Sequential(
            nn.Linear(self.seq_bert_dim, 1), nn.Softmax(dim=1)  ## glob_attn_module foe seq embeddings
        )
        self.seq_proj = nn.Linear(self.seq_bert_dim, 256)

        self.anno_encoder = create_proteinBERT("anno_bert_itc", args)

        self.anno_proj = nn.Linear(self.anno_bert_dim, 256)

        self.itm_head = nn.Linear(self.anno_bert_dim, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        # create the decoder
        self.seq_decoder = create_proteinBERT("seq_bert_lm", args)
        self.to_seq_logits = nn.Linear(args.dim, self.num_tokens)
        self.to_seq_logits_mlm = nn.Linear(args.dim, self.num_tokens)

        self.anno_decoder = create_proteinBERT("anno_bert_lm", args)
        self.to_annotation_logits = nn.Linear(args.dim_global, self.n_annotation)

        # clip
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, protein_seq, anno, seq_lens, mode="train", mask=None):
        bs = protein_seq.size(0)
        seq_len = protein_seq.size(1)

        device = protein_seq.device

        if not exists(mask):
            mask = torch.ones_like(protein_seq, device=device, dtype=torch.bool)

        # prepare masks for noising sequence

        excluded_tokens_mask = mask

        for token_id in torch.tensor(self.exclude_token_ids):
            excluded_tokens_mask = excluded_tokens_mask & (protein_seq != token_id)

        random_replace_token_prob_mask = get_mask_subset_with_prob(excluded_tokens_mask, self.random_replace_token_prob)

        # prepare masks for noising annotation

        batch_mask = torch.ones(bs, device=device, dtype=torch.bool)
        batch_mask = rearrange(batch_mask, 'b -> b ()')
        remove_annotation_from_batch_mask = get_mask_subset_with_prob(batch_mask, self.remove_all_annotations_prob)

        annotation_mask = anno > 0
        remove_annotation_prob_mask = get_mask_subset_with_prob(annotation_mask, self.remove_annotation_prob)
        add_annotation_prob_mask = get_mask_subset_with_prob(~annotation_mask, self.add_annotation_prob)
        remove_annotation_mask = remove_annotation_from_batch_mask & remove_annotation_prob_mask

        # generate random tokens

        random_tokens = torch.randint(0, self.num_tokens, protein_seq.shape, device=protein_seq.device)

        for token_id in self.exclude_token_ids:
            random_replace_token_prob_mask = random_replace_token_prob_mask & (
                    random_tokens != token_id)  # make sure you never substitute a token with an excluded token type (pad, start, end)

        # noise sequence

        noised_seq = torch.where(random_replace_token_prob_mask, random_tokens, protein_seq)

        # noise annotation

        noised_annotation = anno + add_annotation_prob_mask.type(anno.dtype)
        noised_annotation = noised_annotation * remove_annotation_mask.type(anno.dtype)

        if mode == "test":
            encoder_input_seq = protein_seq.clone()
            encoder_input_anno = anno.clone()
        else:
            encoder_input_seq = noised_seq.clone()
            encoder_input_anno = noised_annotation.clone()

        seq_hidden_embeds = self.seq_encoder(encoder_input_seq,
                                             mask)  ## return (batch_size, seq_len=512, seq_embed_dim=512)
        anno_embeds = self.anno_encoder(encoder_input_anno, mask)  ## return (batch_size, anno_len=256)


        ###============== seq-anno Matching ===================###

        # forward the positve seq-anno pair
        output_pos = self.anno_encoder(encoder_input_anno, mask, seq_embed=seq_hidden_embeds)
        with torch.no_grad():
            weights_t2i = torch.zeros((bs, bs)) + 1e-4
            weights_t2i.fill_diagonal_(0)
            weights_i2t = torch.zeros((bs, bs)) + 1e-4
            weights_i2t.fill_diagonal_(0)

        # select a negative protein-seq for each anno
        seq_hidden_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            seq_hidden_embeds_neg.append(seq_hidden_embeds[neg_idx])
        seq_hidden_embeds_neg = torch.stack(seq_hidden_embeds_neg, dim=0)
        # select a negative anno for each seq
        anno_ids_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            anno_ids_neg.append(encoder_input_anno[neg_idx])
        anno_ids_neg = torch.stack(anno_ids_neg, dim=0)
        anno_ids_all = torch.cat([encoder_input_anno, anno_ids_neg], dim=0)
        seq_hidden_embeds_all = torch.cat([seq_hidden_embeds_neg, seq_hidden_embeds], dim=0)
        output_neg = self.anno_encoder(anno_ids_all, mask, seq_embed=seq_hidden_embeds_all)
        vl_embeddings = torch.cat([output_pos, output_neg], dim=0)
        vl_output = self.itm_head(vl_embeddings)
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                                dim=0).to(protein_seq.device)
        itm_labels_one_hot = F.one_hot(itm_labels).float()
        loss_itm = focal_loss(vl_output, itm_labels_one_hot, reduction="mean", alpha=0.33, gamma=3)


        ItmAccuracy = BinaryAccuracy(threshold=0.5).to(itm_labels.device)
        itm_accuracy = ItmAccuracy(itm_labels, vl_output.argmax(dim=-1))



        ##=================Anno LM ========================##

        decoder_targets = anno.clone()

        decoder_output = self.anno_decoder(torch.zeros_like(anno), mask, seq_embed=seq_hidden_embeds)

        predict_anno = self.to_annotation_logits(decoder_output)
        loss_lm_anno = focal_loss(predict_anno,
                                    decoder_targets.to(device=predict_anno.device),
                                    alpha=0.75,
                                    gamma=2,
                                    reduction='mean')

        anno_hat = torch.clamp(predict_anno, 0, 1).to(anno.device)
        anno_accuracy = BinaryAccuracy(threshold=0.5)(anno_hat.to('cpu'), anno.to('cpu'))
        anno_precision = BinaryPrecision(threshold=0.5)(anno_hat.to('cpu'), anno.to('cpu'))
        anno_recall = BinaryRecall(threshold=0.5)(anno_hat.to('cpu'), anno.to('cpu'))
        anno_fmax = 0.0
        for cut in (c / 20 for c in range(21)):
            anno_fmax = max(anno_fmax, BinaryF1Score(threshold=cut)(anno_hat.to('cpu'), anno.to('cpu')))

        anno_auc = AUROC(task="binary")(anno_hat.to('cpu'), anno.to('cpu'))

        ##=================Seq LM ========================##

        protein_seq_targets = protein_seq.clone()  ## [bs, 512]
        seq_emb = self.seq_decoder(encoder_input_seq, mask, anno_embed=anno_embeds)  ## [bs, 512, 512]
        predict_seq = self.to_seq_logits(seq_emb)  ## [bs, 512, 26]
        loss_lm_seq = F.cross_entropy(predict_seq.permute(0, 2, 1),
                                        protein_seq_targets.to(device=predict_seq.device).long(), reduction='mean')

        seq_difference = protein_seq_targets.to(device=predict_seq.device) - predict_seq.argmax(dim=-1)
        seq_difference = seq_difference.count_nonzero() / protein_seq.shape[0] / protein_seq.shape[1]

        # loss MLM
        predict_seq_mlm = self.to_seq_logits_mlm(seq_hidden_embeds)  ## [bs, 512, 26]
        loss_lm_seq_mlm = F.cross_entropy(predict_seq_mlm.permute(0, 2, 1),
                                            protein_seq_targets.to(device=predict_seq_mlm.device).long(),
                                            reduction='mean')

        seq_difference_mlm = protein_seq_targets.to(device=predict_seq_mlm.device) - predict_seq_mlm.argmax(dim=-1)
        seq_difference_mlm = seq_difference_mlm.count_nonzero() / protein_seq.shape[0] / protein_seq.shape[1]


        return loss_itm, loss_lm_anno, loss_lm_seq, loss_lm_seq_mlm, [anno_accuracy, anno_precision,
                                                                                anno_recall, anno_fmax,
                                                                                anno_auc, seq_difference,
                                                                                seq_difference_mlm]


def protac_pretrain_seq_anno_mlm(**kwargs):
    model = ProtAC_Pretrain_Seq_Anno_MLM(**kwargs)
    return model


class ProtAC_Pretrain_Seq_Anno_MLM_esm2_new(nn.Module):
    def __init__(self,
                 seq_model=None,
                 alphabet=None,
                 args=None
                 ):
        super().__init__()
        self.esm_layer = args.esm_layer
        self.seq_encoder, self.alphabet = seq_model, alphabet
        args.dim = self.seq_encoder.embed_dim
        self.seq_bert_dim = self.seq_encoder.embed_dim
        self.anno_bert_dim = args.dim_global
        self.num_tokens = len(self.alphabet.all_toks)
        self.n_annotation = args.num_annotation
        self.seq_sample_token_prob = args.seq_sample_token_prob
        self.seq_mask_token_prob = args.seq_mask_token_prob
        self.random_replace_token_prob = args.random_replace_token_prob
        self.remove_all_annotations_prob = args.remove_all_annotations_prob
        self.remove_annotation_prob = args.remove_annotation_prob
        self.add_annotation_prob = args.add_annotation_prob

        self.anno_encoder = create_proteinBERT("anno_bert_itc", args)

        self.anno_proj = nn.Linear(self.anno_bert_dim, self.seq_bert_dim)

        self.itm_head = nn.Linear(self.anno_bert_dim, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.anno_decoder = create_proteinBERT("anno_bert_lm", args)
        self.to_annotation_logits = nn.Linear(args.dim_global, self.n_annotation)

        # clip
        self.labels = None
        self.last_local_batch_size = None

        # adaptive loss
        if args.actual_epoch > 0 and args.mode == "train":
            self.use_adaptive_loss = True
        else:
            self.use_adaptive_loss = False

    def forward(self, protein_seq, anno, seq_lens, mode="train", mask=None):
        bs = protein_seq.size(0)
        seq_len = protein_seq.size(1)
        device = protein_seq.device

        if not exists(mask):
            mask = torch.ones_like(protein_seq, device=device, dtype=torch.bool)

        # prepare masks for noising sequence

        excluded_tokens_mask = mask

        for token_id in self.alphabet.all_special_tokens:
            special_token_ids = self.alphabet.get_idx(token_id)
            excluded_tokens_mask = excluded_tokens_mask & (protein_seq != special_token_ids)

        seq_sample_token_prob_mask = new_get_mask_subset_with_prob(excluded_tokens_mask, self.seq_sample_token_prob)
        # print((excluded_tokens_mask&seq_sample_token_prob_mask).sum(), seq_sample_token_prob_mask.sum())

        seq_mask_token_prob_mask = new_get_mask_subset_with_prob(seq_sample_token_prob_mask, self.seq_mask_token_prob)
        # print((seq_sample_token_prob_mask&seq_mask_token_prob_mask).sum(), seq_mask_token_prob_mask.sum())

        seq_no_mask_token_prob_mask = seq_sample_token_prob_mask & (~seq_mask_token_prob_mask)
        random_replace_token_prob_mask = new_get_mask_subset_with_prob(seq_no_mask_token_prob_mask,
                                                                       self.random_replace_token_prob / (
                                                                               1 - self.seq_mask_token_prob))

        batch_mask = torch.ones(bs, device=device, dtype=torch.bool)
        batch_mask = rearrange(batch_mask, 'b -> b ()')
        remove_annotation_from_batch_mask = new_get_mask_subset_with_prob(batch_mask,
                                                                          1 - self.remove_all_annotations_prob)

        annotation_mask = anno > 0
        remove_annotation_prob_mask = new_get_mask_subset_with_prob(annotation_mask, 1 - self.remove_annotation_prob)
        # print(remove_annotation_from_batch_mask, torch.count_nonzero(remove_annotation_from_batch_mask))
        # sys.exit()
        add_annotation_prob_mask = new_get_mask_subset_with_prob(~annotation_mask, self.add_annotation_prob)
        removed_annotation_mask = remove_annotation_from_batch_mask & remove_annotation_prob_mask

        # generate random tokens
        mask_tokens = torch.full(protein_seq.shape, self.alphabet.mask_idx, dtype=torch.int, device=protein_seq.device)
        random_tokens = torch.randint(0, self.num_tokens, protein_seq.shape, device=protein_seq.device)

        for token_id in self.alphabet.all_special_tokens:
            special_token_ids = self.alphabet.get_idx(token_id)
            random_replace_token_prob_mask = random_replace_token_prob_mask & (
                    random_tokens != special_token_ids)  # make sure you never substitute a token with an excluded token type (pad, start, end)

        # noise sequence

        masked_seq = torch.where(seq_mask_token_prob_mask, mask_tokens, protein_seq)
        noised_seq = torch.where(random_replace_token_prob_mask, random_tokens, masked_seq)

        # noise annotation

        noised_annotation = anno + add_annotation_prob_mask.type(anno.dtype)
        noised_annotation = noised_annotation * removed_annotation_mask.type(anno.dtype)

        if mode == "test":
            encoder_input_seq = protein_seq.clone()
            encoder_input_anno = torch.zeros_like(anno)
        else:
            encoder_input_seq = noised_seq.clone()
            encoder_input_anno = noised_annotation.clone()

        results = self.seq_encoder(encoder_input_seq, repr_layers=[self.esm_layer], return_contacts=True)
        seq_hidden_embeds = results["representations"][self.esm_layer]
        seq_logits = results["logits"]
        ## seq_hidden_embeds (batch_size, seq_len=512+2=514, seq_embed_dim)
        ## seq_logits (batch_size, seq_len=512+2=514, num_token=33)

        ###============== seq-anno Matching ===================###


        # forward the positve seq-anno pair
        output_pos = self.anno_encoder(anno, mask, seq_embed=seq_hidden_embeds)
        with torch.no_grad():
            weights_t2i = torch.zeros((bs, bs)) + 1e-4
            weights_t2i.fill_diagonal_(0)
            weights_i2t = torch.zeros((bs, bs)) + 1e-4
            weights_i2t.fill_diagonal_(0)


        seq_hidden_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            seq_hidden_embeds_neg.append(seq_hidden_embeds[neg_idx])
        seq_hidden_embeds_neg = torch.stack(seq_hidden_embeds_neg, dim=0)

        # select a negative anno for each seq
        anno_ids_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            anno_ids_neg.append(anno[neg_idx])
        anno_ids_neg = torch.stack(anno_ids_neg, dim=0)

        anno_ids_all = torch.cat([anno, anno_ids_neg], dim=0)

        seq_hidden_embeds_all = torch.cat([seq_hidden_embeds_neg, seq_hidden_embeds], dim=0)

        output_neg = self.anno_encoder(anno_ids_all, mask, seq_embed=seq_hidden_embeds_all)

        vl_embeddings = torch.cat([output_pos, output_neg], dim=0)

        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                                dim=0).to(protein_seq.device)
        itm_labels_one_hot = F.one_hot(itm_labels).float()
        loss_itm = focal_loss(vl_output, itm_labels_one_hot, reduction="mean", alpha=0.33, gamma=3)

        ItmAccuracy = BinaryAccuracy(threshold=0.5).to(itm_labels.device)
        itm_accuracy = ItmAccuracy(itm_labels, vl_output.argmax(dim=-1))


        ##================= Anno LM ========================##


        decoder_targets = anno.clone()

        decoder_output = self.anno_decoder(torch.zeros_like(anno), mask, seq_embed=seq_hidden_embeds)

        predict_anno = self.to_annotation_logits(decoder_output)
        update_count = 0

        anno_hat = torch.sigmoid(predict_anno).to(anno.device)
        output_pos_hat = self.anno_encoder(anno_hat.round(), seq_embed=seq_hidden_embeds)
        vl_output_true = self.itm_head(output_pos)  # [bs, 2]
        vl_output_hat = self.itm_head(output_pos_hat)  # [bs, 2]

        score_true = vl_output_true[:, 1]
        score_hat = vl_output_hat[:, 1]
        loss_mask = score_true >= score_hat

        if loss_mask.any():
            update_count = loss_mask.sum().item()  ## the number of updated samples
        if self.use_adaptive_loss:
            loss_lm_anno = focal_loss(predict_anno,
                                        decoder_targets.to(device=predict_anno.device),
                                        alpha=0.75,
                                        gamma=2,
                                        reduction='none')
            loss_lm_anno = loss_lm_anno[loss_mask]
            if loss_lm_anno.shape[0] == 0:
                loss_lm_anno = 0
            else:
                loss_lm_anno = loss_lm_anno.mean()
        else:
            loss_lm_anno = focal_loss(predict_anno,
                                        decoder_targets.to(device=predict_anno.device),
                                        alpha=0.75,
                                        gamma=2,
                                        reduction='mean')

        anno_accuracy = BinaryAccuracy(threshold=0.5)(anno_hat.to('cpu'), anno.to('cpu'))
        anno_precision = BinaryPrecision(threshold=0.5)(anno_hat.to('cpu'), anno.to('cpu'))
        anno_recall = BinaryRecall(threshold=0.5)(anno_hat.to('cpu'), anno.to('cpu'))
        anno_fmax = 0.0
        for cut in (c / 20 for c in range(21)):
            anno_fmax = max(anno_fmax, BinaryF1Score(threshold=cut)(anno_hat.to('cpu'), anno.to('cpu')))

        anno_auc = AUROC(task="binary")(anno_hat.to('cpu'), anno.to('cpu'))

        ##=================Seq LM ========================##

        protein_seq_targets = protein_seq.clone()  ## [bs, 512]

        loss_lm_seq = F.cross_entropy(seq_logits.permute(0, 2, 1),
                                        protein_seq_targets.to(device=seq_logits.device).long(), reduction='mean')
        seq_hat = seq_logits.argmax(-1)
        range_tensor = torch.arange(0, seq_hat.shape[1]).expand(seq_hat.shape[0], -1).to(seq_lens.device)

        mask = range_tensor <= seq_lens.unsqueeze(1)
        mask[:, 0] = False
        diff = seq_hat - protein_seq_targets
        cnt = torch.count_nonzero(diff * mask)
        seq_difference = cnt / seq_lens.sum()

        return loss_itm, loss_lm_anno, loss_lm_seq, [anno_accuracy, anno_precision,
                                                               anno_recall, anno_fmax,
                                                               anno_auc, seq_difference
                                                               ], itm_accuracy, update_count


def protac_pretrain_seq_anno_mlm_esm2_new(**kwargs):
    model = ProtAC_Pretrain_Seq_Anno_MLM_esm2_new(**kwargs)
    return model


def new_get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    sub_mask = torch.zeros_like(mask, device=device)
    # Process each row
    for i in range(batch):
        # Calculate the total number of 1s in each row
        num_ones = mask[i].sum().item()
        # Calculate the number of 1s to select in each row, i.e., 80% of the total number of 1s
        num_to_select = math.ceil(num_ones * prob)
        # Create a random array to select elements with value 1
        random_tensor = torch.rand(seq_len, device=device)
        # Select the elements with value 1 in the current row
        _, idx = torch.topk(random_tensor * mask[i], k=num_to_select)
        # Set the selected positions to 1 to form the sub_mask for the current row
        sub_mask[i].scatter_(dim=0, index=idx, value=1)
    return sub_mask


def exists(val):
    return val is not None
