import numpy as np
import random
import torch
from speechbrain.utils.data_utils import batch_pad_right
import logging

logger = logging.getLogger(__name__)

def compute_mask(shape, sample_lens, mask_prob, mask_length):
    min_sample_len = min(sample_lens)

    # int always floors the float number so adding + random.random() 
    # makes it 50% change of rounding up and 50% of rounding down
    num_mask = int(mask_prob * min_sample_len + random.random())

    # make sure there is at least 1 mask 
    if num_mask == 0:
        num_mask = 1

    permutation = torch.randperm(min_sample_len//mask_length) * mask_length
    selected_indices = permutation[:num_mask]
    selected_indices, _ = selected_indices.sort()

    # get next for frames
    idx = []
    for  i in selected_indices:
        idx.append(torch.arange(start=i, end=i+mask_length))
    idx = torch.cat(idx)

    return idx


def brq_mask_collate_fn(samples_lst, get_out_len_fn, mask_prob, mask_length, n_mels):
    wav_lst, latent_length_lst = [], []
    ids = []
    for sample in samples_lst:
        ids.append(sample["id"])
        sig = sample["sig"]
        wav_lst.append(sig)
        latent_length = get_out_len_fn(torch.as_tensor(sig.size(-1)))
        latent_length_lst.append(latent_length.item())
    bs = len(wav_lst)
    wavs_padded, wav_lens = batch_pad_right(wav_lst)

    batch_time_len = max(latent_length_lst)
    batch_time_len
    mask = compute_mask(
        (bs, batch_time_len, n_mels), latent_length_lst, mask_prob, mask_length
    )
    return (
        torch.as_tensor(wavs_padded),
        torch.as_tensor(wav_lens),
        torch.as_tensor(mask),
    )


def brq_with_lang_mask_collate_fn(samples_lst, get_out_len_fn, mask_prob, mask_length, n_mels):
    wav_lst, latent_length_lst = [], []
    ids, languages = [], []
    for sample in samples_lst:
        ids.append(sample["id"])
        sig = sample["sig"]
        wav_lst.append(sig)
        # print("sig size", sig.size(-1))
        latent_length = get_out_len_fn(torch.as_tensor(sig.size(-1)))
        # print("latent length", latent_length)
        latent_length_lst.append(latent_length.item())
        languages.append(sample["lang"])

    bs = len(wav_lst)
    # print("bsz", bs)
    wavs_padded, wav_lens = batch_pad_right(wav_lst)

    batch_time_len = max(latent_length_lst)
    mask = compute_mask(
        (bs, batch_time_len, n_mels), latent_length_lst, mask_prob, mask_length
    )

    return (
        torch.as_tensor(wavs_padded),
        torch.as_tensor(wav_lens),
        torch.as_tensor(mask),
        torch.as_tensor(languages)
    )

def brq_with_label_mask_collate_fn(samples_lst, get_out_len_fn, mask_prob, mask_length, n_mels):
    wav_lst, latent_length_lst = [], []
    ids, labels = [], []
    for sample in samples_lst:
        ids.append(sample["id"])
        sig = sample["sig"]
        wav_lst.append(sig)
        # print("sig size", sig.size(-1))
        latent_length = get_out_len_fn(torch.as_tensor(sig.size(-1)))
        # print("latent length", latent_length)
        latent_length_lst.append(latent_length.item())
        labels.append(sample["label"])

    bs = len(wav_lst)
    # print("bsz", bs)
    wavs_padded, wav_lens = batch_pad_right(wav_lst)

    batch_time_len = max(latent_length_lst)
    mask = compute_mask(
        (bs, batch_time_len, n_mels), latent_length_lst, mask_prob, mask_length
    )
    #logger.warning("clusters", clusters)

    return (
        torch.as_tensor(wavs_padded),
        torch.as_tensor(wav_lens),
        torch.as_tensor(mask),
        torch.as_tensor(labels)
    )
