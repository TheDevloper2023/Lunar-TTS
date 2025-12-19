import numpy as np
from scipy.io.wavfile import read
import torch


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)

def get_alignment_metrics( #From Uberduck
    alignments, average_across_batch=True, input_lengths=None, output_lengths=None
):
    alignments = alignments.transpose(1, 2)  # [B, dec, enc] -> [B, enc, dec]
    if input_lengths == None:
        input_lengths = torch.ones(alignments.size(0), device=alignments.device) * (
            alignments.shape[1] - 1
        )  # [B] # 147
    if output_lengths == None:
        output_lengths = torch.ones(alignments.size(0), device=alignments.device) * (
            alignments.shape[2] - 1
        )  # [B] # 767

    batch_size = alignments.size(0)
    optimums = torch.sqrt(
        input_lengths.double().pow(2) + output_lengths.double().pow(2)
    ).view(batch_size)

    # [B, enc, dec] -> [B, dec], [B, dec]
    values, cur_idxs = torch.max(alignments, 1)

    cur_idxs = cur_idxs.float()
    prev_indx = torch.cat((cur_idxs[:, 0][:, None], cur_idxs[:, :-1]), dim=1)
    dist = ((prev_indx - cur_idxs).pow(2) + 1).pow(0.5)  # [B, dec]
    dist.masked_fill_(
        ~get_mask_from_lengths(output_lengths, max_len=dist.size(1)), 0.0
    )  # set dist of padded to zero
    dist = dist.sum(dim=(1))  # get total dist for each B
    diagonalness = (dist + 1.4142135) / optimums  # dist / optimal dist

    maxes = alignments.max(axis=1)[0].mean(axis=1)
    if average_across_batch:
        diagonalness = diagonalness.mean()
        maxes = maxes.mean()

    output = {}
    output["diagonalness"] = diagonalness
    output["max"] = maxes

    return output