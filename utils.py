import numpy as np
from scipy.io.wavfile import read
import torch
import os


@torch.jit.script
def get_mask_from_lengths(lengths: torch.Tensor, max_len:int = 0): # From CookieTTS, this one's better and more future proof
    if max_len == 0:
        max_len = int(torch.max(lengths).item())
    ids = torch.arange(0, max_len, device=lengths.device, dtype=torch.long)
    mask = (ids < lengths.unsqueeze(1))
    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename: str, split: str = "|", realtive: bool = True):
    """
    Load a Tacotron2-style filelist and optionally convert relative paths to absolute paths.

    Args:
        filename (str): path to the filelist
        split (str): delimiter used in the filelist (default "|")
        dataset_root (str, optional): if provided, converts file paths to absolute paths

    Returns:
        List of [filepath, transcript, speaker_id]
    """
    filepaths_and_text = []

    with open(filename, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = line.split(split)
            # Ensure there is a speaker_id
            if len(data) < 3:
                data.append("0")

            # Convert relative path to absolute
            if realtive:
                dataset_root = os.path.dirname(os.path.abspath(filename))
                data[0] = os.path.abspath(os.path.join(dataset_root, data[0]))

            filepaths_and_text.append(data)

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



class HParams(object):
    hparamdict = []
    def __init__(self, **hparams):
        self.hparamdict = hparams
        for k, v in hparams.items():
            setattr(self, k, v)
    def __repr__(self):
        return "HParams(" + repr([(k, v) for k, v in self.hparamdict.items()]) + ")"
    def __str__(self):
        return ','.join([(k + '=' + str(v)) for k, v in self.hparamdict.items()])
    def parse(self, params):
        for s in params.split(","):
            k, v = s.split("=", 1)
            k = k.strip()
            t = type(self.hparamdict[k])
            if t == bool:
                v = v.strip().lower()
                if v in ['true', '1']:
                    v = True
                elif v in ['false', '0']:
                    v = False
                else:
                    raise ValueError(v)
            else:
                v = t(v)
            self.hparamdict[k] = v
            setattr(self, k, v)
        return self
