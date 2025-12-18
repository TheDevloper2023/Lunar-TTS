import numpy as np
from scipy.io.wavfile import read
import torch


@torch.jit.script
def get_mask_from_lengths_alternitive(lengths: torch.Tensor, max_len:int = 0): # Used for the guided attention loss and whatever CookieTTS has that I will copy paste from
    if max_len == 0:
        max_len = int(torch.max(lengths).item())
    ids = torch.arange(0, max_len, device=lengths.device, dtype=torch.long)
    mask = (ids < lengths.unsqueeze(1))
    return mask

def get_mask_from_lengths(lengths): # Used for the original stuff in TT2
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


"""
for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            text_padded, input_lengths, mel_padded, max_len, output_lengths, raw_text = x
            y_pred = model(x)
            mel_out, mel_out_postnet, gate_out, alignments, tp_gst_output = y_pred
            # TP-GST
            tpcw_output, tpse_output, tpse_linear_output, embedded_gst, scores_gst = tp_gst_output

            loss_tpcw = criterion_tpcw(tpcw_output, scores_gst)
            loss_tpse = criterion_tpse(tpse_output, embedded_gst)
            loss_tpse_l = criterion_tpse(tpse_linear_output, embedded_gst)

            loss = criterion(y_pred, y, alignments, input_lengths, output_lengths)
            loss = loss + loss_tpcw + loss_tpse + loss_tpse_l

            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)
        """