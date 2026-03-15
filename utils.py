import numpy as np
from scipy.io.wavfile import read
import torch
import os

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



# Dodo? Thy stole code from CookiePPP... again?! #Luna

# Yes moon pricness, what do you want me to do? Implement functions that hadn't have a proper implementaion in almost a decade? #Dodo

# That is exactly what we want thou to do. #Luna

# Please just buck yourself? #Dodo

# With Nightmare Moon?  She is far more daunting than the tasks thou hast been neglecting this week.

# At least I write something, you have been just adding comments in train.py, isn't this your repo blank-flank? #Dodo

# If thine continue insult like that, we shall summon big sis to forget you over #Luna

# Can you just at the very least add this into train.py #Dodo

# No dodo, thou hast dumb as thine namesake. We have business with protecting dreams.

# # Dodo: What about I dream railing you… in shadows, if you catch my drift? #Dodo



## After that, Dodo was never heard from again. The Uberduck Discord server believes he is at school, 
## but I, as Princess Celestia and as his big sister know the truth. 
## He now serves Nightmare Moon for all eternity.




def alignment_metric(alignments, input_lengths=None, output_lengths=None, enc_min_thresh=0.7, average_across_batch=False): # Credit to CookiePPP/CookieTTS, please tell me your license allows me to just plop in your code like this
    alignments = alignments.transpose(1,2) # [B, dec, enc] -> [B, enc, dec]
    # alignments [batch size, x, y]
    # input_lengths [batch size] for len_x
    # output_lengths [batch size] for len_y
    if input_lengths == None:
        input_lengths =  torch.ones(alignments.size(0), device=alignments.device)*(alignments.shape[1]-1) # [B] # 147
    if output_lengths == None:
        output_lengths = torch.ones(alignments.size(0), device=alignments.device)*(alignments.shape[2]-1) # [B] # 767
    batch_size = alignments.size(0)
    optimums = torch.sqrt(input_lengths.double().pow(2) + output_lengths.double().pow(2)).view(batch_size)
    
    # [B, enc, dec] -> [B, dec], [B, dec]
    values, cur_idxs = torch.max(alignments, 1) # get max value in column and location of max value
    
    cur_idxs = cur_idxs.float()
    prev_indx = torch.cat((cur_idxs[:,0][:,None], cur_idxs[:,:-1]), dim=1) # shift entire tensor by one.
    dist = ((prev_indx - cur_idxs).pow(2) + 1).pow(0.5) # [B, dec]
    dist.masked_fill_(~get_mask_from_lengths(output_lengths, max_len=dist.size(1)), 0.0) # set dist of padded to zero
    dist = dist.sum(dim=(1)) # get total dist for each B
    diagonalitys = (dist + 1.4142135)/optimums # dist / optimal dist
    
    alignments.masked_fill_(~get_mask_from_lengths(output_lengths, max_len=alignments.size(2))[:,None,:], 0.0)
    attm_enc_total = torch.sum(alignments, dim=2)# [B, enc, dec] -> [B, enc]
    
    # calc max  encoder durations (with padding ignored)
    attm_enc_total.masked_fill_(~get_mask_from_lengths(input_lengths, max_len=attm_enc_total.size(1)), 0.0)
    encoder_max_focus = attm_enc_total.max(dim=1)[0] # [B, enc] -> [B]
    
    # calc mean encoder durations (with padding ignored)
    encoder_avg_focus = attm_enc_total.mean(dim=1)   # [B, enc] -> [B]
    encoder_avg_focus *= (attm_enc_total.size(1)/input_lengths.float())
    
    # calc min encoder durations (with padding ignored)
    attm_enc_total.masked_fill_(~get_mask_from_lengths(input_lengths, max_len=attm_enc_total.size(1)), 1.0)
    encoder_min_focus = attm_enc_total.min(dim=1)[0] # [B, enc] -> [B]
    
    # calc average max attention (with padding ignored)
    values.masked_fill_(~get_mask_from_lengths(output_lengths, max_len=values.size(1)), 0.0) # because padding
    avg_prob = values.mean(dim=1)
    avg_prob *= (alignments.size(2)/output_lengths.float()) # because padding
    
    # calc portion of encoder durations under min threshold
    attm_enc_total.masked_fill_(~get_mask_from_lengths(input_lengths, max_len=attm_enc_total.size(1)), float(1e3))
    p_missing_enc = (torch.sum(attm_enc_total < enc_min_thresh, dim=1)) / input_lengths.float()
    
    if average_across_batch:
        diagonalitys      = diagonalitys     .mean()
        encoder_max_focus = encoder_max_focus.mean()
        encoder_min_focus = encoder_min_focus.mean()
        encoder_avg_focus = encoder_avg_focus.mean()
        avg_prob          = avg_prob         .mean()
        p_missing_enc     = p_missing_enc    .mean()
    
    output = {}
    output["diagonalitys"     ] = diagonalitys
    output["avg_prob"         ] = avg_prob
    output["encoder_max_focus"] = encoder_max_focus
    output["encoder_min_focus"] = encoder_min_focus
    output["encoder_avg_focus"] = encoder_avg_focus
    output["p_missing_enc"]     = p_missing_enc
    return output
