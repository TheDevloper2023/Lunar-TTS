#Api class for simple use in colab.
## Most of this is copied from Lightbooster's colab 

### Thine rewrote tis because it was unprofesional for my big sis
from model import load_model
import numpy as np
import torch
from text import text_to_sequence
import librosa
from layers import TacotronSTFT

class LunarTTS:
    def __init__(self, model_path, hparams, device='cuda'):
        self.device = device
        self.model = load_model(hparams=hparams)
        self.model.load_state_dict(torch.load(model_path)['state_dict'], strict=False)
        self.model = self.model.to(self.device).eval()  # move model to device and set eval mode
        self.device = device

        self.stft = TacotronSTFT(
            hparams.filter_length,
            hparams.hop_length,
            hparams.win_length,
            hparams.n_mel_channels,
            hparams.sampling_rate,
            hparams.mel_fmin,
            hparams.mel_fmax
        )


        self.hparams = hparams

    def _load_mel(self, path):
            audio, sampling_rate = librosa.core.load(path, sr=self.stft.sampling_rate)
            audio = torch.from_numpy(audio)
            if sampling_rate != self.hparams.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = melspec.to(self.device)
            return melspec
    


    def __call__(self, text, emotion=None, arpabet = None ,ref_mode=0 ,reference_audio=None, tpgst_mode="tpse"):
            arpabet = 1.0 if arpabet else 0.0
            emotion = text if emotion is None or emotion == "" else emotion

            if ref_mode not in [0,1,2]:
                raise ValueError(f"invalid infer_mode {ref_mode}, must be either 0 - No style ; 1 - TPGST-BERT ; 2 - Reference Audio")

            tpgst_mode = tpgst_mode.lower()
            if tpgst_mode not in ["tpse", "tpcw", "tpse-linear"]:
                  raise ValueError(f"invalid tpgst_mode {tpgst_mode}, must pe TPSE, TPSE-Linear, TPCW")

            sequence = np.array(text_to_sequence(text, ['english_cleaners'], p_arpabet=arpabet))[None, :]
            sequence = torch.from_numpy(sequence).to(device='cuda', dtype=torch.int64)

            if ref_mode == 1: # Ref audio
                  if reference_audio is None:
                        raise ValueError("Reference Audio must be included with ref_mode = 1!")
                  ref_mel = self._load_mel(reference_audio)
                  mel_outputs, mel_outputs_postnet, gate_outputs, alignments = self.model.inference_reference((sequence, ref_mel))
            
            elif ref_mode == 2: # TPGST 
                  mel_outputs, mel_outputs_postnet, gate_outputs, alignments = self.model.inference((sequence, emotion), tpgst_mode)
            
            elif ref_mode == 0: # No Style
                  zeros = torch.zeros(sequence.size(0), self.hparams.token_embedding_size).to(self.device)
                  mel_outputs, mel_outputs_postnet, gate_outputs, alignments = self.model.inference_reference((sequence, zeros))

            return {
                  "mel_outputs": mel_outputs,
                  "mel_outputs_postnet": mel_outputs_postnet,
                  "gate_outputs": gate_outputs,
                  "alignments": alignments,
            }
    



                  
                  



        
