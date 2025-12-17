#Api class for simple use in colab, maybe 41.ai
## This is half vibecoded half copy pasted from demo.ipynb
from model import load_model
import numpy as np
import torch
from text import text_to_sequence
import librosa
from layers import TacotronSTFT

class TPGSTTacotron2:
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
    
    @torch.no_grad()
    def infer_bert(self, text, emo_overwrite_text, arpabet = True, tpgst_mode = "tpse"): # The more sigma, cooler method

        arpabet = 1.0 if arpabet else 0.0
        emo_overwrite_text = text if emo_overwrite_text is None else emo_overwrite_text
        availbe_modes = ["tpse", "tpcw", "tpse-linear"]

        if tpgst_mode not in availbe_modes:
            raise ValueError(f"invalid tpgst_mode {tpgst_mode}, available modes are {availbe_modes}")


        sequence = np.array(text_to_sequence(text, ['english_cleaners'], p_arpabet=arpabet))[None, :]
        sequence = torch.from_numpy(sequence).to(device='cuda', dtype=torch.int64)
        
        #predict emotion embedding
        predicted = self.model.inference((sequence, emo_overwrite_text), tpgst_mode)
        mel_outputs, mel_outputs_postnet, gate_outputs , alignments = predicted

        return mel_outputs, mel_outputs_postnet, gate_outputs ,alignments
    
    @torch.no_grad()
    def infer_ref_audio(self, text, ref_audio, arpabet = True): #Boring method, who is going to use it?
        arpabet = 1.0 if arpabet else 0.0
        ref_mel = self.load_mel(ref_audio)
        sequence = np.array(text_to_sequence(text, ['english_cleaners'], p_arpabet=arpabet))[None, :]
        sequence = torch.from_numpy(sequence).to(device='cuda', dtype=torch.int64)

        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = self.model.inference_reference((sequence, ref_mel))
        return mel_outputs, mel_outputs_postnet, gate_outputs ,alignments
    
    def infer_no_gst(self, text, arpabet = True):
        # I am lazy, I think there are people who will hack it themselves if they really need it
        import sys
        print("infer_no_gst is not available." + "\n" + "Use infer_bert() or infer_ref_audio() for now.")
        sys.exit(0)

    def load_mel(self, path):
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



        
