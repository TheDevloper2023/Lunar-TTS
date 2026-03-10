import random
import torch
from tensorboardX import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy
import numpy as np

class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
                     iteration):
            self.add_scalar("training.loss", reduced_loss, iteration)
            self.add_scalar("grad.norm", grad_norm, iteration)
            self.add_scalar("learning.rate", learning_rate, iteration)
            self.add_scalar("duration", duration, iteration)

    def log_validation(self, reduced_loss, model, y, y_pred, iteration):
        self.add_scalar("validation.loss", reduced_loss, iteration)

        mel_pred, mel_pred_postnet, gate_pred, alignments, _ = y_pred
        mel_targets, gate_targets = y

        # Parameter histograms (safe even early)
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # Optional: Skip images entirely before stable (e.g., iteration 1000)
        if iteration < 1000:
            return

        idx = random.randint(0, alignments.size(0) - 1)

        # Ultimate safe plotting function — forces clean float array
        def safe_mel_plot(mel_tensor):
            mel = mel_tensor[idx].data.cpu().numpy()
            # Replace NaN/Inf
            mel = np.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0)
            # Clamp to visual range
            mel = np.clip(mel, -12.0, 4.0)
            # FORCE float64 dtype to avoid "flexible type" error
            mel = mel.astype(np.float64)
            return plot_spectrogram_to_numpy(mel)

        # Alignment (usually clean)
        align_np = alignments[idx].data.cpu().numpy().T
        align_np = np.nan_to_num(align_np).astype(np.float64)

        self.add_image(
            "alignment",
            plot_alignment_to_numpy(align_np),
            iteration, dataformats='HWC')

        self.add_image(
            "mel_target",
            safe_mel_plot(mel_targets),
            iteration, dataformats='HWC')

        self.add_image(
            "mel_predicted",
            safe_mel_plot(mel_pred_postnet),
            iteration, dataformats='HWC')

        # Gate
        gate_target_np = gate_targets[idx].data.cpu().numpy()
        gate_pred_sig = torch.sigmoid(gate_pred[idx]).data.cpu().numpy()
        gate_pred_sig = np.nan_to_num(gate_pred_sig).astype(np.float64)

        self.add_image(
            "gate",
            plot_gate_outputs_to_numpy(gate_target_np, gate_pred_sig),
            iteration, dataformats='HWC')
