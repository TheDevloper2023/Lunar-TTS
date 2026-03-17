import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np


# For audio thingy
import torch, numpy as np
import torchaudio

def save_figure_to_numpy(fig):
    "Save figure to a numpy array."""
    data = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return data


def plot_alignment_to_numpy(alignment, info=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect='auto', origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_gate_outputs_to_numpy(gate_targets, gate_outputs):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.scatter(range(len(gate_targets)), gate_targets, alpha=0.5,
               color='green', marker='+', s=1, label='target')
    ax.scatter(range(len(gate_outputs)), gate_outputs, alpha=0.5,
               color='red', marker='.', s=1, label='predicted')

    plt.xlabel("Frames (Green target, Red predicted)")
    plt.ylabel("Gate State")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


# This is griflim, I kinda vibecoded it but it works at the very least
def mel_to_audio(mel_spec, hparams):
    with torch.no_grad():
        n_stft = hparams.filter_length // 2 + 1
        
        mel_to_linear = torchaudio.transforms.InverseMelScale(
            n_stft=n_stft,
            n_mels=hparams.n_mel_channels,
            sample_rate=hparams.sampling_rate,
            f_min=hparams.mel_fmin,
            f_max=hparams.mel_fmax
        ).to(mel_spec.device)

        
        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=hparams.filter_length,
            win_length=hparams.win_length,
            hop_length=hparams.hop_length,
            n_iter=45 # I guess it is good enough
        ).to(mel_spec.device)

       
        magnitude = mel_to_linear(torch.exp(mel_spec))


        audio = griffin_lim(magnitude)
        
        return audio.squeeze().cpu()
