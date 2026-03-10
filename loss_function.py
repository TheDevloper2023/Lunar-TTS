import torch
from torch import nn
from utils import get_mask_from_lengths

class Tacotron2Loss(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2Loss, self).__init__()

        self.hparams = hparams
        self.sigma = hparams.guided_attention_sigma or 0.2
        self.alpha = hparams.guided_attention_weight or 1.0

        self.guided_attn = None
        if hparams.use_guided_attention:
            self.guided_attn = GuidedAttentionLoss(sigma=self.sigma, alpha=self.alpha)
    def forward(self, model_output, targets, ilens, olens):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, attention = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        total_loss = mel_loss + gate_loss
        if self.guided_attn is not None:
            att_loss = self.guided_attn(attention, ilens, olens)
            total_loss += att_loss
        return total_loss

class GuidedAttentionLoss(torch.nn.Module):
    """Guided attention loss function module.
    This module calculates the guided attention loss described in `Efficiently Trainable Text-to-Speech System Based
    on Deep Convolutional Networks with Guided Attention`_, which forces the attention to be diagonal.
    .. _`Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention`:
        https://arxiv.org/abs/1710.08969
    """
    
    # Guided attention loss comes from https://github.com/gothiswaysir/Transformer_Multi_encoder/blob/952868b01d5e077657a036ced04933ce53dcbf4c/nets/pytorch_backend/e2e_tts_tacotron2.py#L28-L156
    # Cookie did modifications to make it work with nvidia's tacotron2 (mellotron) implementation

    # So uhh thanks Cookie and gothiswaysir I guess
    def __init__(self, sigma=0.4, alpha=1.0, reset_always=True):
        """Initialize guided attention loss module.
        Args:
            sigma (float, optional): Standard deviation to control how close attention to a diagonal.
            alpha (float, optional): Scaling coefficient (lambda).
            reset_always (bool, optional): Whether to always reset masks.
        """
        super(GuidedAttentionLoss, self).__init__()
        self.sigma = sigma
        self.alpha = alpha
        self.reset_always = reset_always
        self.guided_attn_masks = None
        self.masks = None

    def _reset_masks(self):
        self.guided_attn_masks = None
        self.masks = None
    
    def forward(self, att_ws, ilens, olens):
        """Calculate forward propagation.
        Args:
            att_ws (Tensor): Batch of attention weights (B, T_max_out, T_max_in).
            ilens (LongTensor): Batch of input lenghts (B,).
            olens (LongTensor): Batch of output lenghts (B,).
        Returns:
            Tensor: Guided attention loss value.
        """
        if self.guided_attn_masks is None:
            self.guided_attn_masks = self._make_guided_attention_masks(ilens, olens).to(att_ws.device)
        if self.masks is None:
            self.masks = self._make_masks(ilens, olens).to(att_ws.device)
        losses = self.guided_attn_masks * att_ws
        loss = torch.mean(losses.masked_select(self.masks))
        if self.reset_always:
            self._reset_masks()
        return self.alpha * loss

    def _make_guided_attention_masks(self, ilens, olens):
        n_batches = ilens.shape[0]
        max_ilen = int(ilens.max().item())
        max_olen = int(olens.max().item())
        device = ilens.device  # Get device from input lengths
        guided_attn_masks = torch.zeros((n_batches, max_olen, max_ilen), device=device)
        for idx, (ilen, olen) in enumerate(zip(ilens, olens)):
            guided_attn_masks[idx, :olen, :ilen] = self._make_guided_attention_mask(
                ilen.item(), olen.item(), self.sigma, device
            )
        return guided_attn_masks

    @staticmethod
    def _make_guided_attention_mask(ilen, olen, sigma, device):  # Add device argument
        grid_x, grid_y = torch.meshgrid(
            torch.arange(olen, device=device),
            torch.arange(ilen, device=device),
            indexing='ij'
        )
        grid_x = grid_x.float()
        grid_y = grid_y.float()
        return 1.0 - torch.exp(-(grid_y / ilen - grid_x / olen) ** 2 / (2 * (sigma ** 2)))

    @staticmethod
    def _make_masks(ilens, olens):
        in_masks = get_mask_from_lengths(ilens)  # (B, T_in)
        out_masks = get_mask_from_lengths(olens)  # (B, T_out)
        return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)  # (B, T_out, T_in)

class TPCWLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def cross_entropy(w_combination, target):
        return -(target * torch.log(w_combination)).sum(dim=1).mean()

    def forward(self, w_combination, target):
        """
        calculates cross-entropy loss over soft classes (GSTs distributions) and predicted weights
        :param w_combination: predicted combination weights tensor shape of (batch_size, token_num)
                                                                         or (batch_size, atn_head_num, token_num)
        :param target: GSTs' combination weights tensor shape of (batch_size, token_num)
                                                              or (batch_size, atn_head_num, token_num)
        :return: cross-entropy loss value or sum of cross-entropy loss values
        """
        if w_combination.dim() == 2:
            return self.cross_entropy(w_combination, target)
        else:
            losses = []
            for atn_head_index in range(w_combination.size(1)):
                loss = self.cross_entropy(w_combination[:, atn_head_index, :], target[:, atn_head_index, :])
                losses.append(loss)
            return sum(losses)


class TPSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, predicted_tokens, target):
        """
        calculate L1 loss function between predicted and target GST
        :param predicted_tokens: tensor shape of (batch_size, token_dim)
        :param target: tensor shape of (batch_size, token_dim)
        :return: L1 loss
        """
        return self.l1(predicted_tokens, target)
