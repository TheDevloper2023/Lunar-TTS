import os
import time
import argparse
import math

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from model import load_model
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss, TPCWLoss, TPSELoss
from logger import Tacotron2Logger
from hparams import create_hparams
from utils import alignment_metric 
from torch.amp import autocast, GradScaler


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams, speaker_ids=trainset.speaker_ids)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(trainset, num_workers=hparams.num_workers, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=hparams.pin_worker,
                              drop_last=True, collate_fn=collate_fn, persistent_workers=True)
    return train_loader, valset, collate_fn, train_sampler


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def warm_start_model(checkpoint_path, model, ignore_layers, freeze_layers, unfreeze_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict, strict=False)



    if len(freeze_layers) > 0:
        for layer, param in list(model.named_parameters()):
            if any(layer.startswith(module) for module in freeze_layers):
                param.requires_grad = False
                print(f"Froze layer {layer}")

    if len(unfreeze_layers) > 0:
        for layer, param in list(model.named_parameters()):
            if any(layer.startswith(module) for module in unfreeze_layers):
                param.requires_grad = True
                print(f"Unfroze layer {layer}")
    return model

def load_checkpoint(checkpoint_path, model, optimizer, loading_bert=False):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if not loading_bert:
        bert_keys = list()
        for key in checkpoint_dict['state_dict'].keys():
            parent_net = key[: key.find('.')]
            if parent_net == 'bert':
                bert_keys.append(key)

        for key in bert_keys:
            del checkpoint_dict['state_dict'][key]

    model.load_state_dict(checkpoint_dict['state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))

    if len(hparams.frozen_layers) > 0:
        for layer, param in list(model.named_parameters()):
            if any(layer.startswith(module) for module in hparams.frozen_layers):
                param.requires_grad = False
                print(f"Froze layer {layer}")

    if len(hparams.unfrozen_layers) > 0:
        for layer, param in list(model.named_parameters()):
            if any(layer.startswith(module) for module in hparams.unfrozen_layers):
                param.requires_grad = True
                print(f"Unfroze layer {layer}")

    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath, saving_bert=False):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))

    model_state_dict = model.state_dict().copy()

    if not saving_bert:
        bert_keys = list()
        for key in model.state_dict().keys():
            parent_net = key[: key.find('.')]
            if parent_net == 'bert':
                bert_keys.append(key)

        for key in bert_keys:
            del model_state_dict[key]

    torch.save({'iteration': iteration,
                'state_dict': model_state_dict,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def validate(model, criterions, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=hparams.val_num_workers,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=hparams.val_pin_worker, collate_fn=collate_fn, persistent_workers=True)

        criterion, criterion_tpcw, criterion_tpse = criterions
        val_loss = 0.0
        taco_val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            text_padded, input_lengths, mel_padded, max_len, output_lengths ,speaker_ids,raw_text, *_ = x
            y_pred = model(x)
            mel_out, mel_out_postnet, gate_out, alignments, tp_gst_output, *_ = y_pred
            # TP-GST
            tpcw_output, tpse_output, tpse_linear_output, embedded_gst, scores_gst = tp_gst_output

            loss_tpcw = criterion_tpcw(tpcw_output, scores_gst) / 100
            loss_tpse = criterion_tpse(tpse_output, embedded_gst)
            loss_tpse_l = criterion_tpse(tpse_linear_output, embedded_gst)

            tacotron_outputs = (mel_out, mel_out_postnet, gate_out, alignments)

            loss = criterion(tacotron_outputs, y, input_lengths, output_lengths)
            taco_loss = loss
            loss = loss + loss_tpcw + loss_tpse + loss_tpse_l

            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
                reduced_val_loss_taco = reduce_tensor(taco_loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
                reduced_val_loss_taco = taco_loss.item()
            val_loss += reduced_val_loss
            taco_val_loss += reduced_val_loss_taco
        val_loss = val_loss / (i + 1)
        taco_val_loss /= (i + 1)

    model.train()
    if rank == 0:
        print("Validation loss {}: {:9f}  ".format(iteration, reduced_val_loss))
        logger.log_validation(val_loss, model, y, y_pred, iteration)
    
    
    atd = alignment_metric(alignments=alignments, input_lengths=x[1], output_lengths=x[4], enc_min_thresh=0.7, average_across_batch=True)

    diagonality_batch   = atd['diagonalitys']
    avg_prob_batch      = atd['avg_prob']
    enc_max_dur_batch   = atd['encoder_max_focus']
    enc_min_dur_batch   = atd['encoder_min_focus']
    enc_avg_dur_batch   = atd['encoder_avg_focus']
    p_missing_enc_batch = atd['p_missing_enc']

    # Use avg_prob_batch directly
    weighted_score = avg_prob_batch.item()  # scalar because average_across_batch=True

    # Apply penalties
    diagonality_punishment = (max(diagonality_batch.item(), 1.10) - 1.10) * 0.5 * 0.5
    max_dur_punishment      = max(enc_max_dur_batch.item()-60, 0) * 0.005
    min_dur_punishment      = max(0.0 - enc_min_dur_batch.item(),0) * 0.5
    avg_dur_punishment      = max(3.6 - enc_avg_dur_batch.item(), 0)
    mis_dur_punishment      = max(p_missing_enc_batch.item() - 0.08, 0)

    weighted_score -= (diagonality_punishment + max_dur_punishment + min_dur_punishment + avg_dur_punishment + mis_dur_punishment)

    style_loss = loss_tpcw + loss_tpse + loss_tpse_l

    return val_loss, weighted_score, style_loss, taco_val_loss


def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """

    if hparams.fp16_run and torch.cuda.is_available():
        dtype = torch.float16
    elif hparams.bf16_run and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32


    if hparams.fp16_run and hparams.bf16_run:
        assert False, "Are you high or something? You can't set both fp16_run and bf16_run to True. Pick one! Check your hparams."


    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay, fused=True)

    

    scaler = GradScaler('cuda', enabled=hparams.fp16_run, init_scale=2**16, growth_interval=100) if hparams.fp16_run else None


    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    criterion = Tacotron2Loss(hparams=hparams)
    criterion_tpse = TPSELoss()
    criterion_tpcw = TPCWLoss()

    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank)

    train_loader, valset, collate_fn, train_sampler = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers, hparams.frozen_layers, hparams.unfrozen_layers)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer, hparams.bert_load_from_checkpoint)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))
    
    best_val_loss = 1e3
    best_attsc_loss = 9e9
    best_gst_loss = 1e3
    best_tv_loss = 1e3

    model.train()
    is_overflow = False
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            if iteration > 0 and iteration % hparams.learning_rate_anneal == 0:
                learning_rate = max(
                    hparams.learning_rate_min, learning_rate * 0.5)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate

            optimizer.zero_grad(set_to_none=True)
            x, y = model.parse_batch(batch)
            text_padded, input_lengths, mel_padded, max_len, output_lengths, speaker_ids ,raw_text, *_ = x

            with autocast('cuda', enabled=hparams.fp16_run or hparams.bf16_run, dtype=dtype):
                y_pred = model(x)
                mel_out, mel_out_postnet, gate_out, alignments, tp_gst_output, *_ = y_pred

                # Attention metrics

                atd = alignment_metric(alignments=alignments, input_lengths=x[1], output_lengths=x[4], enc_min_thresh=0.7, average_across_batch=True)
                diagonality_batch   = atd['diagonalitys']
                avg_prob_batch      = atd['avg_prob']
                enc_max_dur_batch   = atd['encoder_max_focus']
                enc_min_dur_batch   = atd['encoder_min_focus']
                enc_avg_dur_batch   = atd['encoder_avg_focus']
                p_missing_enc_batch = atd['p_missing_enc']

                # Use avg_prob_batch directly
                weighted_score = avg_prob_batch.item()  # scalar because average_across_batch=True

                # Apply penalties
                diagonality_punishment = (max(diagonality_batch.item(), 1.10) - 1.10) * 0.5 * 0.5
                max_dur_punishment      = max(enc_max_dur_batch.item()-60, 0) * 0.005
                min_dur_punishment      = max(0.0 - enc_min_dur_batch.item(),0) * 0.5
                avg_dur_punishment      = max(3.6 - enc_avg_dur_batch.item(), 0)
                mis_dur_punishment      = max(p_missing_enc_batch.item() - 0.08, 0)

                weighted_score -= (diagonality_punishment + max_dur_punishment + min_dur_punishment + avg_dur_punishment + mis_dur_punishment)

                # TP-GST
                tpcw_output, tpse_output, tpse_linear_output, embedded_gst, scores_gst = tp_gst_output

                loss_tpcw = criterion_tpcw(tpcw_output, scores_gst) / 100
                loss_tpse = criterion_tpse(tpse_output, embedded_gst)
                loss_tpse_l = criterion_tpse(tpse_linear_output, embedded_gst)
                tacotron_outputs = (mel_out, mel_out_postnet, gate_out, alignments)
                loss = criterion(tacotron_outputs, y, input_lengths, output_lengths)
                taco_loss = loss
                loss = loss + loss_tpcw + loss_tpse + loss_tpse_l

            

            # Luna's eternal wisdom for the next fool who touches this code:
            # BF16 doesn't require a GradScaler (unlike fp16) because bfloat16 has basically fp32 dynamic range.
            # That's why fp32 and bf16 share the exact same .backward() → .step() path.
            #
            # If thou darest write:
            """
            if hparams.fp16_run or hparams.bf16_run:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            """
            #
            # Then prepare to face the wrath of Luna.
            # Thou hast committed a grave sin against the sacred training loop,
            # shattering the delicate balance of mixed precision,
            # unleashing chaos and destruction upon the realm of deep learning.
            # Beware — the training loop is holy ground.
            # Only the purest of intentions may tread here.


            # Uhh since when does she even write code? didn't she just like, write poetry or something? I guess she writes code now. Cool. Just don't let her near the training loop, that's all I'm saying.
            # TL:DR BF16 doesn't require a GradScaler, it is closer to FP32 than FP16 in dynamic range.
           
           
            if hparams.distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()

            if hparams.fp16_run:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            
            else:
                loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), hparams.grad_clip_thresh)

            is_overflow = math.isnan(grad_norm) or math.isinf(grad_norm)

            if is_overflow and rank==0:
                print(f"Gradient overflow detected at iteration {iteration} — skipping step")
                if hparams.fp16_run:
                    print(f"scaler factor = {scaler.get_scale():.0f}")
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)
           
                continue                 
            
            if hparams.fp16_run:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if not is_overflow and rank == 0:
                duration = time.perf_counter() - start
                #print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                #    iteration, reduced_loss, grad_norm, duration))
                
                print("\n" * 2)
                print(f"---->  / Step: {i} / {len(train_loader)} | Epoch: {epoch} | Global Step: {iteration}")
                print(f"Total Loss: {reduced_loss:.6f} | Tacotron2 Loss: {taco_loss.item():.6f} | TPSE Loss: {loss_tpse.item():.6f} | TPCW Loss: {loss_tpcw.item():.6f} | TPSE Linear Loss: {loss_tpse_l.item():.6f}")
                print(f"Attention Score: {weighted_score:.6f} | Max Attn (Avg_prob): {avg_prob_batch.mean().item():.6f} | Diagonalness: {diagonality_batch.mean().item():.6f} | Max Focus: {enc_max_dur_batch.item():.6f} | Min Focus: {enc_min_dur_batch.item():.6f}")
                print(f"Grad Norm: {grad_norm:.6f}")
                print(f"Learning Rate: {learning_rate:.6f}")
                print(f"Duration: {duration:.2f}s/it")
                if hparams.fp16_run:
                    print(f"Scaler factor: {scaler.get_scale():.0f}")
                print("-" * 15 + ">")    

                
                logger.log_training(
                    reduced_loss, grad_norm, learning_rate, duration, iteration)

            if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
                val_loss, att_score, style_loss, taco_val_loss = validate(model, (criterion, criterion_tpcw, criterion_tpse), valset, iteration,
                        hparams.val_batch_size, n_gpus, collate_fn, logger,
                        hparams.distributed_run, rank)
                if rank == 0:
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration))
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path, hparams.bert_save_in_checkpoint)
                    
                    if val_loss < best_val_loss and hparams.save_best_validation:
                        best_val_loss = val_loss
                        checkpoint_path = os.path.join(
                        output_directory, "best_val_style__taco")
                        save_checkpoint(model, optimizer, learning_rate, iteration,
                                        checkpoint_path, hparams.bert_save_in_checkpoint)
                    if att_score < best_attsc_loss and hparams.save_best_attsc:
                        best_attsc_loss = att_score
                        checkpoint_path = os.path.join(
                        output_directory, "best_inf_attsc")
                        save_checkpoint(model, optimizer, learning_rate, iteration,
                                        checkpoint_path, hparams.bert_save_in_checkpoint)
                    
                    if style_loss < best_gst_loss and hparams.save_best_gst:
                        best_gst_loss = style_loss
                        checkpoint_path = os.path.join(
                        output_directory, "best__val_style_model")
                        save_checkpoint(model, optimizer, learning_rate, iteration,
                                        checkpoint_path, hparams.bert_save_in_checkpoint)
                    if best_tv_loss > taco_val_loss and hparams.best_best_taco:
                        best_tv_loss = taco_val_loss
                        checkpoint_path = os.path.join(
                        output_directory, "best_val_taco_model")
                        save_checkpoint(model, optimizer, learning_rate, iteration,
                                        checkpoint_path, hparams.bert_save_in_checkpoint)
                        
                        

            iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--warm_start_force', action='store_true',
                        help='load model weights only')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("BF16 Run:", hparams.bf16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    train(args.output_directory, args.log_directory, args.checkpoint_path,
          args.warm_start, args.n_gpus, args.rank, args.group_name,hparams)
