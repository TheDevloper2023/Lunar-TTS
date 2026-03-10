from text.symbols import symbols
#import tensorflow as tf

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


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = HParams(
        #==============================#
        # Experiment Parameters        #
        #==============================#
        epochs=500,
        iters_per_checkpoint=250,
        seed=1234,
        cudnn_enabled=True,
        cudnn_benchmark=False,


        ignore_layers=['embedding.weight'], # for `warm_start`-ing
        frozen_layers=["Add-Layers"], # only the module names are required e.g: "encoder." will freeze all parameters INSIDE the encoder recursively
        unfrozen_layers=["Add-Layers"], #TODO # modules that are unfrozen
    

        dynamic_loss_scaling=False,
        fp16_run=False, # requires 20 Series or Better
        bf16_run=True, # requires Amper GPUS or better # So no Tesla T4


        
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",


        #==============================#
        # Checkpoint Saving            #
        #==============================#

        save_best_validation = True,  # Save the model with the lowest combined validation loss (Tacotron2 + GST losses)
        save_best_attsc = True,      # Save the model with the highest attention score (best alignment quality)
        save_best_gst = True,      # Save the model with the lowest GST/style loss (TPSE + TPSE Linear + TPCW)
        best_best_taco = True,      # Save the model with the lowest Tacotron2 validation loss (reconstruction loss only)



        #==============================#
        # Data Parameters             #
        #==============================#
        training_files='filelists/ljs_audiopaths_text_sid_train_filelist.txt',
        validation_files='filelists/ljs_audiopaths_text_sid_val_filelist.txt',
        text_cleaners=['english_cleaners'],
        p_arpabet=1.0,
        cmudict_path=None,

        #==============================#
        # Audio Parameters             #
        #==============================#
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,
        harm_thresh=0.25,
        f0_min = 80.0,
        f0_max = 880.0,

        #==============================#
        # Model Parameters             #
        #==============================#
        n_symbols=len(symbols),
        symbols_embedding_dim=512,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,
        p_teacher_forcing=1.0,     # TODO: Re-add it from Mellotron

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        # Speaker embedding          # TODO: Train a multi-speaker model And add it back in
        use_speaker_embedding=False,
        n_speakers=123,              
        speaker_embedding_dim=128,   

        # Reference encoder
        ref_enc_filters=[32, 32, 64, 64, 128, 128],
        ref_enc_size=[3, 3],
        ref_enc_strides=[2, 2],
        ref_enc_pad=[1, 1],
        ref_enc_gru_size=128,

        # Style Token Layer
        token_embedding_size=512,
        token_num=64,
        num_heads=8,

        # TP-GST parameters
        tpcw_gru_hidden_state_dim=128,
        tpse_gru_hidden_state_dim=128,
        tpse_fc_layer_dim=128,
        tpse_fc_layers=3,
        tpse_linear_fc_layer_dim=256,
        tpse_linear_fc_layers=3,
        tp_gst_use_bert=True,

        # BERT parameters
        bert_encoder_dim=768,
        bert_checkpoint_path='bert/',
        bert_config_path='bert/config.json',
        bert_vocab_path='bert/vocab.txt',
        bert_cased=True,
        bert_pretrained=True,
        bert_save_in_checkpoint=True,
        bert_load_from_checkpoint=True,
        bert_train=True,

        #==============================#
        # Optimization Hyperparameters #
        #==============================#
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        learning_rate_min=1e-5,
        learning_rate_anneal=50000,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,

        batch_size=8,
        val_batch_size=8,
        num_workers=8,
        val_num_workers=2,

        pin_worker = True,
        val_pin_worker = True,
        mask_padding=True,  # set model's padded outputs to padded values

        #==============================#
        # Guided Attention             #
        #==============================#
        use_guided_attention = True,
        guided_attention_sigma=0.2,
        guided_attention_weight=1.0,
    )
    
    return hparams
