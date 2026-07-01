import os
import time
import torch
import argparse

from utils.basic_utils import mkdirp, load_json, save_json, make_zipfile, dict_to_markdown
import shutil

class BaseOptions(object):
    saved_option_filename = "opt.json"
    ckpt_filename = "model.ckpt"
    tensorboard_log_dir = "tensorboard_log"
    train_log_filename = "train.log.txt"
    eval_log_filename = "eval.log.txt"
    train_resume_data_options = (
        "temporal_aug_prob",
        "temporal_aug_min_keep",
        "context_extend_prob",
        "context_extend_max_frac",
        "temporal_mask_prob",
        "temporal_mask_n",
        "temporal_mask_max_len",
        "feat_noise_prob",
        "feat_noise_std",
        "multi_moment_prob",
        "aug_stop_epoch",
        "length_balance",
        "length_balance_bins",
        "position_jitter_prob",
        "position_jitter_context_sec",
        "position_jitter_max_shift_frac",
    )

    def __init__(self):
        self.parser = None
        self.initialized = False
        self.opt = None

    def initialize(self):
        self.initialized = True
        parser = argparse.ArgumentParser()
        parser.add_argument("--dset_name", type=str, choices=["hl", 'tvsum', "charades_sta"])
        parser.add_argument("--dset_domain", type=str, choices=["BK", "BT", "DS", "FM", "GA", "MS", "PK", "PR", "VT", "VU"], 
                            help="Domain to train for tvsum dataset. (Only used for tvsum)")
        
        parser.add_argument("--eval_split_name", type=str, default="val",
                            help="should match keys in video_duration_idx_path, must set for VCMR")
        parser.add_argument("--debug", action="store_true",
                            help="debug (fast) mode, break all loops, do not load all data into memory.")
        parser.add_argument("--data_ratio", type=float, default=1.0,
                            help="how many training and eval data to use. 1.0: use all, 0.1: use 10%."
                                 "Use small portion for debug purposes. Note this is different from --debug, "
                                 "which works by breaking the loops, typically they are not used together.")
        parser.add_argument("--results_root", type=str, default="results")
        parser.add_argument("--exp_id", type=str, default=None, help="id of this run, required at training")
        parser.add_argument("--seed", type=int, default=2018, help="random seed")
        parser.add_argument("--device", type=int, default=0, help="0 cuda, -1 cpu")
        parser.add_argument("--num_workers", type=int, default=4,
                            help="num subprocesses used to load the data, 0: use main process")
        parser.add_argument("--no_pin_memory", action="store_true",
                            help="Don't use pin_memory=True for dataloader. "
                                 "ref: https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4")

        # training config
        parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        parser.add_argument("--lr_drop", type=int, default=400, help="drop learning rate to 1/10 every lr_drop epochs")
        parser.add_argument("--lr_scheduler", type=str, default="auto", choices=["auto", "step", "cosine", "linear"],
                            help="LR scheduler. auto uses cosine when --cosine_T0 > 0, otherwise step.")
        parser.add_argument("--lrf", type=float, default=None,
                            help="Final LR ratio for cosine/linear LambdaLR. Defaults to --cosine_eta_min_ratio.")
        parser.add_argument("--warmup_epochs", type=int, default=0,
                            help="linear LR warmup epochs. 0 disables warmup")
        parser.add_argument("--cosine_T0", type=int, default=0,
                            help="if > 0, use cosine annealing with warm restarts; first cycle length")
        parser.add_argument("--cosine_Tmult", type=int, default=2,
                            help="cycle length multiplier for cosine warm restarts")
        parser.add_argument("--cosine_eta_min_ratio", type=float, default=0.01,
                            help="minimum LR ratio for cosine schedule (eta_min/base_lr)")
        parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
        parser.add_argument("--n_epoch", type=int, default=200, help="number of epochs to run")
        parser.add_argument("--max_es_cnt", type=int, default=200,
                            help="number of epochs to early stop, use -1 to disable early stop")
        parser.add_argument("--bsz", type=int, default=32, help="mini-batch size")
        parser.add_argument("--eval_bsz", type=int, default=100,
                            help="mini-batch size at inference, for query")
        parser.add_argument("--grad_clip", type=float, default=0.1, help="perform gradient clip, -1: disable")
        parser.add_argument("--ema_decay", type=float, default=0.0,
                            help="EMA decay for model weights. Set <=0 to disable.")
        parser.add_argument("--ema_start_epoch", type=int, default=0,
                            help="Epoch index (1-based) to start EMA updates.")
        parser.add_argument("--ema_scheduler", action="store_true",
                            help="Enable scheduled EMA decay warmup. Disabled keeps fixed --ema_decay behavior.")
        parser.add_argument("--ema_start_decay", type=float, default=0.99,
                            help="Initial EMA decay when --ema_scheduler is enabled.")
        parser.add_argument("--ema_warmup_updates", type=int, default=2000,
                            help="Number of EMA updates to ramp from --ema_start_decay to --ema_decay.")
        parser.add_argument("--ema_update_every", type=int, default=1,
                            help="Run one EMA update every N optimizer updates.")
        parser.add_argument("--ema_schedule", type=str, default="cosine", choices=["cosine", "linear", "constant"],
                            help="EMA decay schedule used when --ema_scheduler is enabled.")
        parser.add_argument("--eval_epoch_interval", type=int, default=5,
                            help="Run validation every N epochs.")
        parser.add_argument("--eval_every_epoch_after", type=int, default=-1,
                            help="If > 0, run validation every epoch once epoch >= this value (1-based).")
        parser.add_argument("--eval_untrained", action="store_true", help="Evaluate on un-trained model")
        parser.add_argument("--resume", type=str, default=None,
                            help="checkpoint path to resume or evaluate, without --resume_all this only load weights")
        parser.add_argument("--resume_all", action="store_true",
                            help="if --resume_all, load optimizer/scheduler/epoch as well")
        parser.add_argument("--start_epoch", type=int, default=None,
                            help="if None, will be set automatically when using --resume_all")

        # Data config
        parser.add_argument("--max_q_l", type=int, default=32)
        parser.add_argument("--max_v_l", type=int, default=75)
        parser.add_argument("--clip_length", type=int, default=2)
        parser.add_argument("--max_windows", type=int, default=5)
        parser.add_argument("--v_feat_len_mode", type=str, default="min", choices=["min", "max", "time_grid"],
                            help="How to align multiple video feature streams before concatenation.")

        parser.add_argument("--train_path", type=str, default=None)
        parser.add_argument("--eval_path", type=str, default=None,
                            help="Evaluating during training, for Dev set. If None, will only do training, ")
        parser.add_argument("--no_norm_vfeat", action="store_true", help="Do not do normalize video feat")
        parser.add_argument("--no_norm_tfeat", action="store_true", help="Do not do normalize text feat")
        parser.add_argument("--v_feat_dirs", type=str, nargs="+",
                            help="video feature dirs. If more than one, will concat their features. "
                                 "Note that sub ctx features are also accepted here.")
        parser.add_argument("--t_feat_dir", type=str, help="text/query feature dir")
        parser.add_argument("--a_feat_dir", type=str, help="audio feature dir")
        parser.add_argument("--v_feat_dim", type=int, help="video feature dim")
        parser.add_argument("--t_feat_dim", type=int, help="text/query feature dim")
        parser.add_argument("--a_feat_dim", type=int, help="audio feature dim")
        parser.add_argument("--ctx_mode", type=str, default="video_tef")

        # Model config
        parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                            help="Type of positional embedding to use on top of the image features")
        # * Transformer
        parser.add_argument('--enc_layers', default=2, type=int,
                            help="Number of encoding layers in the transformer")
        parser.add_argument('--dec_layers', default=2, type=int,
                            help="Number of decoding layers in the transformer")
        parser.add_argument('--dim_feedforward', default=1024, type=int,
                            help="Intermediate size of the feedforward layers in the transformer blocks")
        parser.add_argument('--hidden_dim', default=256, type=int,
                            help="Size of the embeddings (dimension of the transformer)")
        parser.add_argument('--input_dropout', default=0.5, type=float,
                            help="Dropout applied in input")
        parser.add_argument('--dropout', default=0.1, type=float,
                            help="Dropout applied in the transformer")
        parser.add_argument("--txt_drop_ratio", default=0, type=float,
                            help="drop txt_drop_ratio tokens from text input. 0.1=10%")
        parser.add_argument("--temporal_aug_prob", type=float, default=0.0,
                            help="Probability of applying temporal crop augmentation per sample. 0 disables.")
        parser.add_argument("--temporal_aug_min_keep", type=float, default=0.5,
                            help="Minimum fraction of the video to retain after temporal crop (in clips).")
        parser.add_argument("--context_extend_prob", type=float, default=0.0,
                            help="Probability of padding context from another video to reduce normalised GT width. 0 disables.")
        parser.add_argument("--context_extend_max_frac", type=float, default=1.0,
                            help="Maximum extra clips to add as a fraction of the current video length.")
        parser.add_argument("--temporal_mask_prob", type=float, default=0.0,
                            help="Probability of masking non-GT temporal clips per sample. 0 disables.")
        parser.add_argument("--temporal_mask_n", type=int, default=0,
                            help="Number of non-GT temporal spans to mask per augmented sample.")
        parser.add_argument("--temporal_mask_max_len", type=int, default=0,
                            help="Maximum masked temporal span length in clips. 0 disables.")
        parser.add_argument("--feat_noise_prob", type=float, default=0.0,
                            help="Probability of adding Gaussian noise to video features. 0 disables.")
        parser.add_argument("--feat_noise_std", type=float, default=0.0,
                            help="Stddev for Gaussian video feature noise.")
        parser.add_argument("--multi_moment_prob", type=float, default=0.0,
                            help="Probability of pasting another video's GT region as hard negative context. 0 disables.")
        parser.add_argument("--position_jitter_prob", type=float, default=0.0,
                            help="Probability of applying position-jitter augmentation per sample. 0 disables.")
        parser.add_argument("--position_jitter_context_sec", type=float, default=2.0,
                            help="Context reserve (seconds) on each side of the GT moment moved with it during jitter.")
        parser.add_argument("--position_jitter_max_shift_frac", type=float, default=0.0,
                            help="Maximum slide distance as fraction of ctx_l. 0 means relocate anywhere.")
        parser.add_argument("--aug_stop_epoch", type=int, default=0,
                            help="disable train-time augmentations after this epoch "
                                 "(1-indexed, inclusive); 0 = keep aug for the whole run. "
                                 "With --resume_all, restored from the checkpoint opt.")
        parser.add_argument("--length_balance", action="store_true",
                            help="Use WeightedRandomSampler when annotation GT-width bins are non-uniform.")
        parser.add_argument("--length_balance_bins", type=float, nargs="+", default=[0.1, 0.25],
                            help="Annotation-width bin edges for length-balanced sampling (normalised GT width thresholds).")
        parser.add_argument("--use_txt_pos", action="store_true", help="use position_embedding for text as well.")
        parser.add_argument('--nheads', default=8, type=int,
                            help="Number of attention heads inside the transformer's attentions")
        parser.add_argument('--num_queries', default=10, type=int,
                            help="Number of query slots")
        parser.add_argument("--query_init", default="random", choices=["random", "temporal_anchors"],
                            help="Query reference initialization strategy.")
        parser.add_argument("--query_anchor_widths", default="", type=str,
                            help="Comma-separated normalized widths for temporal anchor query initialization.")
        parser.add_argument('--pre_norm', action='store_true')
        # other model configs
        parser.add_argument("--n_input_proj", type=int, default=2, help="#layers to encoder input")
        parser.add_argument("--video_input_proj", type=str, default="linear", choices=["linear", "conv"],
                            help="Projection type for video input features.")
        parser.add_argument("--contrastive_hdim", type=int, default=64, help="dim for contrastive embeddings")
        parser.add_argument("--temperature", type=float, default=0.07, help="temperature nce contrastive_align_loss")
        # Loss
        parser.add_argument("--lw_saliency", type=float, default=1.,
                            help="weight for saliency loss, set to 0 will ignore")
        parser.add_argument("--saliency_margin", type=float, default=0.2)
        parser.add_argument("--intra_video_hard_neg_ratio", type=float, default=0.0,
                            help="Fraction of neg clips drawn from other moments of the same video (Level A). 0 disables.")
        parser.add_argument("--intra_video_hardneg_iou_thd", type=float, default=0.1,
                            help="Seconds-IoU threshold above which a same-video moment is excluded from the hard pool.")
        parser.add_argument("--saliency_hardneg_margin", type=float, default=0.4,
                            help="Margin for the intra-video hard-negative saliency term (Level B).")
        parser.add_argument("--hardneg_loss_coef", type=float, default=0.0,
                            help="Coefficient for the intra-video hard-negative loss term. 0 disables (Level B).")
        parser.add_argument("--hardneg_warmup_epoch", type=int, default=0,
                            help="Epoch before which the hard-negative loss term is zero.")
        parser.add_argument("--hardneg_ramp_epoch", type=int, default=-1,
                            help="Epoch at which the hard-negative loss reaches full strength. <=0 means instant after warmup.")
        # Unified proposal refinement and localization-quality stage
        parser.add_argument("--use_stage2", action="store_true",
                            help="Enable joint proposal localization and quality refinement.")
        parser.add_argument("--stage2_dim", type=int, default=128,
                            help="Hidden dimension for Stage 2 localization and quality heads.")
        parser.add_argument("--stage2_inner_bins", type=int, default=4,
                            help="Number of ordered samples pooled inside each proposal.")
        parser.add_argument("--stage2_boundary_samples", type=int, default=2,
                            help="Number of samples pooled on each side of each boundary.")
        parser.add_argument("--stage2_max_shift_clips", type=float, default=1.0,
                            help="Maximum start/end correction measured in valid video clips.")
        parser.add_argument("--stage2_shift_frac", type=float, default=0.0,
                            help="Width-proportional boundary shift fraction for Stage-2 refinement (0 disables).")
        parser.add_argument("--stage2_positive_iou", type=float, default=0.2,
                            help="Minimum base-proposal IoU for Stage 2 localization supervision.")
        parser.add_argument("--stage2_start_epoch", type=int, default=10,
                            help="Last epoch trained with Stage 1 losses only.")
        parser.add_argument("--stage2_joint_epoch", type=int, default=30,
                            help="Last epoch where Stage 2 inputs are detached from Stage 1.")
        parser.add_argument("--stage2_boundary_loss_coef", type=float, default=0.0)
        parser.add_argument("--stage2_giou_loss_coef", type=float, default=0.0)
        parser.add_argument("--stage2_quality_loss_coef", type=float, default=0.0)
        parser.add_argument("--stage2_at_inference", action="store_true",
                            help="Use refined spans and class-times-quality scores at inference.")
        parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                            help="Disables auxiliary decoding losses (loss at each layer)")
        parser.add_argument("--span_loss_type", default="l1", type=str, choices=['l1', 'ce', 'dfl', 'fdr'],
                            help="l1: (center-x, width) regression. ce: (st_idx, ed_idx) classification. "
                                 "dfl: start/end boundary distributions with expectation decoding. "
                                 "fdr: residual boundary-offset distributions around decoder references.")
        parser.add_argument("--dfl_num_bins", default=16, type=int,
                            help="Number of distribution bins for --span_loss_type dfl.")
        parser.add_argument("--dfl_ref_prior_sigma", default=2.0, type=float,
                            help="Gaussian prior sigma in DFL bins for decoder reference-conditioned logits.")
        parser.add_argument("--fdr_num_bins", default=32, type=int,
                            help="Number of offset distribution bins for --span_loss_type fdr.")
        parser.add_argument("--fdr_reg_scale", default=1.5, type=float,
                            help="Curvature for non-uniform FDR offset bins; >1 is denser around zero.")
        parser.add_argument("--fdr_min_ref_width", default=-1.0, type=float,
                            help="Minimum reference width for FDR offset scaling. <=0 uses 1 / max_v_l.")
        parser.add_argument("--contrastive_align_loss", action="store_true",
                            help="Enable contrastive_align_loss between matched query spans and text.")
        parser.add_argument("--contrastive_start_epoch", default=0, type=int,
                            help="1-based epoch to start contrastive loss. 0 enables it from the beginning.")
        parser.add_argument("--contrastive_decay_epoch", default=-1, type=int,
                            help="1-based epoch to switch contrastive loss to contrastive_decay_coef. <=0 disables decay.")
        parser.add_argument("--contrastive_decay_coef", default=0.0, type=float,
                            help="contrastive_align_loss weight after contrastive_decay_epoch.")
        # * Matcher
        parser.add_argument('--set_cost_span', default=10, type=float,
                            help="L1 span coefficient in the matching cost")
        parser.add_argument('--set_cost_giou', default=1, type=float,
                            help="giou span coefficient in the matching cost")
        parser.add_argument('--set_cost_class', default=4, type=float,
                            help="Class coefficient in the matching cost")
        parser.add_argument("--matching_type", default="hungarian",
                            choices=["hungarian", "tal"],
                            help="Matcher used for the final decoder layer.")
        parser.add_argument("--tal_topk", default=2, type=int,
                            help="Number of top task-aligned queries selected per target for --matching_type tal.")
        parser.add_argument("--tal_alpha", default=1.0, type=float,
                            help="Classification score exponent for task-aligned matching.")
        parser.add_argument("--tal_beta", default=6.0, type=float,
                            help="Temporal IoU exponent for task-aligned matching.")
        parser.add_argument("--aux_matching_type", default="hungarian",
                            choices=["hungarian", "one_to_many"],
                            help="Matcher used only for auxiliary decoder losses.")
        parser.add_argument("--aux_one_to_many_k", default=2, type=int,
                            help="Number of lowest-cost auxiliary queries selected per target when "
                                 "--aux_matching_type one_to_many is enabled.")

        # * Loss coefficients
        parser.add_argument('--span_loss_coef', default=10, type=float)
        parser.add_argument("--span_xx_loss_coef", default=0.0, type=float,
                            help="Optional L1 loss weight on normalized start/end span boundaries.")
        parser.add_argument("--fgl_loss_coef", default=None, type=float,
                            help="FGL loss weight for --span_loss_type fdr. Defaults to span_loss_coef.")
        parser.add_argument("--go_lsd_loss_coef", default=0.0, type=float,
                            help="GO-LSD/DDF self-distillation weight for FDR auxiliary decoder layers.")
        parser.add_argument("--go_lsd_temperature", default=2.0, type=float,
                            help="Temperature for GO-LSD distribution KL.")
        parser.add_argument("--go_lsd_start_epoch", default=0, type=int,
                            help="1-based epoch to start GO-LSD. 0 enables it from the beginning.")
        parser.add_argument('--giou_loss_coef', default=1, type=float)
        parser.add_argument("--width_loss_coef", default=0.0, type=float,
                            help="Optional width regularization loss weight.")
        parser.add_argument("--width_loss_type", default="log", choices=["none", "l1", "log"],
                            help="Width regularization type. 'log' penalizes relative width errors.")
        parser.add_argument('--label_loss_coef', default=4, type=float)
        parser.add_argument("--label_loss_type", default="ce", choices=["ce", "quality", "vfl"],
                            help="Label loss type: ce uses hard foreground/background labels; quality uses "
                                 "Hungarian matched IoU-softened labels; vfl uses Varifocal Loss.")
        parser.add_argument("--vfl_alpha", default=0.75, type=float,
                            help="Negative sample weight for --label_loss_type vfl.")
        parser.add_argument("--vfl_gamma", default=2.0, type=float,
                            help="Focusing exponent for --label_loss_type vfl.")
        parser.add_argument("--quality_label_loss", action="store_true",
                            help="Deprecated. Use --label_loss_type quality.")
        parser.add_argument("--quality_label_strength", default=0.5, type=float,
                            help="Strength for softening matched IoU label targets. 0 keeps hard positives; "
                                 "1 uses raw IoU after ramp.")
        parser.add_argument("--quality_label_iou_gamma", default=1.0, type=float,
                            help="Exponent applied to matched IoU quality targets. 1 preserves current behavior; "
                                 ">1 sharpens high-IoU ranking.")
        parser.add_argument("--quality_label_warmup_epoch", default=10, type=int,
                            help="Keep hard positive labels before this 1-based epoch.")
        parser.add_argument("--quality_label_ramp_epoch", default=30, type=int,
                            help="Finish ramping from hard labels to IoU targets at this 1-based epoch.")
        parser.add_argument('--eos_coef', default=0.1, type=float,
                            help="Relative classification weight of the no-object class")
        parser.add_argument("--contrastive_align_loss_coef", default=0.0, type=float)
        parser.add_argument("--best_metric", default="MR-full-mAP",
                            help="metric key from validation brief metrics used for best checkpoint/early stopping. "
                                 "Also supports MR-full-R1@0.5+0.7.")

        parser.add_argument("--no_sort_results", action="store_true",
                            help="do not sort results, use this for moment query visualization")
        parser.add_argument("--max_before_nms", type=int, default=10)
        parser.add_argument("--max_after_nms", type=int, default=10)
        parser.add_argument("--conf_thd", type=float, default=0.0, help="only keep windows with conf >= conf_thd")
        parser.add_argument("--nms_thd", type=float, default=-1,
                            help="additionally use non-maximum suppression "
                                 "(or non-minimum suppression for distance)"
                                 "to post-processing the predictions. "
                                 "-1: do not use nms. [0, 1]")
        self.parser = parser

    def display_save(self, opt):
        args = vars(opt)
        # Display settings
        print(dict_to_markdown(vars(opt), max_str_len=120))
        # Save settings
        if not isinstance(self, TestOptions):
            option_file_path = os.path.join(opt.results_dir, self.saved_option_filename)  # not yaml file indeed
            save_json(args, option_file_path, save_pretty=True)

    def parse(self, a_feat_dir=None):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()

        if opt.debug:
            opt.results_root = os.path.sep.join(opt.results_root.split(os.path.sep)[:-1] + ["debug_results", ])
            opt.num_workers = 0

        if isinstance(self, TestOptions):
            # modify model_dir to absolute path
            # opt.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", opt.model_dir)
            opt.model_dir = os.path.dirname(opt.resume)
            if a_feat_dir is not None:
                opt.a_feat_dir = a_feat_dir
            saved_options = load_json(os.path.join(opt.model_dir, self.saved_option_filename))
            for arg in saved_options:  # use saved options to overwrite all BaseOptions args.
                if arg not in ["results_root", "num_workers", "nms_thd", "debug",  # "max_before_nms", "max_after_nms"
                               "max_pred_l", "min_pred_l",
                               "resume", "resume_all", "no_sort_results"]:
                    setattr(opt, arg, saved_options[arg])
            # opt.no_core_driver = True
            if opt.eval_results_dir is not None:
                opt.results_dir = opt.eval_results_dir
            self._validate_aug_options(opt)
            self._normalize_query_anchor_widths(opt)
            if opt.dfl_num_bins < 2:
                raise ValueError("--dfl_num_bins must be >= 2.")
            if opt.dfl_ref_prior_sigma <= 0:
                raise ValueError("--dfl_ref_prior_sigma must be > 0.")
            self._validate_label_loss_options(opt)
            self._validate_fdr_options(opt)
            self._validate_tal_options(opt)
            self._validate_stage2_options(opt)
        else:
            self._restore_train_resume_data_options(opt)
            self._validate_aug_options(opt)
            self._normalize_query_anchor_widths(opt)
            if opt.dfl_num_bins < 2:
                raise ValueError("--dfl_num_bins must be >= 2.")
            if opt.dfl_ref_prior_sigma <= 0:
                raise ValueError("--dfl_ref_prior_sigma must be > 0.")
            self._validate_label_loss_options(opt)
            self._validate_fdr_options(opt)
            self._validate_tal_options(opt)
            self._validate_stage2_options(opt)
            if opt.exp_id is None:
                raise ValueError("--exp_id is required for at a training option!")

            ctx_str = opt.ctx_mode + "_sub" if any(["sub_ctx" in p for p in opt.v_feat_dirs]) else opt.ctx_mode
            opt.results_dir = os.path.join(opt.results_root,
                                           "-".join([opt.dset_name, ctx_str, opt.exp_id,
                                                     time.strftime("%Y_%m_%d_%H_%M_%S")]))
            mkdirp(opt.results_dir)
            save_fns = ['vmr_detr/modeling/model.py', 'vmr_detr/modeling/transformer.py']
            for save_fn in save_fns:
                shutil.copyfile(save_fn, os.path.join(opt.results_dir, os.path.basename(save_fn)))

            # save a copy of current code
            code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            code_zip_filename = os.path.join(opt.results_dir, "code.zip")
            make_zipfile(code_dir, code_zip_filename,
                         enclosing_dir="code",
                         exclude_dirs_substring="results",
                         exclude_dirs=["results", "debug_results", "__pycache__"],
                         exclude_extensions=[".pyc", ".ipynb", ".swap"], )

        self.display_save(opt)

        opt.ckpt_filepath = os.path.join(opt.results_dir, self.ckpt_filename)
        opt.train_log_filepath = os.path.join(opt.results_dir, self.train_log_filename)
        opt.eval_log_filepath = os.path.join(opt.results_dir, self.eval_log_filename)
        opt.tensorboard_log_dir = os.path.join(opt.results_dir, self.tensorboard_log_dir)
        opt.device = torch.device("cuda" if opt.device >= 0 else "cpu")
        opt.pin_memory = not opt.no_pin_memory

        opt.use_tef = "tef" in opt.ctx_mode
        opt.use_video = "video" in opt.ctx_mode
        if not opt.use_video:
            opt.v_feat_dim = 0
        if opt.use_tef:
            opt.v_feat_dim += 2

        self.opt = opt
        return opt

    @classmethod
    def _restore_train_resume_data_options(cls, opt):
        if not getattr(opt, "resume_all", False) or getattr(opt, "resume", None) is None:
            return
        checkpoint = torch.load(opt.resume, map_location="cpu", weights_only=False)
        saved_opt = checkpoint.get("opt")
        if saved_opt is None:
            return
        for name in cls.train_resume_data_options:
            if isinstance(saved_opt, dict):
                if name in saved_opt:
                    setattr(opt, name, saved_opt[name])
            elif hasattr(saved_opt, name):
                setattr(opt, name, getattr(saved_opt, name))

    @staticmethod
    def _validate_aug_options(opt):
        if not 0.0 <= opt.temporal_aug_prob <= 1.0:
            raise ValueError("--temporal_aug_prob must be in [0, 1].")
        if not 0.0 <= opt.context_extend_prob <= 1.0:
            raise ValueError("--context_extend_prob must be in [0, 1].")
        if not 0.0 <= opt.temporal_mask_prob <= 1.0:
            raise ValueError("--temporal_mask_prob must be in [0, 1].")
        if not 0.0 <= opt.feat_noise_prob <= 1.0:
            raise ValueError("--feat_noise_prob must be in [0, 1].")
        if not 0.0 <= opt.multi_moment_prob <= 1.0:
            raise ValueError("--multi_moment_prob must be in [0, 1].")
        if not 0.0 < opt.temporal_aug_min_keep <= 1.0:
            raise ValueError("--temporal_aug_min_keep must be in (0, 1].")
        if not opt.context_extend_max_frac >= 0.0:
            raise ValueError("--context_extend_max_frac must be >= 0.")
        if opt.temporal_mask_n < 0:
            raise ValueError("--temporal_mask_n must be >= 0.")
        if opt.temporal_mask_max_len < 0:
            raise ValueError("--temporal_mask_max_len must be >= 0.")
        if opt.feat_noise_std < 0:
            raise ValueError("--feat_noise_std must be >= 0.")
        if opt.aug_stop_epoch < 0:
            raise ValueError("--aug_stop_epoch must be >= 0 (0 disables the cutoff).")
        if not 0.0 <= opt.position_jitter_prob <= 1.0:
            raise ValueError("--position_jitter_prob must be in [0, 1].")
        if opt.position_jitter_context_sec < 0:
            raise ValueError("--position_jitter_context_sec must be >= 0.")
        if opt.position_jitter_max_shift_frac < 0:
            raise ValueError("--position_jitter_max_shift_frac must be >= 0.")

    @staticmethod
    def _normalize_query_anchor_widths(opt):
        raw_widths = getattr(opt, "query_anchor_widths", "")
        if raw_widths is None:
            opt.query_anchor_widths = None
            return
        if isinstance(raw_widths, str):
            values = [value.strip() for value in raw_widths.split(",")]
            values = [float(value) for value in values if value]
            opt.query_anchor_widths = values if values else None
            return
        opt.query_anchor_widths = [float(value) for value in raw_widths] if raw_widths else None

    @staticmethod
    def _validate_label_loss_options(opt):
        if not hasattr(opt, "label_loss_type"):
            opt.label_loss_type = "ce"
        if not hasattr(opt, "vfl_alpha"):
            opt.vfl_alpha = 0.75
        if not hasattr(opt, "vfl_gamma"):
            opt.vfl_gamma = 2.0
        if getattr(opt, "quality_label_loss", False):
            if opt.label_loss_type not in ("ce", "quality"):
                raise ValueError("--quality_label_loss is only compatible with --label_loss_type quality.")
            opt.label_loss_type = "quality"
        if opt.label_loss_type not in ("ce", "quality", "vfl"):
            raise ValueError("--label_loss_type must be one of ce, quality, or vfl.")
        if opt.vfl_alpha < 0:
            raise ValueError("--vfl_alpha must be >= 0.")
        if opt.vfl_gamma < 0:
            raise ValueError("--vfl_gamma must be >= 0.")
        if opt.label_loss_type in ("quality", "vfl") and opt.matching_type != "hungarian":
            raise ValueError("--label_loss_type quality/vfl is only supported with --matching_type hungarian.")
        if opt.label_loss_type in ("quality", "vfl") and opt.span_loss_type == "ce":
            raise ValueError("--label_loss_type quality/vfl requires --span_loss_type l1, dfl, or fdr.")

    @staticmethod
    def _validate_tal_options(opt):
        if opt.tal_topk < 1:
            raise ValueError("--tal_topk must be >= 1.")
        if opt.tal_alpha < 0 or opt.tal_beta < 0:
            raise ValueError("--tal_alpha and --tal_beta must be non-negative.")
        if opt.matching_type == "tal" and opt.span_loss_type not in ("l1", "dfl", "fdr"):
            raise ValueError("--matching_type tal requires --span_loss_type l1, dfl, or fdr.")

    @staticmethod
    def _validate_stage2_options(opt):
        defaults = {
            "use_stage2": False,
            "stage2_dim": 128,
            "stage2_inner_bins": 4,
            "stage2_boundary_samples": 2,
            "stage2_max_shift_clips": 1.0,
            "stage2_shift_frac": 0.0,
            "stage2_positive_iou": 0.2,
            "stage2_start_epoch": 10,
            "stage2_joint_epoch": 30,
            "stage2_boundary_loss_coef": 0.0,
            "stage2_giou_loss_coef": 0.0,
            "stage2_quality_loss_coef": 0.0,
            "stage2_at_inference": False,
        }
        for name, value in defaults.items():
            if not hasattr(opt, name):
                setattr(opt, name, value)

        if opt.stage2_dim < 1:
            raise ValueError("--stage2_dim must be >= 1.")
        if opt.stage2_inner_bins < 1:
            raise ValueError("--stage2_inner_bins must be >= 1.")
        if opt.stage2_boundary_samples < 1:
            raise ValueError("--stage2_boundary_samples must be >= 1.")
        if opt.stage2_max_shift_clips <= 0:
            raise ValueError("--stage2_max_shift_clips must be > 0.")
        if opt.stage2_shift_frac < 0:
            raise ValueError("--stage2_shift_frac must be >= 0.")
        if not 0 <= opt.stage2_positive_iou <= 1:
            raise ValueError("--stage2_positive_iou must be in [0, 1].")
        if opt.stage2_start_epoch < 0:
            raise ValueError("--stage2_start_epoch must be >= 0.")
        if opt.stage2_joint_epoch < opt.stage2_start_epoch:
            raise ValueError("--stage2_joint_epoch must be >= --stage2_start_epoch.")
        coefficients = (
            opt.stage2_boundary_loss_coef,
            opt.stage2_giou_loss_coef,
            opt.stage2_quality_loss_coef,
        )
        if min(coefficients) < 0:
            raise ValueError("Stage 2 loss coefficients must be >= 0.")
        if opt.use_stage2 and opt.span_loss_type not in ("l1", "dfl", "fdr"):
            raise ValueError("--use_stage2 requires --span_loss_type l1, dfl, or fdr.")
        if opt.use_stage2 and opt.dset_name == "tvsum":
            raise ValueError("--use_stage2 is only supported for matcher-based VMR training.")
        if any(coef > 0 for coef in coefficients) and not opt.use_stage2:
            raise ValueError("Stage 2 loss coefficients require --use_stage2.")
        if opt.stage2_at_inference and not opt.use_stage2:
            raise ValueError("--stage2_at_inference requires --use_stage2.")

    @staticmethod
    def _validate_fdr_options(opt):
        if opt.fdr_num_bins < 2:
            raise ValueError("--fdr_num_bins must be >= 2.")
        if opt.fdr_reg_scale <= 0:
            raise ValueError("--fdr_reg_scale must be > 0.")
        if opt.fdr_min_ref_width <= 0:
            opt.fdr_min_ref_width = 1.0 / float(opt.max_v_l)
        if opt.fgl_loss_coef is not None and opt.fgl_loss_coef < 0:
            raise ValueError("--fgl_loss_coef must be >= 0.")
        if opt.go_lsd_loss_coef < 0:
            raise ValueError("--go_lsd_loss_coef must be >= 0.")
        if opt.go_lsd_temperature <= 0:
            raise ValueError("--go_lsd_temperature must be > 0.")
        if opt.go_lsd_start_epoch < 0:
            raise ValueError("--go_lsd_start_epoch must be >= 0.")
        if opt.go_lsd_loss_coef > 0 and opt.span_loss_type != "fdr":
            raise ValueError("--go_lsd_loss_coef > 0 requires --span_loss_type fdr.")
        if opt.go_lsd_loss_coef > 0 and not opt.aux_loss:
            raise ValueError("--go_lsd_loss_coef > 0 requires auxiliary decoder losses.")
        if opt.go_lsd_loss_coef > 0 and opt.dset_name == "tvsum":
            raise ValueError("--go_lsd_loss_coef > 0 requires matcher-based VMR training.")
        if not hasattr(opt, "span_xx_loss_coef"):
            opt.span_xx_loss_coef = 0.0
        if opt.span_xx_loss_coef < 0:
            raise ValueError("--span_xx_loss_coef must be >= 0.")
        if opt.span_xx_loss_coef > 0 and opt.span_loss_type == "ce":
            raise ValueError("--span_xx_loss_coef > 0 requires --span_loss_type l1, dfl, or fdr.")
        if opt.width_loss_coef < 0:
            raise ValueError("--width_loss_coef must be >= 0.")
        if opt.width_loss_coef > 0 and opt.width_loss_type != "none" and opt.span_loss_type == "ce":
            raise ValueError("--width_loss_type l1/log requires --span_loss_type l1, dfl, or fdr.")
        if opt.quality_label_warmup_epoch < 0:
            raise ValueError("--quality_label_warmup_epoch must be >= 0.")
        if opt.quality_label_ramp_epoch < opt.quality_label_warmup_epoch:
            raise ValueError("--quality_label_ramp_epoch must be >= --quality_label_warmup_epoch.")
        if opt.quality_label_strength < 0 or opt.quality_label_strength > 1:
            raise ValueError("--quality_label_strength must be in [0, 1].")
        if opt.quality_label_iou_gamma <= 0:
            raise ValueError("--quality_label_iou_gamma must be > 0.")

class TestOptions(BaseOptions):
    """add additional options for evaluating"""

    def initialize(self):
        BaseOptions.initialize(self)
        # also need to specify --eval_split_name
        self.parser.add_argument("--eval_id", type=str, help="evaluation id")
        self.parser.add_argument("--eval_results_dir", type=str, default=None,
                                 help="dir to save results, if not set, fall back to training results_dir")
        self.parser.add_argument("--model_dir", type=str,
                                 help="dir contains the model file, will be converted to absolute path afterwards")
