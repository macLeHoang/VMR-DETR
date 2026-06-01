import math
import re
import random
import numpy as np
import torch


MAIN_METRIC_ORDER = [
    "MR-full-mAP",
    "MR-full-mAP@0.5",
    "MR-full-mAP@0.75",
    "MR-full-mIoU",
    "MR-full-R1@0.3",
    "MR-full-R1@0.5",
    "MR-full-R1@0.7",
    "MR-short-mAP",
    "MR-middle-mAP",
    "MR-long-mAP",
]


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def _format_float(value, precision=4):
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def _format_table(headers, rows):
    str_rows = [[str(cell) for cell in row] for row in rows]
    if not str_rows:
        return []
    widths = [
        max(len(str(header)), *(len(row[idx]) for row in str_rows))
        for idx, header in enumerate(headers)
    ]
    header_line = "  " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers))
    sep_line = "  " + "-+-".join("-" * width for width in widths)
    body_lines = [
        "  " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers)))
        for row in str_rows
    ]
    return [header_line, sep_line, *body_lines]


def _is_aux_loss(name):
    return re.search(r"_\d+$", name) is not None


def _loss_rows(loss_meters):
    rows = []
    for name in sorted(loss_meters.keys()):
        if _is_aux_loss(name):
            continue
        meter = loss_meters[name]
        rows.append([name, _format_float(meter.avg, precision=4)])
    return rows


def _ordered_metric_names(metrics):
    ordered = [name for name in MAIN_METRIC_ORDER if name in metrics]
    ordered.extend(name for name in metrics.keys() if name not in ordered)
    return ordered


def _primary_metric_name(metrics):
    for name in ("MR-full-mAP", "mAP"):
        if name in metrics:
            return name
    return next(iter(metrics), None)


def log_validation_summary(logger, epoch, loss_meters, metrics, title="validation summary"):
    logger.info(f"Epoch {epoch:3d} {title}")

    rows = _loss_rows(loss_meters)
    if rows:
        logger.info("[Losses]")
        for line in _format_table(["Loss", "Avg"], rows):
            logger.info(line)

    metric_names = _ordered_metric_names(metrics)
    primary_name = _primary_metric_name(metrics)
    if metric_names and primary_name is not None:
        logger.info("[Main Metrics]")
        headers = ["primary", *metric_names]
        row = [_format_float(metrics[primary_name], precision=2)]
        row.extend(_format_float(metrics[name], precision=2) for name in metric_names)
        for line in _format_table(headers, [row]):
            logger.info(line)


def should_run_eval(opt, epoch_i):
    epoch_num = epoch_i + 1
    after_threshold = opt.eval_every_epoch_after > 0 and epoch_num >= opt.eval_every_epoch_after
    interval_hit = opt.eval_epoch_interval > 0 and (epoch_num % opt.eval_epoch_interval == 0)
    return after_threshold or interval_hit


class ModelEMA(object):
    def __init__(self, model, decay):
        self.decay = float(decay)
        self.shadow_params = {}
        self.shadow_buffers = {}
        self.backup_params = None
        self.backup_buffers = None
        self.reset(model)

    def reset(self, model):
        self.shadow_params = {}
        self.shadow_buffers = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.detach().clone()
        for name, buf in model.named_buffers():
            self.shadow_buffers[name] = buf.detach().clone()
        self.backup_params = None
        self.backup_buffers = None

    def update(self, model):
        one_minus_decay = 1.0 - self.decay
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad or name not in self.shadow_params:
                    continue
                self.shadow_params[name].mul_(self.decay).add_(param.detach(), alpha=one_minus_decay)
            for name, buf in model.named_buffers():
                if name in self.shadow_buffers:
                    self.shadow_buffers[name].copy_(buf.detach())

    def store(self, model):
        self.backup_params = {}
        self.backup_buffers = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup_params[name] = param.detach().clone()
        for name, buf in model.named_buffers():
            self.backup_buffers[name] = buf.detach().clone()

    def copy_to(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad or name not in self.shadow_params:
                    continue
                param.copy_(self.shadow_params[name])
            for name, buf in model.named_buffers():
                if name in self.shadow_buffers:
                    buf.copy_(self.shadow_buffers[name])

    def restore(self, model):
        if self.backup_params is None or self.backup_buffers is None:
            return
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad or name not in self.backup_params:
                    continue
                param.copy_(self.backup_params[name])
            for name, buf in model.named_buffers():
                if name in self.backup_buffers:
                    buf.copy_(self.backup_buffers[name])
        self.backup_params = None
        self.backup_buffers = None

    def state_dict(self):
        return dict(
            decay=self.decay,
            shadow_params={k: v.detach().clone() for k, v in self.shadow_params.items()},
            shadow_buffers={k: v.detach().clone() for k, v in self.shadow_buffers.items()},
        )

    def load_state_dict(self, state):
        if state is None:
            return
        if "decay" in state:
            self.decay = float(state["decay"])
        self.shadow_params = {k: v.detach().clone() for k, v in state.get("shadow_params", {}).items()}
        self.shadow_buffers = {k: v.detach().clone() for k, v in state.get("shadow_buffers", {}).items()}


class EMAScheduler(object):
    def __init__(
        self,
        ema,
        start_decay=0.99,
        target_decay=0.999,
        warmup_updates=2000,
        update_every=1,
        schedule="cosine",
    ):
        self.ema = ema
        self.start_decay = float(start_decay)
        self.target_decay = float(target_decay)
        self.warmup_updates = max(int(warmup_updates), 1)
        self.update_every = max(int(update_every), 1)
        self.schedule = schedule
        self.num_updates = 0
        self.num_ema_updates = 0

    def _progress(self):
        if self.warmup_updates <= 1:
            return 1.0
        return min(max((self.num_ema_updates - 1) / (self.warmup_updates - 1), 0.0), 1.0)

    def get_decay(self):
        progress = self._progress()

        if self.schedule == "linear":
            decay = self.start_decay + progress * (self.target_decay - self.start_decay)
        elif self.schedule == "cosine":
            decay = self.target_decay - (
                self.target_decay - self.start_decay
            ) * (math.cos(math.pi * progress) + 1.0) / 2.0
        else:
            decay = self.target_decay

        return decay

    def step(self, model):
        self.num_updates += 1

        if self.num_updates % self.update_every != 0:
            return None

        self.num_ema_updates += 1
        decay = self.get_decay()

        self.ema.decay = decay ** self.update_every
        self.ema.update(model)

        return self.ema.decay

    def state_dict(self):
        return dict(
            start_decay=self.start_decay,
            target_decay=self.target_decay,
            warmup_updates=self.warmup_updates,
            update_every=self.update_every,
            schedule=self.schedule,
            num_updates=self.num_updates,
            num_ema_updates=self.num_ema_updates,
        )

    def load_state_dict(self, state):
        if state is None:
            return
        self.start_decay = float(state.get("start_decay", self.start_decay))
        self.target_decay = float(state.get("target_decay", self.target_decay))
        self.warmup_updates = max(int(state.get("warmup_updates", self.warmup_updates)), 1)
        self.update_every = max(int(state.get("update_every", self.update_every)), 1)
        self.schedule = state.get("schedule", self.schedule)
        self.num_updates = int(state.get("num_updates", 0))
        self.num_ema_updates = int(state.get("num_ema_updates", 0))


def build_datasets(opt):
    from vmr_detr.data.start_end_dataset import StartEndDataset
    from vmr_detr.data.start_end_dataset_audio import StartEndDataset_audio

    if opt.a_feat_dir is None:
        dataset_config = dict(
            dset_name=opt.dset_name,
            data_path=opt.train_path,
            v_feat_dirs=opt.v_feat_dirs,
            q_feat_dir=opt.t_feat_dir,
            q_feat_type="last_hidden_state",
            max_q_l=opt.max_q_l,
            max_v_l=opt.max_v_l,
            ctx_mode=opt.ctx_mode,
            data_ratio=opt.data_ratio,
            normalize_v=not opt.no_norm_vfeat,
            normalize_t=not opt.no_norm_tfeat,
            clip_len=opt.clip_length,
            max_windows=opt.max_windows,
            span_loss_type=opt.span_loss_type,
            txt_drop_ratio=opt.txt_drop_ratio,
            dset_domain=opt.dset_domain,
            v_feat_len_mode=opt.v_feat_len_mode,
        )
        train_dataset = StartEndDataset(**dataset_config)
    else:
        dataset_config = dict(
            dset_name=opt.dset_name,
            data_path=opt.train_path,
            v_feat_dirs=opt.v_feat_dirs,
            q_feat_dir=opt.t_feat_dir,
            a_feat_dir=opt.a_feat_dir,
            q_feat_type="last_hidden_state",
            max_q_l=opt.max_q_l,
            max_v_l=opt.max_v_l,
            ctx_mode=opt.ctx_mode,
            data_ratio=opt.data_ratio,
            normalize_v=not opt.no_norm_vfeat,
            normalize_t=not opt.no_norm_tfeat,
            clip_len=opt.clip_length,
            max_windows=opt.max_windows,
            span_loss_type=opt.span_loss_type,
            txt_drop_ratio=opt.txt_drop_ratio,
            dset_domain=opt.dset_domain,
        )
        train_dataset = StartEndDataset_audio(**dataset_config)

    if opt.eval_path is None:
        return train_dataset, None

    dataset_config["data_path"] = opt.eval_path
    dataset_config["txt_drop_ratio"] = 0
    if opt.a_feat_dir is None:
        eval_dataset = StartEndDataset(**dataset_config)
    else:
        eval_dataset = StartEndDataset_audio(**dataset_config)
    return train_dataset, eval_dataset
