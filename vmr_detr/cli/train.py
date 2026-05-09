import os
import time
import json
import sys
import copy
from tqdm.auto import tqdm, trange
from collections import defaultdict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from vmr_detr.config.options import BaseOptions
from vmr_detr.data.start_end_dataset import \
    start_end_collate, prepare_batch_inputs
from vmr_detr.data.start_end_dataset_audio import \
    start_end_collate_audio, prepare_batch_inputs_audio
from vmr_detr.cli.inference import eval_epoch, start_inference, setup_model
from utils.basic_utils import AverageMeter
from utils.model_utils import count_parameters
from vmr_detr.cli.train_utils import (
    ModelEMA,
    build_datasets,
    log_validation_summary,
    set_seed,
    should_run_eval,
)

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                    level=logging.INFO)

COMPOSITE_BEST_METRICS = {
    "MR-full-R1@0.5+0.7": ("MR-full-R1@0.5", "MR-full-R1@0.7"),
}


def train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, ema=None):
    logger.info(f"[Epoch {epoch_i+1}]")
    model.train()
    criterion.train()
    if hasattr(criterion, "set_epoch"):
        criterion.set_epoch(epoch_i + 1)

    # init meters
    time_meters = defaultdict(AverageMeter)
    loss_meters = defaultdict(AverageMeter)

    num_training_examples = len(train_loader)
    timer_dataloading = time.time()

    pbar = tqdm(enumerate(train_loader), desc="Training Iteration", total=num_training_examples)
    for batch_idx, batch in pbar:
        time_meters["dataloading_time"].update(time.time() - timer_dataloading)

        timer_start = time.time()
        if opt.a_feat_dir is None:
            model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)
        else:
            model_inputs, targets = prepare_batch_inputs_audio(batch[1], opt.device, non_blocking=opt.pin_memory)
        time_meters["prepare_inputs_time"].update(time.time() - timer_start)
        timer_start = time.time()
        outputs = model(**model_inputs)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        time_meters["model_forward_time"].update(time.time() - timer_start)

        timer_start = time.time()
        optimizer.zero_grad()
        losses.backward()
        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        if ema is not None and (epoch_i + 1) >= opt.ema_start_epoch:
            ema.update(model)
        time_meters["model_backward_time"].update(time.time() - timer_start)

        loss_dict["loss_overall"] = losses.detach().item()  # for logging only
        for k, v in loss_dict.items():
            val = v.detach().item() if isinstance(v, torch.Tensor) else float(v)
            loss_meters[k].update(val * weight_dict[k] if k in weight_dict else val)

        timer_dataloading = time.time()
        if opt.debug and batch_idx == 3:
            break
        
        display_stats = {k: v.item() if isinstance(v, torch.Tensor) else v 
                 for k, v in loss_dict.items()}
        pbar.set_postfix(display_stats)

    to_write = opt.train_log_txt_formatter.format(
        time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
        epoch=epoch_i+1,
        loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()]))
    with open(opt.train_log_filepath, "a") as f:
        f.write(to_write)

    logger.info("Epoch time stats:")
    for name, meter in time_meters.items():
        d = {k: f"{getattr(meter, k):.4f}" for k in ["max", "min", "avg"]}
        logger.info(f"{name} ==> {d}")


def get_stop_score(metrics, opt, fallback_metric):
    brief = metrics["brief"]
    metric_name = opt.best_metric
    if metric_name in COMPOSITE_BEST_METRICS:
        required_metrics = COMPOSITE_BEST_METRICS[metric_name]
        missing = [name for name in required_metrics if name not in brief]
        if missing:
            available = ", ".join(brief.keys())
            raise KeyError(
                f"best metric '{metric_name}' requires {', '.join(required_metrics)}; "
                f"missing {', '.join(missing)}. Available metrics: {available}"
            )
        return sum(brief[name] for name in required_metrics), metric_name
    if metric_name not in brief and metric_name == "MR-full-mAP" and fallback_metric in brief:
        metric_name = fallback_metric
    if metric_name not in brief:
        available = ", ".join(brief.keys())
        raise KeyError(f"best metric '{opt.best_metric}' not found. Available metrics: {available}")
    return brief[metric_name], metric_name


def train(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt):
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)

    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    if opt.a_feat_dir is None:
        train_loader = DataLoader(
            train_dataset,
            collate_fn=start_end_collate,
            batch_size=opt.bsz,
            num_workers=opt.num_workers,
            shuffle=True,
            pin_memory=opt.pin_memory
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            collate_fn=start_end_collate_audio,
            batch_size=opt.bsz,
            num_workers=opt.num_workers,
            shuffle=True,
            pin_memory=opt.pin_memory
        )

    prev_best_score = 0.
    es_cnt = 0
    use_ema = opt.ema_decay > 0
    ema = ModelEMA(model, opt.ema_decay) if use_ema else None
    if ema is not None and hasattr(opt, "_ema_state_dict"):
        ema.load_state_dict(opt._ema_state_dict)
        logger.info(f"Restored EMA state from checkpoint (decay={ema.decay:.6f}).")
    if ema is not None:
        logger.info(f"EMA enabled with decay={ema.decay:.6f}, start_epoch={opt.ema_start_epoch}.")
    # start_epoch = 0
    if opt.start_epoch is None:
        start_epoch = -1 if opt.eval_untrained else 0
    else:
        start_epoch = opt.start_epoch
    save_submission_filename = "latest_{}_{}_preds.jsonl".format(opt.dset_name, opt.eval_split_name)

    for epoch_i in range(start_epoch, opt.n_epoch):
        if epoch_i > -1:
            train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, ema=ema)
            lr_scheduler.step()

        if opt.eval_path is not None and should_run_eval(opt, epoch_i):
            using_ema_eval = ema is not None and (epoch_i + 1) >= opt.ema_start_epoch
            raw_model_state = copy.deepcopy(model.state_dict())
            if using_ema_eval:
                ema.store(model)
                ema.copy_to(model)
            with torch.no_grad():
                metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = \
                    eval_epoch(model, val_dataset, opt, save_submission_filename, epoch_i, criterion, None)
            eval_model_state = copy.deepcopy(model.state_dict())
            if using_ema_eval:
                ema.restore(model)

            # log
            to_write = opt.eval_log_txt_formatter.format(
                time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i,
                loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in eval_loss_meters.items()]),
                eval_metrics_str=json.dumps(metrics_no_nms))

            with open(opt.eval_log_filepath, "a") as f:
                f.write(to_write)
            log_validation_summary(logger, epoch_i + 1, eval_loss_meters, metrics_no_nms["brief"])
            if metrics_nms is not None:
                log_validation_summary(logger, epoch_i + 1, eval_loss_meters, metrics_nms["brief"], title="nms validation summary")

            metrics = metrics_no_nms

            stop_score, stop_metric_name = get_stop_score(metrics, opt, "MR-full-mAP")
                
            if stop_score > prev_best_score:
                es_cnt = 0
                prev_best_score = stop_score

                checkpoint = {
                    "model": eval_model_state,
                    "model_raw": raw_model_state,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt
                }
                if ema is not None:
                    checkpoint["ema_state"] = ema.state_dict()
                torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"))

                best_file_paths = [e.replace("latest", "best") for e in latest_file_paths]
                for src, tgt in zip(latest_file_paths, best_file_paths):
                    os.renames(src, tgt)
                logger.info(f"The checkpoint file has been updated using {stop_metric_name}={stop_score:.4f}.")
            else:
                es_cnt += 1
                if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:  # early stop
                    with open(opt.train_log_filepath, "a") as f:
                        f.write(f"Early Stop at epoch {epoch_i}")
                    logger.info(f"\n>>>>> Early stop at epoch {epoch_i}  {prev_best_score}\n")
                    break

            # save ckpt
            checkpoint = {
                "model": eval_model_state,
                "model_raw": raw_model_state,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            if ema is not None:
                checkpoint["ema_state"] = ema.state_dict()
            torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt"))

        save_interval = 10 if "subs_train" in opt.train_path else 50  # smaller for pretrain
        if (epoch_i + 1) % save_interval == 0 or (epoch_i + 1) % opt.lr_drop == 0:  # additional copies
            using_ema_save = ema is not None and (epoch_i + 1) >= opt.ema_start_epoch
            raw_model_state = copy.deepcopy(model.state_dict())
            if using_ema_save:
                ema.store(model)
                ema.copy_to(model)
            save_model_state = copy.deepcopy(model.state_dict())
            if using_ema_save:
                ema.restore(model)
            checkpoint = {
                "model": save_model_state,
                "model_raw": raw_model_state,
                "optimizer": optimizer.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            if ema is not None:
                checkpoint["ema_state"] = ema.state_dict()
            torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", f"_e{epoch_i:04d}.ckpt"))

        if opt.debug:
            break

def train_hl(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt):
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)

    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
        pin_memory=opt.pin_memory
    )

    prev_best_score = 0.
    es_cnt = 0
    use_ema = opt.ema_decay > 0
    ema = ModelEMA(model, opt.ema_decay) if use_ema else None
    if ema is not None and hasattr(opt, "_ema_state_dict"):
        ema.load_state_dict(opt._ema_state_dict)
        logger.info(f"Restored EMA state from checkpoint (decay={ema.decay:.6f}).")
    if ema is not None:
        logger.info(f"EMA enabled with decay={ema.decay:.6f}, start_epoch={opt.ema_start_epoch}.")
    # start_epoch = 0
    if opt.start_epoch is None:
        start_epoch = -1 if opt.eval_untrained else 0
    else:
        start_epoch = opt.start_epoch
    save_submission_filename = "latest_{}_{}_preds.jsonl".format(opt.dset_name, opt.eval_split_name)
    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        if epoch_i > -1:
            train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, ema=ema)
            lr_scheduler.step()
        if opt.eval_path is not None and should_run_eval(opt, epoch_i):
            using_ema_eval = ema is not None and (epoch_i + 1) >= opt.ema_start_epoch
            raw_model_state = copy.deepcopy(model.state_dict())
            if using_ema_eval:
                ema.store(model)
                ema.copy_to(model)
            with torch.no_grad():
                metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = \
                    eval_epoch(model, val_dataset, opt, save_submission_filename, epoch_i, criterion, None)
            eval_model_state = copy.deepcopy(model.state_dict())
            if using_ema_eval:
                ema.restore(model)

            # log
            to_write = opt.eval_log_txt_formatter.format(
                time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i,
                loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in eval_loss_meters.items()]),
                eval_metrics_str=json.dumps(metrics_no_nms))

            with open(opt.eval_log_filepath, "a") as f:
                f.write(to_write)
            log_validation_summary(logger, epoch_i + 1, eval_loss_meters, metrics_no_nms["brief"])
            if metrics_nms is not None:
                log_validation_summary(logger, epoch_i + 1, eval_loss_meters, metrics_nms["brief"], title="nms validation summary")

            metrics = metrics_no_nms

            # stop_score = metrics["brief"]["MR-full-mAP"]
            stop_score, stop_metric_name = get_stop_score(metrics, opt, "mAP")
            if stop_score > prev_best_score:
                es_cnt = 0
                prev_best_score = stop_score

                checkpoint = {
                    "model": eval_model_state,
                    "model_raw": raw_model_state,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt
                }
                if ema is not None:
                    checkpoint["ema_state"] = ema.state_dict()
                torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"))

                best_file_paths = [e.replace("latest", "best") for e in latest_file_paths]
                for src, tgt in zip(latest_file_paths, best_file_paths):
                    os.renames(src, tgt)
                logger.info(f"The checkpoint file has been updated using {stop_metric_name}={stop_score:.4f}.")
            else:
                es_cnt += 1
                if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:  # early stop
                    with open(opt.train_log_filepath, "a") as f:
                        f.write(f"Early Stop at epoch {epoch_i}")
                    logger.info(f"\n>>>>> Early stop at epoch {epoch_i}  {prev_best_score}\n")
                    break

            # save ckpt
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            # torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt"))

        save_interval = 10 if "subs_train" in opt.train_path else 50  # smaller for pretrain
        if (epoch_i + 1) % save_interval == 0 or (epoch_i + 1) % opt.lr_drop == 0:  # additional copies
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            # torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", f"_e{epoch_i:04d}.ckpt"))

        if opt.debug:
            break

def start_training():
    logger.info("Setup config, data and model...")
    opt = BaseOptions().parse()
    set_seed(opt.seed)
    if opt.debug:  # keep the model run deterministically
        # 'cudnn.benchmark = True' enabled auto finding the best algorithm for a specific input/net config.
        # Enable this only when input size is fixed.
        cudnn.benchmark = False
        cudnn.deterministic = True
    train_dataset, eval_dataset = build_datasets(opt)

    model, criterion, optimizer, lr_scheduler = setup_model(opt)
    logger.info(f"Model {model}")
    count_parameters(model)
    logger.info("Start Training...")
    
    # For tvsum dataset, use train_hl function
    if opt.dset_name in ['tvsum']:
        train_hl(model, criterion, optimizer, lr_scheduler, train_dataset, eval_dataset, opt)
    else:
        train(model, criterion, optimizer, lr_scheduler, train_dataset, eval_dataset, opt)
    
    return opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"), opt.eval_split_name, opt.eval_path, opt.debug, opt


if __name__ == '__main__':
    best_ckpt_path, eval_split_name, eval_path, debug, opt = start_training()
    if not debug:
        input_args = ["--resume", best_ckpt_path,
                      "--eval_split_name", eval_split_name,
                      "--eval_path", eval_path]
        if opt.use_quality_ranking:
            input_args += ["--use_quality_ranking", "--quality_score_beta", str(opt.quality_score_beta)]

        import sys
        sys.argv[1:] = input_args
        logger.info("\n\n\nFINISHED TRAINING!!!")
        logger.info("Evaluating model at {}".format(best_ckpt_path))
        logger.info("Input args {}".format(sys.argv[1:]))
        start_inference(opt)
