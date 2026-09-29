from typing import Any, Dict, Callable, ContextManager
from types import SimpleNamespace
import os
import atexit
from omegaconf import OmegaConf, DictConfig
from uuid import uuid4
import csv

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from ssl_space.utils import distributed
from ssl_space.configs import load_full_config_eval
from ssl_space.datasets import make_dataset
from ssl_space.transforms import make_transform_pipeline

from ssl_space.methods_downstream import METHODS_DOWNSTREAM, BaseLinear
from ssl_space.utils.misc import MetricLogger, set_seed, has_batchnorms, append_csv_row

@atexit.register
def _cleanup():
    """
    Naive process group destruction to destroy process on early termination
    as well as the expected termination.
    """
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

### TRAINING ###

@torch.no_grad()
def run_validation(
    model: BaseLinear,
    loader_eval: DataLoader
):
    model.train(mode=False)
    for batch_idx, batch in enumerate(loader_eval):
        batch = model.move_batch_to_device(batch)
        model.validation_step(batch=batch)
    model.train(mode=True)

    return

def train_one_epoch(
        model: BaseLinear,
        model_ddp: DistributedDataParallel,
        dataloader: DataLoader, 
        optimizer: torch.optim.Optimizer, 
        lr_scheduler: Any,
        wd_scheduler: Any,
        autocast_ctx: Callable[[], ContextManager[None]],
        scaler: torch.amp.GradScaler,
        epoch_idx: int,
):
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch_idx, model.max_epochs)

    for batch_idx, batch in enumerate(metric_logger.log_every(dataloader, print_freq=25, header=header)):
        batch = model.move_batch_to_device(batch)
        model.optimizer_zero_grad(optimizer)

        with autocast_ctx():
            outs: Dict[str, Any] = model_ddp(batch, batch_idx, epoch_idx)   # go through DDP to activate forward hooks correctly
            targets = batch[-1]
            outs["train_loss"] = model.learning_algorithm(outs, targets, batch_idx, epoch_idx)  # compute the loss

        scaler.scale(outs["train_loss"]).backward()
        scaler.unscale_(optimizer)
        
        scaler.step(optimizer)
        scaler.update()

        # logging
        torch.cuda.synchronize()
        metric_logger.update(train_loss=outs["train_loss"])
        metric_logger.update(acc1=outs.get('acc1', -1))
        for i, group in enumerate(optimizer.param_groups):
            name = group.get("name", f"group{i}")
            if name == "backbone":
                metric_logger.update(**{
                    f"lr_{name}": group["lr"],
                    f"wd_{name}": group["weight_decay"]
                })

        # step the schedulers
        if lr_scheduler and lr_scheduler["interval"] == "step":
            if batch_idx % lr_scheduler["frequency"] == 0:
                lr_scheduler["lr_scheduler"].step()

        if wd_scheduler and wd_scheduler["interval"] == "step":
            if batch_idx % wd_scheduler["frequency"] == 0:
                wd_scheduler["wd_scheduler"].step()

    if lr_scheduler and lr_scheduler["interval"] == "epoch":
        if epoch_idx % lr_scheduler["frequency"] == 0:
            lr_scheduler["lr_scheduler"].step()
    
    if wd_scheduler and wd_scheduler["interval"] == "epoch":
        if epoch_idx % wd_scheduler["frequency"] == 0:
            wd_scheduler["wd_scheduler"].step()


    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return

def train_model(
        model: BaseLinear,
        model_ddp: DistributedDataParallel, 
        loader_train: DataLoader, 
        loader_eval: DataLoader | None,
        optimizer: torch.optim.Optimizer, 
        lr_scheduler: Any,
        wd_scheduler: Any,
        autocast_ctx: Callable[[], ContextManager[None]],
        scaler: torch.amp.GradScaler,
        start_epoch: int,
) -> Dict[str, Any]:
    model_ddp.train(mode=True)
    for epoch_idx in range(start_epoch, model.max_epochs):

        print(f"\nEpoch {epoch_idx}/{model.max_epochs}")
        
        loader_train.sampler.set_epoch(epoch_idx)
        
        train_one_epoch(
            model,
            model_ddp,
            loader_train, 
            optimizer, 
            lr_scheduler,
            wd_scheduler,
            autocast_ctx,
            scaler,
            epoch_idx, 
        )

    if loader_eval:
        run_validation(
            model=model, 
            loader_eval=loader_eval
        )

    metrics = model.on_validation_end()
    return metrics

def main(cfg: DictConfig, log_info: SimpleNamespace | None, seed) -> int:
    
    ### ------------------ AUG & DATA PIPES ------------------ ###

    aug_train_pipe = make_transform_pipeline(
        dset_cfg=cfg.dataset_train, 
        aug_cfg=cfg.augmentation_train
    )
    
    aug_eval_pipe = make_transform_pipeline(
        dset_cfg=cfg.dataset_eval, 
        aug_cfg=cfg.augmentation_eval
    ) if cfg.augmentation_eval is not None else None

    dset_train = make_dataset(
        dset_cfg=cfg.dataset_train, 
        transform=aug_train_pipe,
        target_transform=None,
    )

    dset_eval = make_dataset(
        dset_cfg=cfg.dataset_eval, 
        transform=aug_eval_pipe,
        target_transform=None
    ) if cfg.dataset_eval is not None else None

    ### ------ CREATE THE SAMPLER(s) and DATA LOADER(s) ------ ###

    sampler_train = DistributedSampler(
        dataset=dset_train,
        num_replicas=distributed.get_world_size(),
        rank=distributed.get_rank(),
        shuffle=True,
        seed=seed,
        drop_last=True,
    )
    sampler_eval = DistributedSampler(
        dataset=dset_eval,
        num_replicas=distributed.get_world_size(),
        rank=distributed.get_rank(),
        shuffle=False,
        seed=0,
        drop_last=False,
    ) if dset_eval else None

    loader_train = DataLoader(
        dataset=dset_train,
        batch_size=cfg.optimizer.batch_size,
        sampler=sampler_train,
        num_workers=cfg.performance.num_workers,  
        pin_memory=True,
        drop_last=True,
    )
    loader_eval = DataLoader(
        dataset=dset_eval, 
        batch_size=cfg.dataset_eval.inference_batch_size if cfg.dataset_eval.inference_batch_size is not None else cfg.optimizer.batch_size, 
        sampler=sampler_eval, 
        num_workers=cfg.performance.num_workers,
        pin_memory=True, 
        drop_last=False,
    ) if dset_eval else None

    ipe_train = len(loader_train)

    ### --------------- INITIALISE THE MODEL ------------------ ###

    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    model: BaseLinear = METHODS_DOWNSTREAM[cfg.method_eval.name](cfg, ipe_train)
    if cfg.performance.disable_channel_last:
        model = model.to(device=device)
    else:
        model = model.to(device=device, memory_format=torch.channels_last)
    
    model.device = device

    if has_batchnorms(model, include_sync_bn=True) and cfg.performance.sync_batchnorm:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    model_ddp = DistributedDataParallel(model, device_ids=[int(os.environ['LOCAL_RANK'])])

    ### ------------- INITIALISE THE OPTIMIZERS --------------- ###

    optimizers, lr_schedulers, wd_schedulers = model.configure_optimizers()
    optimizer = optimizers[0]
    lr_scheduler = lr_schedulers[0] if lr_schedulers else None
    wd_scheduler = wd_schedulers[0] if wd_schedulers else None

    ### ----------------- AMP / GRADSCALERS ------------------- ###

    use_amp = (cfg.performance.precision == "mixed-fp16")
    autocast_ctx = lambda: torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)
    print(f"Using AMP: {use_amp} | precision {cfg.performance.precision}")

    ### ----------------------- TRAIN ------------------------- ###

    metrics = train_model(
        model=model,
        model_ddp=model_ddp,
        loader_train=loader_train,
        loader_eval=loader_eval,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        wd_scheduler=wd_scheduler,
        autocast_ctx=autocast_ctx,
        scaler=scaler,
        start_epoch=0,
    )
       
    return metrics

def setup_logging_dirs(cfg: DictConfig) -> SimpleNamespace:
    pretrain_dir = cfg.method_eval.pretrain_dir     
    assert os.path.exists(pretrain_dir), f"pretrain dir {pretrain_dir} does not exist."
    run_id = uuid4().hex[:6]
    if cfg.method_eval.finetune:
        eval_type = "finetuned"
    else:
        eval_type = "frozen"
    eval_dir = os.path.join(pretrain_dir, cfg.dataset_train.name, eval_type, run_id)

    print('Creating directory {}'.format(eval_dir))
    os.makedirs(eval_dir, exist_ok=False)

    log_dir = os.path.join(eval_dir, 'logs')
    os.mkdir(log_dir)

    # dump the parameters to the directory for tracking
    OmegaConf.save(config=cfg, f=os.path.join(eval_dir, 'params-eval.yaml'), resolve=True)
    return SimpleNamespace(run_id=run_id, eval_dir=eval_dir, log_dir=log_dir)

if __name__ == '__main__':
    _SEEDS = [0, 1, 2]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", type=str, required=True)
    cli_args = parser.parse_args()

    cfg = load_full_config_eval(cli_args.fname)

    assert cfg.method_eval.name in METHODS_DOWNSTREAM, f"method {cfg.method_eval.name} not supported."

    # setup the distributed. Assumes running with torchrun
    distributed.init_distributed_mode()
    
    log_info = setup_logging_dirs(cfg) if distributed.is_main_process() else None
    print(OmegaConf.to_yaml(cfg))
    torch.backends.cudnn.benchmark = True
    
    # run for seeds
    all_metrics = [] 
    for seed in _SEEDS:
        set_seed(seed)
        metrics = main(cfg, log_info, seed=seed)
        print(metrics)
        all_metrics.append(metrics)
    
    assert len(all_metrics) == len(_SEEDS)
    if distributed.is_main_process():
        import numpy as np
        from scipy import stats

        # Get all metric keys
        metric_keys = list(all_metrics[0].keys())
        
        # Compute statistics for each metric
        statsistics = {}
        for key in metric_keys:
            values = [metrics[key] for metrics in all_metrics]
            mean = np.mean(values)
            std = np.std(values)
            
            # Compute 95% confidence interval
            n = len(values)
            if n > 1:
                t_value = stats.t.ppf(0.975, n-1)  # 95% confidence
                ci = t_value * (std / np.sqrt(n))
                ci_lower = mean - ci
                ci_upper = mean + ci
                ci_str = f"[{ci_lower:.4f}, {ci_upper:.4f}]"
            else:
                ci_lower = ci_upper = mean
                ci_str = "N/A (only one run)"
            
            statsistics[key] = {
                'mean': mean,
                'std': std,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'mean_±_std': f"{mean:.4f} ± {std:.4f}",
                '95%_ci': ci_str
            }
        
        # Write to CSV
        stats_csv_path = os.path.join(log_info.log_dir, "test_result.csv")
        with open(stats_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'mean', 'std', 'mean_±_std', 'ci_lower', 'ci_upper', '95%_ci'])
            for key in metric_keys:
                writer.writerow([
                    key,
                    f"{statsistics[key]['mean']:.4f}",
                    f"{statsistics[key]['std']:.4f}",
                    statsistics[key]['mean_±_std'],
                    f"{statsistics[key]['ci_lower']:.4f}",
                    f"{statsistics[key]['ci_upper']:.4f}",
                    statsistics[key]['95%_ci']
                ])

    print(f"Finished")