from typing import Any, Dict, Callable, ContextManager
from types import SimpleNamespace
import os
import atexit
from omegaconf import OmegaConf, DictConfig
from uuid import uuid4
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from ssl_space.utils import distributed
from ssl_space.configs import load_full_config
from ssl_space.datasets import make_dataset
from ssl_space.transforms import make_transform_pipeline
from ssl_space.methods import METHODS, BaseMethod, BaseMethodMomentum
from ssl_space.utils.misc import MetricLogger, set_seed, has_batchnorms

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
    model: BaseMethod | BaseMethodMomentum,
    loader_eval: DataLoader,
    update_outputs: bool = True
):
    model.train(mode=False)
    for batch_idx, batch in enumerate(loader_eval):
        batch = model.move_batch_to_device(batch)
        model.validation_step(
            batch=batch, update_validation_step_outputs=update_outputs
        )
    model.train(mode=True)
    return

def train_one_epoch(
        model: BaseMethod | BaseMethodMomentum,
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
        
        # move data and group according to temporal and spatial resolution.
        batch = model.move_batch_to_device(batch)

        model.optimizer_zero_grad(optimizer)

        with autocast_ctx():
            outs: Dict[str, Any] = model_ddp(batch, batch_idx, epoch_idx)   # go through DDP to activate forward hooks correctly
            ssl_loss, loss_logging = model.learning_algorithm(outs, batch_idx, epoch_idx)
            outs["ssl_loss"] = ssl_loss
            bp_loss = model.before_backward_pass(outs, batch_idx, epoch_idx)

        scaler.scale(bp_loss).backward()
        scaler.unscale_(optimizer)
        
        model.after_backward_pass(batch_idx=batch_idx, epoch_idx=epoch_idx)
        
        scaler.step(optimizer)
        scaler.update()

        # when using gradscaler, we should check that the step was actually performed. 
        model.after_optimizer_step(batch_idx=batch_idx, epoch_idx=epoch_idx)

        # logging
        torch.cuda.synchronize()
        for name, v in loss_logging.items():
            metric_logger.update(**{name: v})
        metric_logger.update(ssl_loss=outs["ssl_loss"])
        metric_logger.update(cls_loss=outs.get("cls_loss", -1))
        metric_logger.update(acc1=outs.get('acc1', -1))
        metric_logger.update(acc5=outs.get('acc5', -1))
        for i, group in enumerate(optimizer.param_groups):
            name = group.get("name", f"group{i}")
            if name == "backbone":
                metric_logger.update(**{
                    f"lr_{name}": group["lr"],
                    f"wd_{name}": group["weight_decay"]
                })
        if hasattr(model, "momentum_updater"):
            metric_logger.update(EMA_tau=model.momentum_updater.cur_tau) # will be one step before as we have already stepped it - does not matter as its just logging.

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

    ddp_logging_data = model_ddp._get_ddp_logging_data()
    print("DDP stats:")
    for key in ("avg_forward_compute_time", "avg_backward_compute_time", "avg_backward_comm_time"):
        if key in ddp_logging_data:
            print(f"{key}: {ddp_logging_data[key]}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_model(
        model: BaseMethod | BaseMethodMomentum,
        model_ddp: DistributedDataParallel, 
        loader_train: DataLoader, 
        loader_eval: DataLoader | None,
        optimizer: torch.optim.Optimizer, 
        lr_scheduler: Any,
        wd_scheduler: Any,
        autocast_ctx: Callable[[], ContextManager[None]],
        scaler: torch.amp.GradScaler,
        start_epoch: int,
        eval_freq: int,
        ckpt_freq: int,
        log_info: SimpleNamespace|None,
        update_val_outputs: bool = True,
        writer: SummaryWriter = None
):
    model_ddp.train(mode=True)
    for epoch_idx in range(start_epoch, model.max_epochs):

        print(f"\nEpoch {epoch_idx}/{model.max_epochs}")
            
        if isinstance(loader_train, DataLoader):
            s = getattr(loader_train, "sampler", None)
            bs = getattr(loader_train, "batch_sampler", None)

            # Case 1: standard DistributedSampler path
            if s is not None and hasattr(s, "set_epoch"):
                s.set_epoch(epoch_idx)

            # Case 2: custom batch sampler (DualRatioBatchSampler) path
            elif bs is not None and hasattr(bs, "set_epoch"):
                bs.set_epoch(epoch_idx)
                
        metrics = train_one_epoch(
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

        human_epoch = epoch_idx + 1
        if (
            ckpt_freq > 0
            and human_epoch % ckpt_freq == 0
            and human_epoch != model.max_epochs
            and distributed.is_main_process()
        ):
            ckpt_running = {
                'epoch': human_epoch,
                'model_state_dict': model.state_dict(),
            }
            ckpt_running_path = os.path.join(log_info.models_dir, f"model_running_ep{human_epoch}.pth")
            torch.save(
                ckpt_running,
                f=ckpt_running_path
            )
            print(f"Saved ckpt model to: {ckpt_running_path}")

        if eval_freq != 0 and epoch_idx % eval_freq == 0:

            if loader_eval:
                run_validation(
                    model=model, 
                    loader_eval=loader_eval,
                    update_outputs=update_val_outputs
                )

            val_metrics = model.on_validation_epoch_end()
            assert val_metrics != {}, "running validation without collecting stats... Do not do that!"
            print("VALIDATION:")
            for key in val_metrics:
                val_metrics[key] = distributed.reduce_mean_scalar(val_metrics[key])

            print(val_metrics)

    return

def main(cfg: DictConfig, log_info: SimpleNamespace | None) -> int:
    
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

    sampler_train = DistributedSampler(
        dataset=dset_train,
        num_replicas=distributed.get_world_size(),
        rank=distributed.get_rank(),
        shuffle=True,
        seed=cfg.meta.seed,
        drop_last=True,
    )

    loader_train = DataLoader(
        dataset=dset_train,
        batch_size=cfg.optimizer.batch_size,
        sampler=sampler_train,
        num_workers=cfg.performance.num_workers,  
        pin_memory=True,
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

    loader_eval = DataLoader(
        dataset=dset_eval, 
        batch_size=cfg.optimizer.batch_size, 
        sampler=sampler_eval, 
        num_workers=cfg.performance.num_workers,
        pin_memory=True, 
        drop_last=False,
    ) if dset_eval else None

    ### --------------- INITIALISE THE MODEL ------------------ ###

    ipe_train = len(loader_train)
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    model: BaseMethod | BaseMethodMomentum = METHODS[cfg.method.name](cfg, ipe_train)
    if cfg.performance.disable_channel_last:
        model = model.to(device=device)
    else:
        model = model.to(device=device, memory_format=torch.channels_last)
    
    model.device = device

    if has_batchnorms(model, include_sync_bn=True): 
        print("Training uses batch-norm.")
        if cfg.performance.sync_batchnorm:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            print("Syncing Batchnorm!")
        else:
            print("Config specifies not to sync batchnorm!")
    else:
        print("No batchnorm detected.")

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

    train_model(
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
        eval_freq=cfg.online_eval.evaluation_freq,
        ckpt_freq=cfg.meta.ckpt_freq,
        log_info=log_info,
        update_val_outputs=True,
        writer=None,
    )

    ### ------------------------ SAVE ------------------------ ###

    if distributed.is_main_process():
        ckpt_final = {
            'epoch': model.max_epochs,
            'model_state_dict': model.state_dict(),
        }
        ckpt_final_path = os.path.join(log_info.models_dir, f"model_final_ep{model.max_epochs}.pth")
        torch.save(
            ckpt_final,
            f=ckpt_final_path
        )
        print(f"Saved final model to: {ckpt_final_path}")
    
    return 0

def setup_logging_dirs(cfg: DictConfig) -> SimpleNamespace:
    run_id = uuid4().hex[:12]
    save_name_pre = '{}_{}'.format(cfg.method.name, run_id)
    run_id_dir = os.path.join('exps', cfg.method.name, save_name_pre)

    if not os.path.exists(run_id_dir):
        print('Creating directory {}'.format(run_id_dir))
        os.makedirs(run_id_dir)

    log_dir = os.path.join(run_id_dir, 'logs')
    if not os.path.exists(log_dir):
        print('Creating directory {}'.format(log_dir))
        os.mkdir(log_dir)

    models_dir = os.path.join(run_id_dir, 'models')
    if not os.path.exists(models_dir):
        print('Creating directory {}'.format(models_dir))
        os.mkdir(models_dir)

    # dump the parameters to the directory for tracking
    OmegaConf.save(config=cfg, f=os.path.join(run_id_dir, 'params-pretrain.yaml'), resolve=True)
    
    return SimpleNamespace(run_id=run_id, run_id_dir=run_id_dir, models_dir=models_dir, log_dir=log_dir)

if __name__ == '__main__':
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", type=str, required=True)
    cli_args = parser.parse_args()

    cfg = load_full_config(cli_args.fname)
    assert cfg.method.name in METHODS, f"method {cfg.method.name} not supported."

    # setup the distributed. Assumes running with torchrun
    distributed.init_distributed_mode()

    # create directories for saving models and logging.
    log_info = setup_logging_dirs(cfg) if distributed.is_main_process() else None

    # set seed
    set_seed(cfg.meta.seed)
    torch.backends.cudnn.benchmark = True
    print(OmegaConf.to_yaml(cfg))

    if distributed.is_main_process():
        _t0 = time.perf_counter()
    
    exit_code = main(cfg, log_info)

    if distributed.is_dist_avail_and_initialized():
        distributed.synchronize()

    if distributed.is_main_process():
        _secs = time.perf_counter() - _t0
        _hours = _secs / 3_600.0
        print(f"Training finished : {exit_code} | Duration {_hours} h")