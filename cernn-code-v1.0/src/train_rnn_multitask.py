import os
import wandb
import hydra
import omegaconf
import numpy as np
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as L  # used to be 'pl' keep seeing 'L'
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.pytorch_models import LightningRNNModule

# from src.models.leaky_rnn_base import RNNLayer
from src.tools import get_task_hp
from src.train_utils import configure_name, init_model
from src.dataset import YangTasks, collate_fn
from src.paths import checkpoint_dir, CORTICAL_AREAS_PATH
import pickle


@hydra.main(
    config_path="hydraconfigs",
    config_name="train_CERNN_default",
    version_base=None,
)
def main(cfg: omegaconf.DictConfig) -> None:
    task_hp = get_task_hp(cfg)
    # seed the task generation
    task_hp["rng"] = np.random.RandomState(cfg.seed)
    # seed the model initialisation and dataloader order (but not the chosen task)
    L.seed_everything(cfg.seed)

    if "fast_dev_run" in cfg.train and cfg.train.fast_dev_run == True:
        fast_dev_run = True
        cfg.wandb.mode = "disabled"
    else:
        fast_dev_run = False

    wandb.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(**cfg.wandb, config=wandb.config)

    run_name = configure_name(cfg, task_hp)
    # cfg = resolve_duplicates(cfg)

    print(run_name)

    name = wandb.run.name + run_name

    wandb_logger = WandbLogger(log_model=True, name=name)

    dataset_train = YangTasks(task_hp, mode="train")
    dataset_val = YangTasks(
        task_hp, mode="val"
    )  # different from mode="test" in gen_trials() in RY code
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=1,  # batch size 1 here because we need all trials in batch to be same task/rule
        collate_fn=collate_fn,
        num_workers=cfg.train.n_workers,
        shuffle=True,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=1,
        collate_fn=collate_fn,
        num_workers=cfg.train.n_workers,
        shuffle=False,
    )

    hp_pl_module = cfg.model
    if cfg.model.from_pretrained is not None:
        try:
            model = LightningRNNModule.load_from_checkpoint(cfg.model.from_pretrained)
        except:
            print("could not load pretrained model from checkpoint")
            model = init_model(hp_pl_module, task_hp, LightningRNNModule)
    else:
        model = init_model(hp_pl_module, task_hp, LightningRNNModule)
    model.hp = hp_pl_module
    # model.add_regularisers(
    #     se_rule=hp_pl_module["se_rule"],
    #     se_ramping=hp_pl_module["se_ramping"],
    #     se_lambda=hp_pl_module["se_lambda"],
    # )
    wandb.watch(model.model, log="all")

    # save best performing and last model checkpoints
    model_checkpoint_dir = os.path.join(checkpoint_dir, cfg.model.name, name)
    os.makedirs(model_checkpoint_dir, exist_ok=True)
    # Save task_hp and hp_pl_module configurations as pickle files

    with open(os.path.join(model_checkpoint_dir, "task_hp.pkl"), "wb") as f:
        pickle.dump(task_hp, f)
        print("task_hp saved in:", f"{model_checkpoint_dir}/task_hp.pkl")

    with open(os.path.join(model_checkpoint_dir, "hp_pl_module.pkl"), "wb") as f:
        pickle.dump(hp_pl_module, f)
        print("hp_pl_module saved in:", f"{model_checkpoint_dir}/hp_pl_module.pkl")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_perf",
        mode="max",
        save_last=True,
        dirpath=model_checkpoint_dir,
        filename="{epoch:02d}-{val_perf:.2f}",
    )
    print("model checkpoint saved in:", f"{model_checkpoint_dir}/{name}")

    patience = cfg.train.epochs if cfg.model.from_pretrained is not None else 20
    early_stopping = EarlyStopping(
        monitor="val_perf",
        mode="max",
        patience=patience,  # stop after x epochs with no improvement
        stopping_threshold=cfg.model.target_perf,
    )

    train_kwargs = {
        "max_epochs": cfg.train.epochs,
        "logger": wandb_logger,
        "callbacks": [checkpoint_callback],
        "fast_dev_run": fast_dev_run,
    }

    if cfg.model.clip_grads == True:
        train_kwargs["gradient_clip_val"] = cfg.model.clip_grad_value
        train_kwargs["gradient_clip_algorithm"] = "value"

    # train on multiple GPUs and/or nodes on cluster
    if cfg.train.name == "ddp" or cfg.train.name == "bp":
        train_kwargs["accelerator"] = cfg.train.accelerator
        train_kwargs["strategy"] = cfg.train.strategy
        train_kwargs["devices"] = cfg.train.devices  # number of gpus
        train_kwargs["num_nodes"] = cfg.train.num_nodes

    trainer = Trainer(**train_kwargs)
    trainer.fit(
        model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val
    )

    wandb.finish()


# if __name__ == "__main__":

#     main()
