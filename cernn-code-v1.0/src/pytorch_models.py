"""Custom RNN implementation"""

from typing import Any
import torch
from torch import nn
from omegaconf import OmegaConf

import numpy as np
import pytorch_lightning as L

from src.models.regularisers import regs_mapping

import src.task as task

# from src.models.leaky_rnn_base import Model
from src.models.CERNN import CERNNModel
from src.models.leaky_rnn import LeakyRNNModel
from src.train_utils import get_perf
from src.cortical_embedding import (
    MacaqueCorticalEmbedding,
    HumanCorticalEmbedding,
)
from src.analysis_connectivity import FLN_ij, calculate_spearman_exponential_fit


def gen_trials(rule, hp, mode, batch_size):
    if batch_size is None:
        trial = task.generate_trials(rule, hp, mode=mode)
    else:
        trial = task.generate_trials(rule, hp, mode=mode, batch_size=batch_size)
    trial = _gen_feed_dict(trial, hp)
    return trial


def _gen_feed_dict(trial, hp):
    n_time, batch_size = trial.x.shape[:2]
    if hp["in_type"] == "normal":
        pass
    elif hp["in_type"] == "multi":
        new_shape = [n_time, batch_size, hp["rule_start"] * hp["n_rule"]]

        x = np.zeros(new_shape, dtype=np.float32)
        for i in range(batch_size):
            ind_rule = np.argmax(trial.x[0, i, hp["rule_start"] :])
            i_start = ind_rule * hp["rule_start"]
            x[:, i, i_start : i_start + hp["rule_start"]] = trial.x[
                :, i, : hp["rule_start"]
            ]
        trial.x = x
    else:
        raise ValueError()

    trial.x = torch.tensor(trial.x)
    trial.y = torch.tensor(trial.y)
    trial.c_mask = torch.tensor(trial.c_mask).view(n_time, batch_size, -1)

    return trial


class LightningRNNModule(L.LightningModule):
    def __init__(self, hp):
        super().__init__()

        self.loss_fnc = (
            nn.MSELoss() if hp["loss_type"] == "lsq" else nn.CrossEntropyLoss()
        )
        self.optimiser = hp["optimizer"]

        model_kwargs = {
            "n_input": hp["n_input"],
            "n_output": hp["n_output"],
            "dt": hp["dt"],
            "tau": hp["tau"],
            "noise": hp["noise"],
            "w_rec_init": hp["w_rec_init"],
            "sigma_rec": hp["sigma_rec"],
            "activation": hp["activation"],
        }

        if hp["name"] == "cernn":
            if hp["species"] == "human":
                print(hp["sensory"], hp["motor"])
                ce = HumanCorticalEmbedding(
                    hp["duplicate"], hp["constraints"], hp["sensory"], hp["motor"]
                )
            elif hp["species"] == "macaque":
                ce = MacaqueCorticalEmbedding(
                    hp["duplicate"], hp["constraints"], hp["sensory"], hp["motor"]
                )
            else:
                raise NotImplementedError
            

            try:
                shuffle = OmegaConf.select(hp, "shuffle_d1", default=False)
            except Exception:
                shuffle = hp.get("shuffle_d1", False) if isinstance(hp, dict) else False

            
            self.model = CERNNModel(
                **model_kwargs,
                n_rnn=ce.distance_matrix.shape[0],
                ce=ce,
                shuffle_d1=shuffle,
            )

        elif hp["name"] == "leaky_rnn":
            self.model = LeakyRNNModel(
                **model_kwargs,
                n_rnn=hp["n_rnn"],
            )
        else:
            raise NotImplementedError
        

        regularisers = self.resolve_regularisers(hp["regularisers"])
        self.model.regularisers = nn.ModuleList(regularisers)

        self.save_hyperparameters()

    def resolve_arg(self, key, val):
        if key == "distance_matrix":
            return self.model.ce.distance_matrix
        if key == "sensory_indices":
            return self.model.rnn.rnncell.get_sensory_ind()
        if key == "area_mask":
            return self.model.ce.area_mask
        if key == "diag_mask":
            dim = self.model.rnn.rnncell.weight_hh.shape[0]
            return torch.eye(dim, dtype=torch.float32)
        if key == "duplicates":
            return self.model.ce.duplicates
        if key == "areas":
            return self.model.ce.cortical_areas

        return val

    def resolve_regularisers(self, regularisers):
        regs_list = []
        for reg_name, args in regularisers.items():
            # For each item, replace any "resolve" fields with actual references
            real_args = {k: self.resolve_arg(k, v) for k, v in args.items()}
            regs_list.append(regs_mapping[reg_name](**real_args))
        return regs_list

    def eval_step(self, batch):
        batch.x = batch.x.to(self.device)
        batch.y = batch.y.to(self.device)
        batch.c_mask = batch.c_mask.to(self.device)
        output, hidden = self.model(batch.x)
        return output, hidden

    def training_step(self, batch, batch_index):

        output, hidden = self.eval_step(batch)
        loss_total, loss, reg_losses = self.calculate_loss(output, hidden, batch)
        for name, reg_loss in reg_losses.items():
            self.log(f"train/{name}", reg_loss.item(), batch_size=batch.x.shape[1])
        # /user/home/od23963/.local/lib/python3.8/site-packages/pytorch_lightning/trainer/connectors/logger_connector/result.py:433: It is recommended to use `self.log('val_perf', ..., sync_dist=True)` when logging on epoch level in distributed setting to accumulate the metric across devices.
        self.log("train/task loss", loss.item(), batch_size=batch.x.shape[1])
        self.log("train/total loss", loss_total.item(), batch_size=batch.x.shape[1])
        # if self.se_ramping:
        #     self.log("se_loss", reg_losses["se1_loss"], batch_size=batch.x.shape[1])
        #     self.log("se_scale", self.se1_scale, batch_size=batch.x.shape[1])
        return loss_total

    def validation_step(self, batch, batch_index):
        """Validates over a batch of trails of one task.
        note: we can't do early stopping here because the current task might just be hard
        The datasteps are ordered and return
        """

        rule = batch.rule
        output, hidden = self.eval_step(batch)
        loss_total, loss, reg_losses = self.calculate_loss(output, hidden, batch)
        for name, reg_loss in reg_losses.items():
            self.log(f"validation/{name}", reg_loss, batch_size=batch.x.shape[1])
        perf = np.mean(get_perf(output.detach().cpu(), batch.y_loc))

        self.log("validation/val loss", loss_total, batch_size=batch.x.shape[1])
        self.log(f"val_perf_{rule}", np.mean(perf), batch_size=batch.x.shape[1])
        self.log(f"val_perf", perf, batch_size=batch.x.shape[1])

        try:
            zero_weights_thres = self.model.ce.zero_weights_thres
        except:
            zero_weights_thres = 0
        weights = self.model.rnn.rnncell.weight_hh.detach().cpu().numpy()
        spearman_w, exponential_w, density_w = calculate_spearman_exponential_fit(
            weights, self.model, thres=zero_weights_thres
        )
        r_w, _ = spearman_w
        _, l_w = exponential_w
        self.log("validation/spearman_corr", r_w, batch_size=batch.x.shape[1])
        self.log("validation/lambda", l_w, batch_size=batch.x.shape[1])
        self.log("validation/density", density_w, batch_size=batch.x.shape[1])

        FLNmatrx = FLN_ij(weights)
        spearman, exponential_, density = calculate_spearman_exponential_fit(
            FLNmatrx, self.model, zero_weights_thres
        )
        r, _ = spearman
        _, l = exponential_
        self.log("validation/spearman_corr_FLN", r, batch_size=batch.x.shape[1])
        self.log("validation/lambda_FLN", l, batch_size=batch.x.shape[1])
        self.log("validation/density_FLN", density, batch_size=batch.x.shape[1])

        return loss_total, perf

    def configure_optimizers(self):  # TODO
        if self.optimiser == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.hp["learning_rate"])
        elif self.optimiser == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.hp["learning_rate"])
        else:
            raise NotImplementedError

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        loss = self.trainer.callback_metrics["validation/val loss"]
        print("validation/epoch loss = ", loss.item())
        for rule in self.hp["rule_trains"]:
            try:
                performance = self.trainer.callback_metrics[f"val_perf_{rule}"]
                print(f"performance {rule} = ", performance.item())
            except:
                continue
        total_performance = self.trainer.callback_metrics["val_perf"]
        print(f"total performance step {self.global_step} = ", total_performance.item())

    # def on_train_epoch_start(self):
    #     """Each regulariser has a scheduler method (most are blank) for modulating the stenght"""
    #     super().on_train_epoch_start()
    #     if self.current_epoch > 0:
    #         for regulariser in self.regularisers:
    #             regulariser.on_train_epoch_start(
    #                 self.trainer.callback_metrics["val_perf"]
    #             )
    # if self.se_rule and self.se_ramping:
    #     if self.current_epoch > 0:
    #         if self.trainer.callback_metrics["val_perf"] > 0.8:
    #             self.se1_scale = torch.tensor(self.se1_scale_max)

    # Calculate the current epoch as a fraction of the total number of epochs
    # current_epoch_fraction = self.current_epoch / self.trainer.max_epochs

    # # Determine the scaling based on the current epoch fraction
    # if current_epoch_fraction <= 0.2:
    #     # First 20% of epochs: SE1 loss is not applied
    #     self.se1_scale = torch.tensor(0.0)
    # elif current_epoch_fraction <= 0.8:
    #     # From 20% to 80%: Scale SE1 loss up linearly
    #     scale_progress = (current_epoch_fraction - 0.2) / (0.8 - 0.2)
    #     self.se1_scale = torch.tensor(scale_progress * self.se1_scale_max)
    # else:
    #     # Last 20% of epochs: SE1 loss is at its maximum
    #     self.se1_scale = torch.tensor(self.se1_scale_max)

    def calculate_loss(self, output, hidden, trial):

        task_loss = self.loss_fnc(trial.c_mask * output, trial.c_mask * trial.y)
        loss_total = task_loss

        reg_losses = {}
        for regulariser in self.model.regularisers:
            val, name = regulariser(self.model, hidden)
            val = val.squeeze()
            reg_losses[name] = val
            loss_total += val

        return loss_total, task_loss, reg_losses
    
    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()

        cell = getattr(self.model.rnn, "rnncell", None)
        if cell is None:
            return

        # let the cell print once per epoch again
        if hasattr(cell, "printed_this_epoch"):
            cell.printed_this_epoch = False

        mode = getattr(cell, "dopamine_mod_target", "decay")
        if mode == "gain":
            dm = cell.d1_map

            mean = float(dm.mean().detach().cpu())
            std  = float(dm.std().detach().cpu())
            mn   = float(dm.min().detach().cpu())
            mx   = float(dm.max().detach().cpu())
            print(
                f"[Epoch {self.current_epoch}] D1 mode=gain, "
                f"gain={cell.dopamine_gain:.3f}, "
                f"map mean={mean:.3f}, std={std:.3f}, min={mn:.3f}, max={mx:.3f}"
            )
        else:
            print(
                f"[Epoch {self.current_epoch}] DA mode={mode}, "
                f"level={cell.dopamine_level:.3f}, gain={cell.dopamine_gain:.3f}"
            )


