from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import hydra
import omegaconf

from src.pytorch_models import LightningRNNModule
from src.dataset import YangTasks, collate_fn
from src.tools import get_task_hp


@hydra.main(
    config_path="hydraconfigs",
    config_name="train_CERNN_default",
    version_base=None,
)
def evaluate_trained_model(cfg: omegaconf.DictConfig):
    checkpoint_path = cfg.checkpoint
    model = LightningRNNModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.freeze()

    # get same task config as used for training
    task_hp = get_task_hp(cfg)
    task_hp["rng"] = np.random.RandomState(cfg.seed)

    # if "baseline" in checkpoint_path:
    #     id = "baseline"
    # elif "vm" in checkpoint_path:
    #     id = "vm-ce" if "ce" in checkpoint_path else "vm"
    # else:
    #     id = "unknown"
    id = "cernn"
    # load datasets adapted from the Yang et al. 2019 paper
    test_dataset = YangTasks(task_hp, mode="test")
    dataloader_val = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=collate_fn,
        # num_workers=cfg.train.n_workers,
    )

    # print("TODO: evaluation pipeline not fully implemented")
    create_figure(model, dataloader_val, task_hp, id)

    # save_np_array_for_surface_plot(model, dataloader_val, task_hp, id)


def create_figure(model, dataloader, hp, id):
    for batch in dataloader:
        task = hp["rule_trains"][dataloader.dataset.current_rule_index - 1]
        fig, ax = plt.subplots(1, 4, figsize=(10, 5))
        ax[0].imshow(batch.x[:, 0, :].detach().cpu().numpy().T, aspect="auto")
        ax[0].set_title("Input")
        ax[3].imshow(batch.y[:, 0, :].detach().cpu().numpy().T, aspect="auto")
        ax[3].set_title("Target")
        output, hidden = model.eval_step(batch)
        ax[1].imshow(hidden[:, 0, :].detach().cpu().numpy().T, aspect="auto")
        ax[1].set_title("Hidden")
        ax[2].imshow(output[:, 0, :].detach().cpu().numpy().T, aspect="auto")
        ax[2].set_title("Output")
        fig.suptitle("Rule " + task)
        fig.colorbar(
            ax[2].imshow(output[:, 0, :].detach().cpu().numpy().T, aspect="auto")
        )

        print(f"saving figure for task {task} in," "src/" + id + "-" + task + ".png")
        fig.savefig("src/" + id + "-" + task + ".png")


def save_np_array_for_surface_plot(model, dataloader, hp, id):
    for batch in dataloader:
        task = hp["rule_trains"][dataloader.dataset.current_rule_index - 1]
        if task in ["delaygo", "reactgo"]:
            output, hidden = model.eval_step(batch)
            av = np.mean(hidden.detach().cpu().numpy(), axis=1)
            print(av.shape)
            # for i in range(hidden.size(1)):
            np.save(f"activity-{task}-average", av)

        # np.save(f"activity-{task}", hidden[:, 0, :].detach().cpu().numpy())


def plot_average_activity(model, dataloader, hp, id):
    for batch in dataloader:
        task = hp["rule_trains"][dataloader.dataset.current_rule_index - 1]
        if task in ["contextdelaydm1", "contextdelaydm2"]:
            ouput, hidden = model.eval_step(batch)
            fig, ax = plt.subplots(1, figsize=(7, 5))
            av = np.mean(hidden.detach().cpu().numpy(), axis=1)
            ax.imshow(av.T)
            ax.set_xlabel("Time")
            ax.set_ylabel("Hidden unit")
            ax.set_title("Average activity over a batch of trials for" + task)
            fig.savefig(f"src/plots/av-activity-{task}.png")


def plot_activites(model, dataloader, hp, id):
    for batch in dataloader:
        task = hp["rule_trains"][dataloader.dataset.current_rule_index - 1]
        print(f"task: {task}")
        if task in ["delaygo", "reactgo"]:
            # ax.imshow(batch.x[:, 0, :].detach().cpu().numpy().T, aspect="auto")
            # ax.set_title("Input")
            # ax[3].imshow(batch.y[:, 0, :].detach().cpu().numpy().T, aspect="auto")
            # ax[3].set_title("Target")
            output, hidden = model.eval_step(batch)
            plt.plot(np.mean(hidden.detach().cpu().numpy(), axis=1))
            plt.show()
            raise NotImplementedError
            for i in range(hidden.size(1)):
                fig, ax = plt.subplots(1, figsize=(8, 5))
                ax.imshow(hidden[:, i, :].detach().cpu().numpy().T, aspect="auto")
                ax.set_xlabel("Time")
                ax.set_ylabel("Hidden unit")
                # ax.set_title("Hidden")
                # ax[2].imshow(output[:, 0, :].detach().cpu().numpy().T, aspect="auto")
                # ax[2].set_title("Output")
                fig.suptitle(f"hidden activity for {task}")

                print(
                    f'saving figure for task {task} in "src/plots/activity-{task}.png"'
                )
                fig.savefig(f"src/plots/activity-{task}-{i}.png")


if __name__ == "__main__":

    evaluate_trained_model()
