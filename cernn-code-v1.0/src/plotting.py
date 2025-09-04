import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import seaborn as sns


from src.task import generate_trials


def add_epoch_to_plot(ax, start, end, key):
    if start is None:
        start = 0
    if end is None:
        end = -1
    if isinstance(start, (np.ndarray, list)) and len(start) > 0:
        start = start[0]
    if isinstance(end, (np.ndarray, list)) and len(end) > 0:
        end = end[0]
    if "stim" in key:
        alpha = 0.2
        ax.axvspan(start - 1, end - 1, alpha=alpha, color="gray", ls="--", lw=2)


def plot_trial(hidden, trial):
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.imshow(hidden[:, 0, :].T, cmap="hot", interpolation="nearest", aspect="auto")
    ax.set_title(trial.rule)
    for key, values in trial.epochs.items():
        start, end = values
        add_epoch_to_plot(ax, start, end, key)
    add_epoch_to_plot(ax, None, None, trial.rule)
    plt.show()


def plot_sequence(sequence, task, n, seq_type, save=True):
    """sequence: torch tensor T x B x D"""
    dir = os.path.join("src/plots", task, seq_type)
    os.makedirs(dir, exist_ok=True)
    if len(sequence.shape) == 2:
        sequence = sequence.unsqueeze(-1)
    _, B, _ = sequence.shape
    # plt.figure(figsize=(15, 10))
    fig, axs = plt.subplots(1, B, figsize=(15, 10))
    for i in range(B):
        img = sequence[:, i, :].numpy().T
        axs[i].imshow(img)

        title = f"{seq_type} sequence for task {task}"
        if seq_type == "y_loc":
            title += f", y_loc = {sequence[-1, i, 0]:.2f}"
        axs[i].set_title(title)
        axs[i].set_xlabel("Time")
        axs[i].set_ylabel("'activation'")
        plt.tight_layout()
    if save:
        plt.savefig(os.path.join(dir, f"{seq_type}_{n}.png"))

    return fig


def inspect_data(config):
    config["rng"] = np.random.RandomState(0)
    step = 0
    while step * config["batch_size_train"] <= config["train_steps"]:
        rule_train_now = np.random.choice(config["rule_trains"], p=config["rule_probs"])
        trial = generate_trials(
            rule_train_now, config, "random", batch_size=config["batch_size_train"]
        )
        inputs, targets = torch.tensor(trial.x), torch.tensor(trial.y)
        plot_sequence(inputs, rule_train_now, step, "inputs")
        plot_sequence(targets, rule_train_now, step, "targets")
        plot_sequence(torch.tensor(trial.y_loc), rule_train_now, step, "y_loc")
        plot_sequence(torch.tensor(trial.c_mask), rule_train_now, step, "c_mask")
        if step > 10:
            break

        step += 1


def plot_loss(log):
    if log and "trials" in log and len(log["trials"]) > 1:
        tasks = ["contextdelaydm1", "contextdelaydm2", "contextdm1", "contextdm2"]

        plt.figure(figsize=(15, 10))
        for i, task in enumerate(tasks):
            if f"cost_{task}" in log and f"perf_{task}" in log:
                # Plotting cost
                plt.subplot(4, 2, 2 * i + 1)
                plt.plot(log["trials"], log[f"cost_{task}"], label=f"Cost of {task}")
                plt.xlabel("Trials")
                plt.ylabel("Cost")
                plt.title(f"Cost over Trials for {task}")
                plt.legend()

                # Plotting performance
                plt.subplot(4, 2, 2 * i + 2)
                plt.plot(
                    log["trials"], log[f"perf_{task}"], label=f"Performance of {task}"
                )
                plt.xlabel("Trials")
                plt.ylabel("Performance")
                plt.title(f"Performance over Trials for {task}")
                plt.legend()
            else:
                print(f"Not enough data to plot for task {task}")

        plt.tight_layout()
        plt.show()


def plot_learning_curves(log):
    trials = log["trials"]
    rules = [key.split("_")[1] for key in log.keys() if key.startswith("perf_")]

    fig, ax = plt.subplots(2, 1, figsize=(14, 10))  # Width, Height in inches

    # Plotting cost curves
    for rule in rules:
        cost_key = "cost_" + rule
        if cost_key in log:
            ax[0].plot(trials, log[cost_key], label=cost_key)
    ax[0].set_title("Cost Curves")
    ax[0].set_xlabel("Trials")
    ax[0].set_ylabel("Cost")
    ax[0].legend(fontsize="small", loc="upper left", bbox_to_anchor=(1, 1))

    # Plotting performance curves
    for rule in rules:
        perf_key = "perf_" + rule
        if perf_key in log:
            ax[1].plot(trials, log[perf_key], label=perf_key)
    ax[1].set_title("Performance Curves")
    ax[1].set_xlabel("Trials")
    ax[1].set_ylabel("Performance")
    ax[1].legend(fontsize="small", loc="upper left", bbox_to_anchor=(1, 1))

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    plt.show()


def plot_dists(val_dict, color="C0", xlabel=None, stat="count", use_kde=True):
    columns = len(val_dict)
    fig, ax = plt.subplots(1, columns, figsize=(columns * 3, 2.5))
    fig_index = 0
    for key in sorted(val_dict.keys()):
        key_ax = ax[fig_index % columns]
        sns.histplot(
            val_dict[key],
            ax=key_ax,
            color=color,
            bins=50,
            stat=stat,
            kde=use_kde and ((val_dict[key].max() - val_dict[key].min()) > 1e-8),
        )  # Only plot kde if there is variance
        key_ax.set_title(
            f"{key} "
            + (
                r"(%i $\to$ %i)" % (val_dict[key].shape[1], val_dict[key].shape[0])
                if len(val_dict[key].shape) > 1
                else ""
            )
        )
        if xlabel is not None:
            key_ax.set_xlabel(xlabel)
        fig_index += 1
    fig.subplots_adjust(wspace=0.4)
    return fig


def visualize_weight_distribution(model, color="C0"):
    weights = {}
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            continue
        key_name = f"Layer {name.split('.')[1]}"
        weights[key_name] = param.detach().view(-1).cpu().numpy()

    ## Plotting
    fig = plot_dists(weights, color=color, xlabel="Weight vals")
    fig.suptitle("Weight distribution", fontsize=14, y=1.05)
    plt.show()
    plt.close()


def visualize_activations(
    model, train_set, device="cpu", color="C0", print_variance=False
):
    model.eval()
    small_loader = data.DataLoader(train_set, batch_size=1024, shuffle=False)
    imgs, labels = next(iter(small_loader))
    imgs, labels = imgs.to(device), labels.to(device)

    # Pass one batch through the network, and calculate the gradients for the weights
    feats = imgs.view(imgs.shape[0], -1)
    activations = {}
    with torch.no_grad():
        for layer_index, layer in enumerate(model.layers):
            feats = layer(feats)
            if isinstance(layer, nn.Linear):
                activations[f"Layer {layer_index}"] = (
                    feats.view(-1).detach().cpu().numpy()
                )

    ## Plotting
    fig = plot_dists(activations, color=color, stat="density", xlabel="Activation vals")
    fig.suptitle("Activation distribution", fontsize=14, y=1.05)
    plt.show()
    plt.close()

    if print_variance:
        for key in sorted(activations.keys()):
            print(f"{key} - Variance: {np.var(activations[key])}")


def visualize_gradients(
    model, train_set, device="cpu", color="C0", print_variance=False
):
    """
    Inputs:
        net - Object of class BaseNetwork
        color - Color in which we want to visualize the histogram (for easier separation of activation functions)
    """
    model.eval()
    small_loader = data.DataLoader(train_set, batch_size=1024, shuffle=False)
    imgs, labels = next(iter(small_loader))
    imgs, labels = imgs.to(device), labels.to(device)

    # Pass one batch through the network, and calculate the gradients for the weights
    model.zero_grad()
    preds = model(imgs)
    loss = F.cross_entropy(
        preds, labels
    )  # Same as nn.CrossEntropyLoss, but as a function instead of module
    loss.backward()
    # We limit our visualization to the weight parameters and exclude the bias to reduce the number of plots
    grads = {
        name: params.grad.view(-1).cpu().clone().numpy()
        for name, params in model.named_parameters()
        if "weight" in name
    }
    model.zero_grad()

    ## Plotting
    fig = plot_dists(grads, color=color, xlabel="Grad magnitude")
    fig.suptitle("Gradient distribution", fontsize=14, y=1.05)
    plt.show()
    plt.close()

    if print_variance:
        for key in sorted(grads.keys()):
            print(f"{key} - Variance: {np.var(grads[key])}")
