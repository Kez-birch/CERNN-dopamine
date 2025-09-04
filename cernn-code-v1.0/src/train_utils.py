import torch
import numpy as np
import yaml
import os
import omegaconf


def lsq_loss(y_hat, y, c_mask=None):  # TODO: double check and put losses somewhere
    if c_mask is not None:
        return torch.mean(((y_hat - y) * c_mask) ** 2)
    else:
        return torch.mean((y_hat - y) ** 2)


def load_config(config_dir):
    """Load config file.

    Args:
        config_dir: directory of config file

    Returns:
        config: dictionary of config
    """
    print(config_dir)
    with open(config_dir, "r") as f:
        config = yaml.safe_load(f)
    return config


def popvec(y):
    """Population vector read out.

    Assuming the last dimension is the dimension to be collapsed

    Args:
        y: population output on a ring network. Numpy array (Batch, Units)

    Returns:
        Readout locations: Numpy array (Batch,)
    """

    loc = np.arctan2(y[:, 0], y[:, 1])
    return np.mod(loc, 2 * np.pi)  # check this? January 22 2019


# def popvec(y):
#     """Population vector read out.

#     Assuming the last dimension is the dimension to be collapsed

#     Args:
#         y: population output on a ring network. Numpy array (Batch, Units)

#     Returns:
#         Readout locations: Numpy array (Batch,)
#     """
#     pref = np.arange(0, 2 * np.pi, 2 * np.pi / y.shape[-1])  # preferences
#     temp_sum = y.sum(axis=-1)
#     temp_cos = np.sum(y * np.cos(pref), axis=-1) / temp_sum
#     temp_sin = np.sum(y * np.sin(pref), axis=-1) / temp_sum
#     loc = np.arctan2(temp_sin, temp_cos)
#     return np.mod(loc, 2 * np.pi)


def get_perf(y_hat, y_loc):
    """Get performance.

    Args:
      y_hat: Actual output. Numpy array (Time, Batch, Unit)
      y_loc: Target output location (-1 for fixation).
        Numpy array (Time, Batch)

    Returns:
      perf: Numpy array (Batch,)
    """
    y_hat = np.array(y_hat)
    y_loc = np.array(y_loc)
    if len(y_hat.shape) != 3:
        raise ValueError("y_hat must have shape (Time, Batch, Unit)")
    # Only look at last time points
    y_loc = y_loc[-1]
    y_hat = y_hat[-1]

    # Fixation and location of y_hat
    y_hat_fix = y_hat[..., 0]
    y_hat_loc = popvec(y_hat[..., 1:])

    # Fixating? Correctly saccading?
    fixating = y_hat_fix > 0.5

    original_dist = y_loc - y_hat_loc
    dist = np.minimum(abs(original_dist), 2 * np.pi - abs(original_dist))
    corr_loc = dist < 0.2 * np.pi

    # Should fixate?
    should_fix = y_loc < 0

    # performance
    perf = should_fix * fixating + (1 - should_fix) * corr_loc * (1 - fixating)
    return perf


def configure_name(cfg, task_hp):
    # task [all, rule, lenght of tasks] TODO(eva) think of something better?
    if len(cfg.task.rule_trains) == 26:
        rules_train = "all"
    else:
        rules_train = "_".join([rule for rule in cfg.task.rule_trains])

    # network name
    # name = "vm" if ("se" in cfg.model and cfg.model["se"] == True) else f"baseline_{id}"

    # input and output size
    if "all" in cfg.model.duplicate:
        duplicates = "all_" + str(cfg.model.duplicate["all"])
    else:
        duplicates = "_".join(
            [f"{area[2:]}_{n}" for area, n in cfg.model.duplicate.items()]
        )
        # name += duplicates
    # if "L_V1" and "L_FEF" in cfg.model:
    #     name += f"-{cfg.model['L_V1']}-{cfg.model['L_FEF']}"

    # spatial regularisation
    # if ["se_rule"] in cfg.model and cfg.model["se_rule"] == True:
    #     name += "-ce"
    return "_" + rules_train + "_" + duplicates


def init_model(hp_pl_module, task_hp, PLModule):

    with omegaconf.open_dict(hp_pl_module):
        hp_pl_module["rule_trains"] = task_hp["rule_trains"]
        hp_pl_module["n_input"] = task_hp["n_input"]
        hp_pl_module["n_output"] = task_hp["n_output"]

    return PLModule(hp_pl_module)
