"""Utility functions."""

import os
import errno
import six
import json
import pickle
import numpy as np
import torch
from collections import defaultdict
import math
import scipy
from omegaconf import OmegaConf
from pprint import pprint
import matplotlib.pyplot as plt


from src.task import get_num_ring, get_num_rule, rules_dict


def gen_feed_dict(model, trial, hp):
    """Generate feed_dict for session run."""
    if hp["in_type"] == "normal":
        feed_dict = {model.x: trial.x, model.y: trial.y, model.c_mask: trial.c_mask}
    elif hp["in_type"] == "multi":
        n_time, batch_size = trial.x.shape[:2]
        new_shape = [n_time, batch_size, hp["rule_start"] * hp["n_rule"]]

        x = np.zeros(new_shape, dtype=np.float32)
        for i in range(batch_size):
            ind_rule = np.argmax(trial.x[0, i, hp["rule_start"] :])
            i_start = ind_rule * hp["rule_start"]
            x[:, i, i_start : i_start + hp["rule_start"]] = trial.x[
                :, i, : hp["rule_start"]
            ]

        feed_dict = {model.x: x, model.y: trial.y, model.c_mask: trial.c_mask}
    else:
        raise ValueError()

    return feed_dict


def _contain_model_file(model_dir):
    """Check if the directory contains model files."""
    for f in os.listdir(model_dir):
        if "model.ckpt" in f:
            return True
    return False


def _valid_model_dirs(root_dir):
    """Get valid model directories given a root directory."""
    return [x[0] for x in os.walk(root_dir) if _contain_model_file(x[0])]


def valid_model_dirs(root_dir):
    """Get valid model directories given a root directory(s).

    Args:
        root_dir: str or list of strings
    """
    if isinstance(root_dir, six.string_types):
        return _valid_model_dirs(root_dir)
    else:
        model_dirs = list()
        for d in root_dir:
            model_dirs.extend(_valid_model_dirs(d))
        return model_dirs


def load_log(model_dir):
    """Load the log file of model save_name"""
    fname = os.path.join(model_dir, "log.json")
    if not os.path.isfile(fname):
        return None

    with open(fname, "r") as f:
        log = json.load(f)
    return log


def save_log(log):
    """Save the log file of model."""
    model_dir = log["model_dir"]
    fname = os.path.join(model_dir, "log.json")
    with open(fname, "w") as f:
        json.dump(log, f)


def load_hp(model_dir):
    """Load the hyper-parameter file of model save_name"""
    fname = os.path.join(model_dir, "hp.json")
    if not os.path.isfile(fname):
        fname = os.path.join(model_dir, "hparams.json")  # backward compat
        if not os.path.isfile(fname):
            return None

    with open(fname, "r") as f:
        hp = json.load(f)

    # Use a different seed aftering loading,
    # since loading is typically for analysis
    hp["rng"] = np.random.RandomState(hp["seed"] + 1000)
    return hp


def save_hp(hp, model_dir):
    """Save the hyper-parameter file of model save_name"""
    hp_copy = hp.copy()
    hp_copy.pop("rng")  # rng can not be serialized
    with open(os.path.join(model_dir, "hp.json"), "w") as f:
        json.dump(hp_copy, f)


def load_pickle(file):
    try:
        with open(file, "rb") as f:
            data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(file, "rb") as f:
            data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print("Unable to load data ", file, ":", e)
        raise
    return data


def find_all_models(root_dir, hp_target):
    """Find all models that satisfy hyperparameters.

    Args:
        root_dir: root directory
        hp_target: dictionary of hyperparameters

    Returns:
        model_dirs: list of model directories
    """
    dirs = valid_model_dirs(root_dir)

    model_dirs = list()
    for d in dirs:
        hp = load_hp(d)
        if all(hp[key] == val for key, val in hp_target.items()):
            model_dirs.append(d)

    return model_dirs


def find_model(root_dir, hp_target, perf_min=None):
    """Find one model that satisfies hyperparameters.

    Args:
        root_dir: root directory
        hp_target: dictionary of hyperparameters
        perf_min: float or None. If not None, minimum performance to be chosen

    Returns:
        d: model directory
    """
    model_dirs = find_all_models(root_dir, hp_target)
    if perf_min is not None:
        model_dirs = select_by_perf(model_dirs, perf_min)

    if not model_dirs:
        # If list empty
        print("Model not found")
        return None, None

    d = model_dirs[0]
    hp = load_hp(d)

    log = load_log(d)
    # check if performance exceeds target
    if log["perf_min"][-1] < hp["target_perf"]:
        print(
            """Warning: this network perform {:0.2f}, not reaching target
              performance {:0.2f}.""".format(
                log["perf_min"][-1], hp["target_perf"]
            )
        )

    return d


def select_by_perf(model_dirs, perf_min):
    """Select a list of models by a performance threshold."""
    new_model_dirs = list()
    for model_dir in model_dirs:
        log = load_log(model_dir)
        # check if performance exceeds target
        if log["perf_min"][-1] > perf_min:
            new_model_dirs.append(model_dir)
    return new_model_dirs


def mkdir_p(path):
    """
    Portable mkdir -p

    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def gen_ortho_matrix(dim, rng=None):
    """Generate random orthogonal matrix
    Taken from scipy.stats.ortho_group
    Copied here from compatibilty with older versions of scipy
    """
    H = np.eye(dim)
    for n in range(1, dim):
        if rng is None:
            x = np.random.normal(size=(dim - n + 1,))
        else:
            x = rng.normal(size=(dim - n + 1,))
        # random sign, 50/50, but chosen carefully to avoid roundoff error
        D = np.sign(x[0])
        x[0] += D * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = -D * (np.eye(dim - n + 1) - 2.0 * np.outer(x, x) / (x * x).sum())
        mat = np.eye(dim)
        mat[n - 1 :, n - 1 :] = Hx
        H = np.dot(H, mat)
    return H


def get_task_hp(cfg):
    num_ring = get_num_ring(cfg.task.ruleset)
    n_rule = get_num_rule(cfg.task.ruleset)

    n_input = 1 + num_ring * cfg.task.n_eachring + n_rule
    n_output = cfg.task.n_eachring + 1
    rule_start = 1 + num_ring * cfg.task.n_eachring

    hp = defaultdict()
    hp.update(cfg.task)
    hp["num_ring"] = num_ring
    hp["n_rule"] = n_rule
    hp["n_input"] = n_input
    hp["n_output"] = n_output
    hp["rule_start"] = rule_start

    hp["loss_type"] = cfg.model.loss_type
    hp["dt"] = cfg.model.dt
    hp["batch_size_train"] = cfg.train.batch_size_train
    hp["batch_size_val"] = cfg.train.batch_size_val
    hp["batch_size_test"] = cfg.train.batch_size_test
    alpha = cfg.model.dt / cfg.model.tau
    hp["alpha"] = alpha
    # OmegaConf.update(cfg, "task", {"num_ring": num_ring})
    # OmegaConf.update(cfg, "task", {"n_rule": n_rule})
    # OmegaConf.update(cfg, "task", {"n_input": n_input})
    # OmegaConf.update(cfg, "task", {"n_output": n_output})
    # OmegaConf.update(cfg, "task", {"rule_start": rule_start})

    # cfg.task.num_ring = num_ring
    # cfg.task.n_rule = n_rule
    # cfg.task.n_input = n_input
    # cfg.task.n_output = n_output
    # cfg.task.rule_start = 1 + num_ring * cfg.task.n_eachring
    # cfg.task.ruleset = cfg.task.ruleset

    if "rule_trains" not in hp:
        rules_train = rules_dict[hp["ruleset"]]
        hp["rule_trains"] = rules_train
        # cfg.task.rule_trains = rules_dict[cfg.task.ruleset]

    hp["rules"] = hp["rule_trains"]  # why duplicate?
    # OmegaConf.update(cfg, "task", {"rules": cfg.task.rule_trains}, merge=False)
    # cfg.task.rules = cfg.task.rule_trains

    # Assign probabilities for rule_trains.
    if "rule_prob_map" not in hp:
        rule_prob_map = dict()

    # Turn into rule_trains format
    # cfg.task.rule_probs = None
    if hasattr(hp["rule_trains"], "__iter__"):
        # Set default as 1.
        rule_prob = np.array([rule_prob_map.get(r, 1.0) for r in hp["rule_trains"]])
        hp["rule_probs"] = list(rule_prob / np.sum(rule_prob))
        # OmegaConf.update(
        #     cfg, "task", {"rule_probs": list(rule_prob / np.sum(rule_prob))}
        # )

        # cfg.task.rule_probs = list(rule_prob / np.sum(rule_prob))

    # return cfg
    return hp


def get_default_hp(ruleset):
    """Get a default hp.

    Useful for debugging.

    Returns:
        hp : a dictionary containing training hpuration
    """
    num_ring = get_num_ring(ruleset)
    n_rule = get_num_rule(ruleset)

    n_eachring = 32
    n_input, n_output = 1 + num_ring * n_eachring + n_rule, n_eachring + 1
    hp = {
        # # batch size for training
        # "batch_size_train": 64,
        # # batch_size for testing
        # "batch_size_test": 512,
        # # input type: normal, multi
        # "in_type": "normal",
        # # Type of RNNs: LeakyRNN, LeakyGRU, EILeakyGRU, GRU, LSTM
        # "rnn_type": "LeakyRNN",
        # # whether rule and stimulus inputs are represented separately
        # "use_separate_input": False,
        # # Type of loss functions
        # "loss_type": "lsq",
        # # Optimizer
        # "optimizer": "adam",
        # # Type of activation runctions, relu, softplus, tanh, elu
        # "activation": "relu",
        # # Time constant (ms)
        # "tau": 100,
        # # discretization time step (ms)
        # "dt": 20,
        # # discretization time step/time constant
        # "alpha": 0.2,
        # # recurrent noise
        # "sigma_rec": 0.05,
        # # input noise
        # "sigma_x": 0.01,
        # # leaky_rec weight initialization, diag, randortho, randgauss
        # "w_rec_init": "randortho",
        # # a default weak regularization prevents instability
        # "l1_h": 0,
        # # l2 regularization on activity
        # "l2_h": 0,
        # # l2 regularization on weight
        # "l1_weight": 0,
        # # l2 regularization on weight
        # "l2_weight": 0,
        # # l2 regularization on deviation from initialization
        # "l2_weight_init": 0,
        # # proportion of weights to train, None or float between (0, 1)
        # "p_weight_train": None,
        # # Stopping performance
        # "target_perf": 1.0,
        # number of units each ring
        "n_eachring": n_eachring,
        # number of rings
        "num_ring": num_ring,
        # number of rules
        "n_rule": n_rule,
        # first input index for rule units
        "rule_start": 1 + num_ring * n_eachring,
        # number of input units
        "n_input": n_input,
        # number of output units
        "n_output": n_output,
        # number of visual units
        "n_visual": 10,
        # number of motor units
        "n_motor": 1,
        # number of recurrent units
        # "n_rnn": 256,
        # number of input units
        "ruleset": ruleset,
        # name to save
        # "save_name": "test",
        # learning rate
        # "learning_rate": 0.001,
        # intelligent synapses parameters, tuple (c, ksi)
        # "c_intsyn": 0,
        # "ksi_intsyn": 0,
    }

    return hp


# def display_rich_output(model, sess, step, log, model_dir):
#     """Display step by step outputs during training."""
#     variance._compute_variance_bymodel(model, sess)
#     rule_pair = ["contextdm1", "contextdm2"]
#     save_name = "_atstep" + str(step)
#     title = "Step " + str(step) + " Perf. {:0.2f}".format(log["perf_avg"][-1])
#     variance.plot_hist_varprop(
#         model_dir, rule_pair, figname_extra=save_name, title=title
#     )
#     plt.close("all")
