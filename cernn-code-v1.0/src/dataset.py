import torch
import torch.utils.data as data
import numpy as np

from src.task import generate_trials, rules_dict


def collate_fn(batch):
    """collate funciton as a hacky solution to turning
    tasks generation into a dataloader"""
    return batch[0]


class YangTasks(data.Dataset):
    def __init__(
        self,
        hp,
        mode="train",
    ):
        """creates a batch of trials for the yang tasks
        each batch contains one type of trial (e.g. rule_train_now)
        each rule is randomly sampled for each batch, chosen from 'rules train'"""
        self.hp = hp
        self.mode = mode
        self.batch_size = hp[f"batch_size_{mode}"]
        if self.mode == "val" or self.mode == "test":
            # allows looping through all rules during validation
            self.current_rule_index = 0
        else:
            # randomly choose new rule at each train step
            self.current_rule_index = None

    def _gen_feed_dict(self, trial, hp):
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

    def __getitem__(self, index):
        rule = (
            np.random.choice(self.hp["rule_trains"], p=self.hp["rule_probs"])
            if self.current_rule_index is None  # TODO: if self.mode == "test"?
            else self.hp["rule_trains"][self.current_rule_index]
        )
        if self.mode == "test" or self.mode == "val":
            gen_mode = "test"
        else:
            gen_mode = "random"
        trial = generate_trials(
            rule, self.hp, mode=gen_mode, batch_size=self.batch_size
        )  # mode='test' vor analyse
        trial = self._gen_feed_dict(trial, self.hp)
        if self.mode == "test" or self.mode == "val":
            self.current_rule_index += 1
        trial.rule = rule
        return trial

    def __len__(self):
        # TODO: how large should an epoch be?
        return (
            self.hp["trials_per_epoch"]
            if self.mode == "train"
            else len(self.hp["rule_trains"])
        )
