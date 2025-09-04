import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
import pandas as pd

from src.models.leaky_rnn_base import RNNCell_base, RNNLayer

# Helpers

def _area_counts(area2idx: Dict, n_rnn: int) -> Dict:
    """Derive per-area unit counts from start indices.
    Avoids reliance on ce.duplicates.
    """
    items = sorted(area2idx.items(), key=lambda kv: kv[1])  # [(name, start), ...]
    names = [k for k, _ in items]
    starts = [v for _, v in items] + [n_rnn]  # sentinel for last slice
    return {names[i]: (starts[i + 1] - starts[i]) for i in range(len(names))}

class CERNNModel(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_rnn: int,
        n_output: int,
        dt: float,
        tau: float,
        noise: bool,
        w_rec_init: str,
        sigma_rec: float,
        activation: str,
        ce,
        shuffle_d1: bool = False,
    ):
        super().__init__()
        self.ce = ce
        self.motor = self.ce.motor[0]
        self.n_rnn = n_rnn

        decay = dt / tau

        # Build cell
        rnncell = leaky_RNNCell_CERNN(
            n_input,
            n_rnn,
            activation,
            decay,
            w_rec_init,
            ce,
            True,  # bias
            noise,
            sigma_rec,
        )

    
        # path to map 
        repo_root = Path(__file__).resolve().parents[2]
        per_area_csv = repo_root / "data/d1r_per_area.csv"
        if not per_area_csv.exists():
            candidates = [
                Path(os.path.expanduser("~/data/d1r_per_area.csv")),
                Path("data/d1r_per_area.csv").resolve(),
            ]
            for p in candidates:
                if p.exists():
                    per_area_csv = p
                    break
        if not per_area_csv.exists():
            raise FileNotFoundError(
                f"Could not find d1r_per_area.csv at {per_area_csv}. Place it under <repo>/data/."
            )

        df_area = pd.read_csv(per_area_csv)
        # Region_ID (int), D1R_Mean (float)
        if not {"Region_ID", "D1R_Mean"}.issubset(df_area.columns):
            raise ValueError(
                f"d1r_per_area.csv must have columns ['Region_ID','D1R_Mean']; got {df_area.columns.tolist()}"
            )
        d1_area = {int(r): float(v) for r, v in zip(df_area["Region_ID"], df_area["D1R_Mean"])}

        # Determine per-area counts from ce.area2idx
        counts = _area_counts(self.ce.area2idx, n_rnn)
        if self.motor not in counts:
            raise KeyError(f"Motor area '{self.motor}' not found in ce.area2idx.")
        self.n_motor = counts[self.motor]

        # Expand per-area values to per-unit vector
        area_items = sorted(self.ce.area2idx.items(), key=lambda kv: kv[1])  # [(name, start), ...]
        names = [k for k, _ in area_items]
        starts = [v for _, v in area_items] + [n_rnn]

        d1_vec = torch.zeros(n_rnn, dtype=torch.float32)
        missing_names: List[str] = []
        for i, name in enumerate(names):
            start, end = starts[i], starts[i + 1]
            # Try to parse name as int; else use ordinal fallback (1..40)
            try:
                key_int = int(name)
            except Exception:
                key_int = None

            if key_int is not None and key_int in d1_area:
                val = d1_area[key_int]
            else:
                # ordinal fallback: i-th area in area2idx order → Region_ID i+1
                rid = i + 1
                val = d1_area.get(rid, float(np.nan))
                if not np.isfinite(val):
                    missing_names.append(name)
                    val = float(np.nanmean(list(d1_area.values())))
            d1_vec[start:end] = val

        # normalize to [0,1] to help stabilise 
        mn, mx = float(d1_vec.min()), float(d1_vec.max())
        if mx > mn:
            d1_vec = (d1_vec - mn) / (mx - mn)

        rnncell.register_buffer("d1_map_unshuffled", d1_vec.clone())

        if shuffle_d1:
            g = torch.Generator()
            g.manual_seed(torch.initial_seed())  # use global seed for reproducability 
            idx = torch.randperm(n_rnn, generator=g)  


            rnncell.register_buffer("d1_perm", idx)


            d1_vec = d1_vec[idx]  
            print(f"[D1] Shuffled control applied (seed={torch.initial_seed()}, perm_sum={int(idx.sum().item())})")



        with torch.no_grad():
            rnncell.d1_map.copy_(d1_vec)

        print(
            f"[D1] per-area→per-unit: len={d1_vec.numel()}, "
            f"mean={float(d1_vec.mean()):.3f}, std={float(d1_vec.std()):.3f}"
        )
        if missing_names:
            print(
                f"[D1] areas without direct ID match (used ordinal fallback): "
                f"{missing_names[:5]}{' ...' if len(missing_names) > 5 else ''}"
            )

        # Configure dopamine use in the cell
        rnncell.dopamine_mod_target = "gain"  # 'gain' | 'recurrent' | 'decay'
        rnncell.dopamine_gain = 0.2
        rnncell.dopamine_level = 0.5

        # Wrap cell and readout
        self.rnn = RNNLayer(rnncell)
        self.readout = nn.Linear(self.n_motor, n_output, bias=False)

    def init_hidden_(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        max_steps = 100
        stable_count = 0
        h0 = h0.to(self.rnn.rnncell.weight_hh.device)
        for _ in range(max_steps):
            _, h_next = self.rnn(x[:2, 0:1, :], h0)
            if torch.allclose(h_next, h0, atol=0.1):
                stable_count += 1
                if stable_count >= 4:
                    return h0
            else:
                stable_count = 0
            h0 = h_next
        return h0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, n_input]
        hidden0 = torch.zeros([1, x.shape[1], self.n_rnn], device=x.device)
        hidden0 = self.init_hidden_(x, hidden0)
        hidden, _ = self.rnn(x, hidden0)

        start_idx = self.ce.area2idx[self.motor]
        end_idx = start_idx + self.readout.in_features  # == n_motor
        selected_hidden = hidden[:, :, start_idx:end_idx]

        output = self.readout(selected_hidden)
        return output, hidden


class leaky_RNNCell_CERNN(RNNCell_base):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        nonlinearity: str,
        decay: float,
        w_rec_init: str,
        ce,
        bias: bool = True,
        noise: bool = True,
        sigma_rec: float = 0.05,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            nonlinearity=nonlinearity,
            w_rec_init=w_rec_init,
            noise=noise,
            bias=bias,
            decay=decay,
            sigma_rec=sigma_rec,
        )
        self.ce = ce

        # Keep for compatibility with other modes
        self.dopamine_level = 0.5
        self.dopamine_gain = 0.05
        self.dopamine_mod_target = "recurrent"  # default; model flips to 'gain'
        self.printed_this_epoch = False

        # Optional within-area mask (keep as a real buffer)
        if ce.mask_within_area_weights:
            self.mask_within_area_weights = True
            self.register_buffer(
                "intra_area_mask",
                torch.as_tensor(1 - ce.area_mask, dtype=torch.float32),
            )
        else:
            self.mask_within_area_weights = False

        # Per-unit D1 map (default = ones → no effect until model copies real values)
        self.register_buffer("d1_map", torch.ones(self.hidden_size))

        # Cached names/starts for convenience
        counts = _area_counts(self.ce.area2idx, self.hidden_size)
        self.visual = ce.sensory[0]
        self.somatosensory = ce.sensory[1]
        self.n_visual = counts[self.visual]
        self.n_somatosensory = counts[self.somatosensory]

        v1_start = ce.area2idx[self.visual]
        v1_end = v1_start + self.n_visual
        print("visual start", v1_start, "visual end", v1_end)

        s1_start = ce.area2idx[self.somatosensory]
        s1_end = s1_start + self.n_somatosensory
        print("somatosensory start", s1_start, "somatosensory end", s1_end)

        # Input masks (buffers so they move with device and save in state_dict)
        self.register_buffer("mask_s1", torch.zeros(self.hidden_size, self.input_size))
        self.mask_s1[s1_start:s1_end, 1:3] = 1

        self.register_buffer("mask_v1", torch.zeros(self.hidden_size, self.input_size))
        self.mask_v1[v1_start:v1_end, 3:5] = 1
        self.mask_v1[v1_start:v1_end, 0] = 1

        self.register_buffer("mask_taskid", torch.zeros(self.hidden_size, self.input_size))
        self.mask_taskid[:, 5:-1] = 1

        self.zero_weights_thres = ce.zero_weights_thres

    def get_sensory_ind(self) -> torch.Tensor:
        proj_mask = self.mask_s1 + self.mask_v1                  # [hidden_size, input_size]
        return proj_mask.any(dim=1).to(dtype=torch.float32,      # [hidden_size]
                                       device=self.d1_map.device)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        # Project inputs using masks
        somatosensory = input @ (self.weight_ih * self.mask_s1).t()
        visual = input @ (self.weight_ih * self.mask_v1).t()
        task_id = input @ (self.weight_ih * self.mask_taskid).t()
        input = somatosensory + visual + task_id

        # Optional mask on recurrent weights
        weights_mask = None
        if self.mask_within_area_weights:
            weights_mask = self.intra_area_mask

        if self.zero_weights_thres > 0:
            nonzero_mask = (self.weight_hh.abs() > self.zero_weights_thres).to(self.weight_hh.dtype)
            weights_mask = nonzero_mask if weights_mask is None else (weights_mask * nonzero_mask)


        # Dopamine modulation modes
        if self.dopamine_mod_target == "recurrent":
            weight_hh = self.weight_hh * (1 + self.dopamine_gain * self.dopamine_level)
            out = self.leaky_rnn_step(input, hidden, weights_mask, weight_hh=weight_hh)

        elif self.dopamine_mod_target == "decay":
            decay = self.decay * (1 - self.dopamine_gain * self.dopamine_level)
            out = self.leaky_rnn_step(input, hidden, weights_mask, decay=decay)

        elif self.dopamine_mod_target == "gain":
            # vanilla step, then per-unit post-nonlinearity gain via D1 map
            out = self.leaky_rnn_step(input, hidden, weights_mask)
            d1c = (self.d1_map - 0.5) * 2.0
            mult = 1.0 + self.dopamine_gain * d1c
            out = out * mult 
            
            if self.training and not self.printed_this_epoch:
                if self.dopamine_mod_target in ("recurrent", "decay"):
                    print(f"[DA] mode={self.dopamine_mod_target}, level={self.dopamine_level:.3f}, gain={self.dopamine_gain:.3f}")
                elif self.dopamine_mod_target == "gain":
                    print(
                        f"[D1] mode=gain, gain={self.dopamine_gain:.3f}, "
                        f"map mean={float(self.d1_map.mean()):.3f}, "
                        f"std={float(self.d1_map.std()):.3f}, "
                        f"min={float(self.d1_map.min()):.3f}, "
                        f"max={float(self.d1_map.max()):.3f}"
                    )
                self.printed_this_epoch = True


        else:
            out = self.leaky_rnn_step(input, hidden, weights_mask)

        return out
