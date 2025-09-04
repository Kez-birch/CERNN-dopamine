import numpy as np
import scipy.spatial.distance
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat


from ..cortical_embedding import get_replication_dict
from ..paths import SPINE_COUNT_MACAQUE, SPINE_COUNT_HUMAN


def replicate_areas_in_vec(vals, duplicates):
    """duplciate: dict {idx: times}"""
    additional_neurons = 0
    duplicates = dict(sorted(duplicates.items()))
    for position, times in duplicates.items():
        vals = np.insert(
            vals,
            position + additional_neurons,
            [vals[position + additional_neurons] / times] * (times - 1),
        )
        additional_neurons += times - 1
        vals[position + additional_neurons] /= times
    return vals


# def divide_areas_in_vec(vals, duplicates):
#     duplicates = dict(sorted(duplicates.items()))
#     print("sorted duplciates, ", duplicates)
#     additional_neurons = 0
#     for position, times in duplicates.items():
#         for i in range(times):
#             vals[position + i + additional_neurons] = (
#                 vals[position + i + additional_neurons] / times
#             )
#             additional_neurons += 1
#     return vals


class PerformanceBasedSceduler:
    def __init__(self):
        self.scale = 0

    def update(self, perf):
        if perf > 0.5:
            self.scale = perf
        else:
            self.scale = 0


class RampingScheduler:
    def __init__(self, ramping, lambd, ramping_steps):
        self.ramping = ramping
        self.lambd = lambd
        self.ramping_steps = ramping_steps

    def update(self, perf):
        if self.ramping:
            self.lambd = min(self.lambd + perf / self.ramping_steps, 1.0)
        return self.lambd


class BaseRegulariser(torch.nn.Module):
    def __init__(self, lambd):
        super(BaseRegulariser, self).__init__()
        self.register_buffer("lambd", torch.tensor([lambd], dtype=torch.float32))

    def on_train_epoch_start(self, perf):
        pass

    def forward(self, model, hidden):
        """All child classes must implement forward(model, hidden)."""
        raise NotImplementedError


class L1WeightRegulariser(BaseRegulariser):
    """L1 weight regulariser.
    Applies L1 regularisation to all weights in model.
    Calculation:
        l1_weight * sum[abs(weight_matrix)]
    Attributes:
        l1_weight: Float; Weighting of L1 regularisation term.
    """

    def __init__(self, lambd=0.01):
        super().__init__(lambd)

    def forward(self, model, hidden):
        device = hidden.device
        l1_loss = torch.tensor(0, dtype=torch.float32, device=device)
        for param in model.parameters():
            l1_loss += torch.sum(torch.abs(param))
            # l1_loss += param.abs().mean() * l1_weight # how they had it
        return self.lambd * l1_loss, "l1_weight"


class L2WeightRegulariser(BaseRegulariser):
    """L2 weight regulariser.
    Applies L2 regularisation to all weights in model.
    Calculation:
        l2_weight * sum[weight_matrix^2]
    Attributes:
        l2_weight: Float; Weighting of L2 regularisation term.
    """

    def __init__(self, lambd=0.01):
        super().__init__(lambd)
        # self.lambd = torch.tensor([lambd], dtype=torch.float32)

    def forward(self, model, hidden):
        device = hidden.device
        l2_loss = torch.tensor(0, dtype=torch.float32, device=device)
        for param in model.parameters():
            if param.requires_grad:
                l2_loss += torch.sum(torch.pow(param, 2))
        return self.lambd * l2_loss, "l2_weight"


class L1HiddenRegulariser(BaseRegulariser):
    """L1 on the hidden activity."""

    def __init__(self, lambd=0.01):
        super().__init__(lambd)

    def forward(self, model, hidden):
        return hidden.abs().mean() * self.lambd, "l1_activity"


class L2HiddenRegulariser(BaseRegulariser):
    """L2 on the hidden activity."""

    def __init__(self, lambd=0.01):
        super().__init__(lambd)

    def forward(self, model, hidden):
        return torch.linalg.norm(hidden) * self.lambd, "l2_activity"


class SensoryActivityRegulariserL1(BaseRegulariser):
    def __init__(self, lambd, sensory_indices):
        super().__init__(lambd)
        self.register_buffer("sensory_indices", sensory_indices)

    def forward(self, model, hidden):
        sensory_activity = hidden * self.sensory_indices
        return sensory_activity.abs().mean() * self.lambd, "l1_sensory_activity"


class SensoryActivityRegulariserL2(BaseRegulariser):
    def __init__(self, lambd, sensory_indices):
        super().__init__(lambd)
        self.register_buffer(
            "sensory_indices", torch.tensor(sensory_indices, dtype=torch.int16)
        )
        # self.lambd = torch.tensor([lambd], dtype=torch.float32)

    def forward(self, model, hidden):
        sensory_activity = hidden * self.sensory_indices
        return torch.linalg.norm(sensory_activity) * self.lambd, "l2_sensory_activity"


class SE1(BaseRegulariser):
    """A regulariser for spatially embedded RNNs.
    Applies L1 regularisation to recurrent kernel of
    RNN which is weighted by the distance of units
    in predefined 3D space.
    Calculation:
        se1 * sum[distance_matrix o recurrent_kernel]
    Attributes:
        se1: Float; Weighting of SE1 regularisation term.
        distance_tensor: TF tensor / matrix with cost per
        connection in weight matrix of network.
    """

    def __init__(
        self, distance_matrix, lambd=0.01, ramping=False, dependency="linear", mode="L1"
    ):
        super().__init__(lambd)
        self.register_buffer(
            "distance_matrix", torch.tensor(distance_matrix, dtype=torch.float32)
        )
        self.register_buffer("scale", torch.tensor(1.0))
        self.ramping = ramping
        self.dependency = dependency
        self.mode = mode

    # def on_train_epoch_start(self, perf):
    #     return self.scheduler.update(perf)

    def forward(self, model, hidden):
        w_hh = model.rnn.rnncell.weight_hh
        abs_weight_matrix = torch.abs(w_hh)
        # Transform the distance matrix based on the dependency mode:
        if self.dependency == "quadratic":
            transformed_distance_matrix = self.distance_matrix**2
        elif self.dependency == "exponential":
            transformed_distance_matrix = torch.exp(self.distance_matrix)
        else:  # linear is default
            transformed_distance_matrix = self.distance_matrix
        if self.mode == "L1":
            se1_loss = torch.sum(abs_weight_matrix * transformed_distance_matrix)
        elif self.mode == "L2":
            se1_loss = torch.sum((abs_weight_matrix * transformed_distance_matrix) ** 2)
        else:
            raise NotImplementedError
        return self.lambd * self.scale * se1_loss, "dist*weight_reg"

    def _check_penalty_number(self, x):
        if not isinstance(x, (float, int)):
            raise ValueError(
                (
                    "Value: {} is not a valid regularization penalty number, "
                    "expected an int or float value"
                ).format(x)
            )

    def visualise_distance_matrix(self):
        plt.imshow(self.distance_matrix.numpy())
        plt.colorbar()
        plt.show()

    def visualise_neuron_structure(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(
            self.coordinates[0],
            self.coordinates[1],
            self.coordinates[2],
            c="b",
            marker=".",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()

    def get_config(self):
        return {"lambd": float(self.lambd)}


class IntraAreaPenalty(BaseRegulariser):
    """
    Penalizes recurrent connections within each cortical area via 'area_mask'.
    """

    def __init__(self, lambd=0.01, area_mask=None, penalty_type="l1"):
        super(IntraAreaPenalty, self).__init__(lambd)
        if area_mask is not None:
            self.register_buffer(
                "area_mask", torch.tensor(area_mask, dtype=torch.float32)
            )
        else:
            self.register_buffer("area_mask", None)
        self.penalty_type = penalty_type

    def forward(self, model, hidden):
        if self.penalty_type == "l1":
            penalty_val = torch.sum(
                torch.abs(model.rnn.rnncell.weight_hh) * self.area_mask
            )
        else:
            penalty_val = torch.sum((model.rnn.rnncell.weight_hh**2) * self.area_mask)
        return self.lambd * penalty_val, "intra_area_penalty"


class SelfConnReg(BaseRegulariser):
    """
    Penalizes self-connections on the diagonal of the recurrent weight matrix.
    """

    def __init__(self, lambd=0.01, diag_mask=None):
        super(SelfConnReg, self).__init__(lambd=lambd)
        if diag_mask is not None:
            self.register_buffer("diag_mask", diag_mask)
        else:
            self.register_buffer("diag_mask", None)

    def forward(self, model, hidden):
        W = model.rnn.rnncell.weight_hh
        penalty_val = torch.sum(torch.abs(W * self.diag_mask))
        return self.lambd * penalty_val, "self_conn"


class WiringEntropyRegulariser(BaseRegulariser):
    """
    Wiring entropy regulariser that depends on the model's recurrent weights.

    def forward(self, model, hidden):
        D = self.distance_matrix
        Z = torch.sum(D) + 1e-8
        p = D / Z
        entropy = -torch.sum(p * torch.log(p + 1e-8))
    - We bin these distances, summing |W_ij| for each bin, forming a distribution p_i.
    - Then we compute negative entropy = -sum( p_i * log p_i ) (or scaled by lambda).
    - Optionally, we can add a 'cost_penalty' if average wiring length exceeds some threshold.

    The resulting penalty will produce a gradient wrt the model's W_ij, so it won't be constant.
    """

    def __init__(
        self,
        distance_matrix=None,
        lambd=0.01,
        num_bins=30,
        d_bar_max=None,
        lambda_cost=0,
    ):
        """
        Args:
            distance_matrix: NxN array or tensor of distances
            lambd: multiplier for the negative-entropy portion
            num_bins: how many bins to use for distances
            d_bar_max: if not None, maximum average wiring length allowed
            lambda_cost: how strongly we penalize going beyond d_bar_max
        """
        super(WiringEntropyRegulariser, self).__init__(lambd=lambd)

        if distance_matrix is not None:
            distance_matrix = torch.tensor(distance_matrix, dtype=torch.float32)
        self.register_buffer("distance_matrix", distance_matrix)
        self.num_bins = num_bins
        self.d_bar_max = d_bar_max
        self.lambda_cost = lambda_cost

    def forward(self, model, hidden):
        w_hh = model.rnn.rnncell.weight_hh
        abs_weights = torch.abs(w_hh)
        device = abs_weights.device

        # Flatten
        abs_w_flat = abs_weights.view(-1)  # shape: [N*N]
        dist_flat = self.distance_matrix.view(-1)

        # Make bins in [min_distance, max_distance]
        min_d = dist_flat.min()
        max_d = dist_flat.max()
        bins = torch.linspace(min_d, max_d, steps=self.num_bins + 1, device=device)

        # Assign each connection to a bin
        bin_indices = torch.bucketize(dist_flat, bins, right=False)  # 1..num_bins

        # Sum up weights in each bin
        p_i = torch.zeros(self.num_bins, device=device)
        for i in range(1, self.num_bins + 1):
            mask = bin_indices == i
            p_i[i - 1] = abs_w_flat[mask].sum()

        # Convert to a probability distribution
        total_strength = p_i.sum() + 1e-8
        prob_p_i = p_i / total_strength

        # Compute negative entropy:  - sum( p_i log p_i )
        # Because we do:  'entropy_loss = - lambda * entropy'
        # So minimizing entropy_loss => maximizing the actual entropy
        negative_entropy = torch.sum(prob_p_i * torch.log(prob_p_i + 1e-8))
        entropy_loss = (
            self.lambd * negative_entropy
        )  # if we want to MINIMIZE negative entropy, use + sign here.

        # (Optional) cost penalty if average wiring length > d_bar_max
        cost_penalty = 0.0
        if (self.d_bar_max is not None) and (self.lambda_cost > 0):
            # bin_centers for each bin
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            avg_wiring_length = (prob_p_i * bin_centers).sum()
            if avg_wiring_length > self.d_bar_max:
                cost_penalty = self.lambda_cost * (avg_wiring_length - self.d_bar_max)

        total_loss = entropy_loss + cost_penalty

        return total_loss, "wiring_entropy"


class DMNActivityRegulariser(BaseRegulariser):
    """
    Penalize hidden-state activity in DMN or core-DMN regions.
    Configured by 'dmn_type' ("core" or "extended") and 'penalty_type' ("l1" or "l2").
    """

    def __init__(self, lambd=0.01, dmn_type="core", penalty_type="l2"):
        super().__init__(lambd)
        self.dmn_type = dmn_type
        self.penalty_type = penalty_type

    def forward(self, model, hidden):
        device = hidden.device
        if self.dmn_type == "core":
            mask_np = model.ce.core_dmn_mask
        else:
            mask_np = model.ce.dmn_mask
        mask = torch.tensor(mask_np, dtype=torch.float32, device=device)
        dmn_activity = (
            hidden * mask
        )  # broadcast multiplication; shape: [Time, Batch, n_rnn]
        if self.penalty_type == "l1":
            penalty_val = dmn_activity.abs().mean()
        else:
            penalty_val = (dmn_activity**2).mean()
        return self.lambd * penalty_val, "dmn_activity"


class InputsSumToAvSpineCountGrad(BaseRegulariser):
    """
    Penalizes the sum of inputs to the network.
    """

    def __init__(self, duplicates, areas, species, lambd=0.01):
        super().__init__(lambd)
        spine_count_gradient = self.get_spine_count_gradient(duplicates, areas, species)
        self.register_buffer("spine_count_gradient", spine_count_gradient)

    def get_macaque_spine_count_gradient(self, duplicates, ordered_areas):
        f = SPINE_COUNT_MACAQUE
        spine_counts = loadmat(f)["spine_count_lyon_regions_40"]
        spine_counts = np.array(
            [counts[0].item() for counts in spine_counts for _ in counts],
            dtype=np.float32,
        )
        spine_counts = torch.tensor(spine_counts, dtype=torch.float16)
        spine_counts = 0.5 + (spine_counts - spine_counts.min()) / (
            spine_counts.max() - spine_counts.min()
        ) * (1.5 - 0.5)
        spine_counts /= spine_counts.mean()
        return spine_counts

    def get_human_spine_count_gradient(self, duplicates, ordered_areas):
        f = SPINE_COUNT_HUMAN
        myelin_vec = loadmat(f)["myelin_parcels_mean"]
        myelin_vec = torch.tensor(myelin_vec, dtype=torch.float16).squeeze()
        myelin_vec = (myelin_vec - myelin_vec.min()) / (
            myelin_vec.max() - myelin_vec.min()
        )
        spine_count_approx = 1 - myelin_vec
        spine_count_approx += 0.5

        return spine_count_approx

    def get_spine_count_gradient(self, duplicates, ordered_areas, species):
        """gradient of spine count with mean 1"""
        if species == "macaque":
            spine_counts = self.get_macaque_spine_count_gradient(
                duplicates, ordered_areas
            )
        elif species == "human":
            spine_counts = self.get_human_spine_count_gradient(
                duplicates, ordered_areas
            ).squeeze()
        else:
            raise NotImplementedError

        # duplicate and devide subareas by n
        duplicates = get_replication_dict(duplicates, ordered_areas)
        spine_counts = replicate_areas_in_vec(spine_counts, duplicates)
        assert (
            spine_counts.mean() - 1.0
        ) < 0.15, f"spine count gradient not normalized, {spine_counts.mean()} not close to 1.0"

        # assert spine_counts.mean() == 1.0
        return spine_counts

    def forward(self, model, hidden):
        w_hh = torch.abs(model.rnn.rnncell.weight_hh)
        weights_sum = torch.sum(w_hh, axis=1)
        av = torch.mean(weights_sum) * self.spine_count_gradient
        reg_val = torch.sum((weights_sum - av))  # ** 2 for L2?
        return self.lambd * reg_val, "inputs_sum_to_spine_count_grad"


regs_mapping = {
    "l1_weight_regulariser": L1WeightRegulariser,
    "l2_weight_regulariser": L2WeightRegulariser,
    "l1_activity_regulariser": L1HiddenRegulariser,
    "l2_activity_regulariser": L2HiddenRegulariser,
    "l1_sensory_activity_regulariser": SensoryActivityRegulariserL1,
    "l2_sensory_activity_regulariser": SensoryActivityRegulariserL2,
    "SE": SE1,
    "intra_area_penalty": IntraAreaPenalty,
    "self_conn_reg": SelfConnReg,
    "wiring_entropy_regulariser": WiringEntropyRegulariser,
    "dmn_activity_regulariser": DMNActivityRegulariser,
    "inputs_sum_to_av": InputsSumToAvSpineCountGrad,
}
