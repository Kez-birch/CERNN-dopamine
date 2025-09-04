from typing import List
import numpy as np
import pandas as pd

from src.paths import (
    CORTICAL_AREAS_PATH,
    DISTANCE_MATRIX_PATH,
    COG_NETWORK_OVERLAP,
    DISTANCE_MATRIX_PATH_MACAQUE,
    CORTICAL_AREAS_PATH_MACAQUE,
)


def get_areas(cortical_areas_path) -> List[str]:
    with open(cortical_areas_path, "rb") as f:
        areas = f.readlines()
        areas = [area.decode("utf-8").strip() for area in areas]
    return areas


# def get_replication_dict(duplicate, cortical_areas_path) -> dict:
#     """dict {area_idx: times}"""
#     idx_n = {}
#     with open(cortical_areas_path, "rb") as f:
#         areas = f.readlines()
#         areas = [area.decode("utf-8").strip() for area in areas]
#         for area, n in duplicate.items():
#             idx_n[areas.index(area)] = n
#     print(idx_n)
#     return idx_n


def load_distance_matrix(distance_matrix_path):
    if distance_matrix_path[-4:] == ".npy":
        distance_matrix = np.load(distance_matrix_path)
    elif distance_matrix_path[-4:] == ".txt":
        distance_matrix = np.loadtxt(distance_matrix_path)
    return distance_matrix


def replicate_row_col(matrix, times, position):
    replicated_column = np.tile(matrix[:, position : position + 1], times)
    new_matrix = np.hstack(
        (matrix[:, :position], replicated_column, matrix[:, position + 1 :])
    )
    replicated_row = np.tile(new_matrix[position : position + 1, :], (times, 1))
    final_matrix = np.vstack(
        (new_matrix[:position, :], replicated_row, new_matrix[position + 1 :, :])
    )
    return final_matrix


def replicate_sensorymotor(matrix, replication_dict):
    """replication dict:
    {
        'position': times,
    }
    """
    if (
        not isinstance(matrix, np.ndarray)
        or matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
    ):
        raise ValueError("Input must be a square numpy array")
    additional_neurons = 0
    for position, times in replication_dict.items():
        matrix = replicate_row_col(matrix, times, position + additional_neurons)
        additional_neurons += times - 1

    return matrix


def get_replication_dict(duplicate, areas) -> dict:
    """dict {area_idx: times}"""
    idx_n = {}
    for area, n in duplicate.items():
        idx_n[areas.index(area)] = n
    print(idx_n)
    return idx_n


def update_distance_matrix(duplicate, original_distance_matrix, cortical_areas):
    duplicate = get_replication_dict(duplicate, cortical_areas)
    updated_distance_matrix = replicate_sensorymotor(
        original_distance_matrix, duplicate
    )
    return updated_distance_matrix


def get_area2idx(cortical_areas, duplicates):
    i = 0
    area2idx = {}
    for area in cortical_areas:
        area2idx[area] = i
        if area in duplicates:
            i += duplicates[area]
        else:
            i += 1
    return area2idx


def get_distance_from(area_idx, distance_matrix, areas):
    distances = []
    for i, _ in enumerate(areas):
        distances.append(distance_matrix[area_idx, i])
    return distances


def resolve_duplicates(duplicate, cortical_areas_path):
    if "all" in duplicate:
        with open(cortical_areas_path, "r") as file:
            area_names = file.readlines()
            area_names = [name.strip() for name in area_names]
        duplicate_new = {a: duplicate["all"] for a in area_names}
        return duplicate_new
    elif "rest" in duplicate:
        with open(cortical_areas_path, "r") as file:
            area_names = file.readlines()
            area_names = [name.strip() for name in area_names]
        duplicate_new = {}
        for a in area_names:
            if a not in duplicate:
                duplicate_new[a] = duplicate["rest"]
            else:
                duplicate_new[a] = duplicate[a]
        return duplicate_new
    return duplicate


class CorticalEmbedding(object):
    def __init__(
        self,
        duplicates,
        constraints,
        distance_matrix_path,
        cortical_areas_path,
        sensory,
        motor,
        species,
    ):

        self.duplicates = resolve_duplicates(duplicates, cortical_areas_path)

        self.mask_within_area_weights = constraints["mask_within_area_weights"]
        self.zero_weights_thres = constraints["zero_weights_thres"]

        self.cortical_areas = get_areas(cortical_areas_path)

        self.original_distance_matrix = load_distance_matrix(distance_matrix_path)
        self.distance_matrix = update_distance_matrix(
            self.duplicates, self.original_distance_matrix, self.cortical_areas
        )

        self.area2idx = get_area2idx(self.cortical_areas, self.duplicates)

        self.area_mask = self._build_area_mask()

        self.sensory = sensory
        self.motor = motor
        self.species = species

        # self.distance_from_sensory = get_distance_from(
        #     self.sensory[0], self.original_distance_matrix, self.cortical_areas
        # )

    def _build_area_mask(self):
        """Make NxN mask where mask[i,j] = 1 if i,j in same area."""
        n_total = self.distance_matrix.shape[0]
        mask = np.zeros((n_total, n_total), dtype=np.float32)
        for area in self.cortical_areas:
            start_idx = self.area2idx[area]
            length = self.duplicates[area] if area in self.duplicates else 1
            end_idx = start_idx + length
            mask[start_idx:end_idx, start_idx:end_idx] = 1.0
        return mask

    def _build_activity_mask(self, area_list: List[str]):
        """
        Return a 1D mask of length n_rnn, with 1.0 if that unit belongs
        to any of the areas in area_list, else 0.0
        """
        n_total = self.distance_matrix.shape[0]
        mask = np.zeros((n_total,), dtype=np.float32)
        for area in area_list:
            if area in self.area2idx:
                start_idx = self.area2idx[area]
                length = self.duplicates[area] if area in self.duplicates else 1
                mask[start_idx : start_idx + length] = 1.0
        return mask


class MacaqueCorticalEmbedding(CorticalEmbedding):
    def __init__(
        self,
        duplicates,
        constraints,
        sensory,
        motor,
        distance_matrix_path=None,
        cortical_areas_path=None,
    ):
        distance_matrix_path = DISTANCE_MATRIX_PATH_MACAQUE
        cortical_areas_path = CORTICAL_AREAS_PATH_MACAQUE
        super().__init__(
            duplicates,
            constraints,
            distance_matrix_path,
            cortical_areas_path,
            sensory,
            motor,
            species="macaque",
        )

        # self.sensory = ["V1", "3"]  # TODO Better way of doign this in the config
        # self.motor = ["8l"]


class HumanCorticalEmbedding(CorticalEmbedding):
    def __init__(
        self,
        duplicates,
        constraints,
        sensory,
        motor,
        distance_matrix_path=None,
        cortical_areas_path=None,
    ):
        distance_matrix_path = DISTANCE_MATRIX_PATH
        cortical_areas_path = CORTICAL_AREAS_PATH
        super().__init__(
            duplicates,
            constraints,
            distance_matrix_path,
            cortical_areas_path,
            sensory,
            motor,
            species="human",
        )

        # self.sensory = ["L_V1", "L_3b"]  # TODO Better way of doign this in the config
        # self.motor = ["L_FEF"]

        #  DMN region lists (extended DMN set)
        self._dmn_areas = [
            "L_7m",
            "L_v23ab",
            "L_d23ab",
            "L_31pv",
            "L_a24",
            "L_d32",
            "L_p32",
            "L_10r",
            "L_47m",
            "L_8Ad",
            "L_9m",
            "L_8BL",
            "L_9p",
            "L_10d",
            "L_STSda",
            "L_STSvp",
            "L_TE1a",
            "L_PGi",
            "L_31pd",
            "L_STSva",
            "L_TE1m",
            "L_p24",
        ]

        # Core DMN region list based on the criteria that the value be at least 0.8 and that it exceed
        # the next-highest network value by at least 0.3
        self._core_dmn_areas = [
            "L_7m",
            "L_v23ab",
            "L_d23ab",
            "L_31pv",
            "L_a24",
            "L_d32",
            "L_p32",
            "L_10r",
            "L_47m",
            "L_8Ad",
            "L_9m",
            "L_8BL",
            "L_9p",
            "L_10d",
            "L_STSda",
            "L_STSvp",
            "L_TE1a",
            "L_PGi",
            "L_31pd",
            "L_STSva",
            "L_TE1m",
            "L_p24",
        ]

        self.cog_network_overlap = self.get_cog_network_overlap()

        # 1D masks for activity-based penalties
        self.dmn_mask = self._build_activity_mask(self._dmn_areas)
        self.core_dmn_mask = self._build_activity_mask(self._core_dmn_areas)

    def get_cog_network_overlap(
        self,
    ):
        df = pd.read_csv(COG_NETWORK_OVERLAP)
        return df
