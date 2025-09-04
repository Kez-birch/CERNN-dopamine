import os
from pathlib import Path

# current_working_dir = Path(os.getcwd())
# print(current_working_dir)

work_dir = Path(os.environ["HOME"].replace("home", "work"))
checkpoint_dir = "saved_models"
try:
    if os.environ["WHEREAMI"] == "cluster":
        checkpoint_dir = work_dir / checkpoint_dir
except:
    pass

current_dir = os.getcwd().split("/")[-1]
app = "../" if current_dir == "notebooks" else ""

DISTANCE_MATRIX_PATH = app + "src/data/LeftParcelGeodesicDistmat.txt"
DISTANCE_MATRIX_PATH_MACAQUE = app + "src/data/macaque_geodesic_distance.npy"

CORTICAL_AREAS_PATH = app + "src/data/areaNamesGlasser180.txt"
CORTICAL_AREAS_PATH_MACAQUE = app + "src/data/macaque_area_names.txt"


COG_NETWORK_OVERLAP = app + "src/data/hcp_regions_cog_networks_overlap.csv"

SPINE_COUNT_MACAQUE = app + "src/data/spine_count_lyon_regions.mat"
SPINE_COUNT_HUMAN = app + "src/data/myelin_HCP_vec.mat"
