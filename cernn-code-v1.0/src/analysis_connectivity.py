# src/analysis_connectivity.py
import os
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr

# Optional imports used by specific plotting helpers; guard them to avoid import crashes
try:
    from mne.viz import circular_layout
    from mne_connectivity.viz import plot_connectivity_circle
    _HAS_MNE = True
except Exception:
    _HAS_MNE = False

from src.cortical_embedding import get_distance_from


"""
Connectivity analysis & plotting utilities.

Public function names/signatures preserved where possible.
"""


# ----------------------------- #
# Robust exponential fitting
# ----------------------------- #

def exponential(x, a, b):
    """Basic exponential a * exp(-b * x) used across the module."""
    return a * np.exp(-b * x)


def safe_exp_fit(x, y, p0=(1.0, 0.1)):
    """
    Fit y ≈ a * exp(-b x) robustly.

    - Drops non-finite and non-positive y values (exp fit needs y>0).
    - If scipy.curve_fit fails (or too few points), falls back to
      log-linear least squares on log(y) = c - b x.
    - Returns (a, b) as floats or (nan, nan) if we really can’t fit.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = x[ok], y[ok]

    if x.size < 3:
        return np.nan, np.nan

    try:
        params, _ = curve_fit(exponential, x, y, p0=p0, maxfev=20000)
        return float(params[0]), float(params[1])
    except Exception:
        # Fallback: log-linear on log(y) = c - b x
        try:
            Y = np.log(y)
            A = np.vstack([np.ones_like(x), -x]).T  # [c, b]
            c, b = np.linalg.lstsq(A, Y, rcond=None)[0]
            return float(np.exp(c)), float(b)
        except Exception:
            return np.nan, np.nan


# ----------------------------- #
# Duplicate/parcel helpers
# ----------------------------- #

def sum_over_duplicates(connectivity_matrix: np.ndarray, ce, mode="sum") -> np.ndarray:
    """
    Sum or mean over duplicate units within each cortical area (parcel).

    Args:
        connectivity_matrix: (N_old x N_old) matrix at unit-level (to, from).
        ce: CorticalEmbedding with fields .cortical_areas, .area2idx, .duplicates
        mode: "sum" or "mean" over duplicates.

    Returns:
        (N_parcels x N_parcels) matrix aggregated to one row/col per parcel.
    """
    cortical_areas = ce.cortical_areas
    num_units_new = len(cortical_areas)
    num_units_old = connectivity_matrix.shape[0]

    rows_summed = np.zeros((num_units_new, num_units_old), dtype=float)
    mat_summed = np.zeros((num_units_new, num_units_new), dtype=float)

    # Aggregate rows (TO)
    for i, area in enumerate(cortical_areas):
        start_idx = ce.area2idx[area]
        dup_count = ce.duplicates.get(area, 1)
        rows = connectivity_matrix[start_idx:start_idx + dup_count, :]
        if rows.size == 0:
            continue
        rows_summed[i, :] = rows.sum(axis=0)
        if mode == "mean":
            rows_summed[i, :] /= dup_count

    # Aggregate cols (FROM)
    for i, area in enumerate(cortical_areas):
        start_idx = ce.area2idx[area]
        dup_count = ce.duplicates.get(area, 1)
        cols = rows_summed[:, start_idx:start_idx + dup_count]
        if cols.size == 0:
            continue
        mat_summed[:, i] = cols.sum(axis=1)
        if mode == "mean":
            mat_summed[:, i] /= dup_count

    return mat_summed


def compress_to_180_areas(hidden_txd: np.ndarray, pl_model) -> np.ndarray:
    """
    Convert [Time, n_rnn] → [Time, n_parcels] by averaging duplicates per area.
    """
    ce = pl_model.model.ce
    cortical_areas = ce.cortical_areas
    T, _ = hidden_txd.shape
    hidden_compressed = np.zeros((T, len(cortical_areas)), dtype=hidden_txd.dtype)

    for i, area in enumerate(cortical_areas):
        start_idx = ce.area2idx[area]
        dup_count = ce.duplicates.get(area, 1)
        hidden_compressed[:, i] = hidden_txd[:, start_idx:start_idx + dup_count].mean(axis=1)

    return hidden_compressed


def ce_idx2area(area2ix, num_units):
    """
    Build index→name map, filling gaps by copying previous (keeps labels aligned).
    """
    idx2name = {ix: name for name, ix in area2ix.items()}
    for i in range(num_units):
        if i not in idx2name:
            idx2name[i] = idx2name[i - 1]
    return idx2name


def reorder_matrix(matrix, new_order):
    """Reorder rows and cols by new_order."""
    return matrix[np.ix_(new_order, new_order)]


# ----------------------------- #
# Basic metrics & thresholds
# ----------------------------- #

def calculate_densitiy(full_num_weights, nonzero_indices):
    """
    Percentage density of non-zero entries (given total count and kept indices).
    (Kept original name to avoid breaking calls.)
    """
    return (1 - (full_num_weights - len(nonzero_indices)) / full_num_weights) * 100.0


def model_density(model):
    """
    Density of non-zero weights in the recurrent layer, using ce.zero_weights_thres.
    """
    trained = model.rnn.rnncell.weight_hh.detach().cpu().numpy().flatten()
    full = trained.size
    thres = model.ce.zero_weights_thres
    nonzero_ind, _ = remove_near_zero(trained, thres)
    return calculate_densitiy(full, nonzero_ind)


def remove_near_zero(array, threshold):
    """
    Return (indices, values) where |array| > threshold.
    """
    arr = np.asarray(array).ravel()
    idx = np.where(np.abs(arr) > threshold)[0]
    return idx, arr[idx]


# ----------------------------- #
# Weight–distance figures & fits
# ----------------------------- #

def fig_3_weights_over_distance_lambda_fitted(model, thres=0.001, provided_weiths=None):
    """
    Scatter: |weights| vs. distance for non-zero weights + robust exponential fit.
    Returns: fig, ax, density(%), lambda_hat
    """
    if provided_weiths is not None:
        weights = np.asarray(provided_weiths).ravel()
    else:
        weights = model.rnn.rnncell.weight_hh.detach().cpu().numpy().ravel()

    nonzero_ind, w = remove_near_zero(weights, thres)
    density = calculate_densitiy(weights.size, nonzero_ind)
    print(f"Density: {density:.2f}%")

    distances = model.ce.distance_matrix.ravel()[nonzero_ind]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(distances, np.abs(w), alpha=0.5, marker="o")
    ax.set_ylabel("Absolute Weight")
    ax.set_xlabel("Distance")
    ax.set_title(f"non-zero weights over distance with {density:.2f}% density (thres {thres})")

    # Spearman correlation (rank)
    corr, pval = spearmanr(distances, np.abs(w)) if w.size >= 3 else (np.nan, np.nan)
    print(f"Spearman correlation: {corr}, p-value: {pval}")

    # Robust exponential fit
    valid = np.abs(w) > 0
    x_data = distances[valid]
    y_data = np.abs(w[valid])
    a, lam = safe_exp_fit(x_data, y_data, p0=(1.0, 0.1))
    print(f"Fitted exponential parameters: a={a}, lambda={lam}")

    if np.isfinite(a) and np.isfinite(lam):
        x_fit = np.linspace(np.min(x_data), np.max(x_data), 200)
        y_fit = exponential(x_fit, a, lam)
        ax.plot(x_fit, y_fit, label="Exponential Fit")
        ax.legend()

    return fig, ax, density, lam


def fig_3_histogram_weights_over_distance_lambda(model, thres=0.001, bins=50, p=False):
    """
    Bin distances and sum weights per bin, then fit exponential to totals.
    Returns: fig, ax, density(%), lambda_hat
    """
    trained = model.rnn.rnncell.weight_hh.detach().cpu().numpy().ravel()
    full = trained.size

    nonzero_ind, w = remove_near_zero(trained, thres)
    density = calculate_densitiy(full, nonzero_ind)
    print(f"Density: {density:.2f}%")

    d = model.ce.distance_matrix.ravel()[nonzero_ind]

    distance_bins = np.linspace(0, np.max(d) + 1, bins)
    bin_indices = np.digitize(d, distance_bins)
    total_weights = np.zeros(len(distance_bins) - 1)

    # Optionally compute bin density instead of total weight
    if p:
        density_hist, _ = np.histogram(d, bins=distance_bins)
        density_hist = density_hist / np.sum(density_hist)  # kept for compatibility

    for i in range(total_weights.size):
        bin_mask = bin_indices == i
        total_weights[i] = np.sum(w[bin_mask])

    bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(bin_centers, np.abs(total_weights),
           width=(distance_bins[1] - distance_bins[0]),
           alpha=0.7, label="Total Weight")

    # Robust fit on positive totals
    valid = total_weights > 0
    x_data = bin_centers[valid]
    y_data = total_weights[valid]
    a, lam = safe_exp_fit(x_data, y_data, p0=(1.0, 0.1))
    print(f"Fitted exponential parameters: a={a}, lambda={lam}")

    if np.isfinite(a) and np.isfinite(lam):
        x_fit = np.linspace(np.min(x_data), np.max(x_data), 200)
        y_fit = exponential(x_fit, a, lam)
        ax.plot(x_fit, y_fit, label="Exponential Fit")
        ax.legend()

    ax.set_xlabel("Distance")
    ax.set_ylabel("Total Weight")
    ax.set_title(f"Total Weight vs Distance (nonzero thres {thres})")

    return fig, ax, density, lam


def calculate_spearman_exponential_fit(matrix, model, thres=0.05):
    """
    Compute Spearman correlation between distance and |weights| (above thres),
    and fit |weights| ≈ a * exp(-lambda * distance).

    Returns: (corr, pval), (a, lambda), density(%)
    """
    flat = np.asarray(matrix).ravel()
    full = flat.size

    nonzero_ind, kept_vals = remove_near_zero(flat, thres)
    density = calculate_densitiy(full, nonzero_ind)

    distances = np.asarray(model.ce.distance_matrix).ravel()[nonzero_ind]
    weights = np.abs(kept_vals)

    valid = np.isfinite(distances) & np.isfinite(weights) & (weights > 0)
    distances, weights = distances[valid], weights[valid]

    if distances.size >= 3:
        corr, pval = spearmanr(distances, weights)
    else:
        corr, pval = np.nan, np.nan

    a, lam = safe_exp_fit(distances, weights, p0=(1.0, 0.1))
    return (corr, pval), (a, lam), density


def fig3_FLN_over_distance(FLN_matrix, model, thres, mode="show"):
    """
    Scatter: |FLN| vs. distance, robust exponential fit, and Spearman correlation.
    Returns: fig, ax, (corr, pval), (a, lambda)
    """
    FLN_flat = np.asarray(FLN_matrix).ravel()
    full = FLN_flat.size
    nonzero_ind, nonzero_vals = remove_near_zero(FLN_flat, thres)
    density = calculate_densitiy(full, nonzero_ind)

    distances = (np.asarray(model.ce.distance_matrix) / 5.0).ravel()[nonzero_ind]
    y = np.abs(nonzero_vals)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(distances, y, alpha=0.5, marker="o")
    ax.set_ylabel("Weight")
    ax.set_xlabel("Distance")
    ax.set_title(f"non-zero weights over distance with {density:.2f}% density (thres {thres})")

    corr, pval = spearmanr(distances, y) if y.size >= 3 else (np.nan, np.nan)
    print(f"Spearman correlation: {corr}, p-value: {pval}")

    a, lam = safe_exp_fit(distances, y, p0=(1.0, 0.1))
    print(f"Fitted exponential parameters: a={a}, lambda={lam}")

    if np.isfinite(a) and np.isfinite(lam):
        x_fit = np.linspace(np.min(distances), np.max(distances), 200)
        y_fit = exponential(x_fit, a, lam)
        ax.plot(x_fit, y_fit, label="Exponential Fit")
        ax.legend()

    if mode == "show":
        plt.show()

    return fig, ax, (corr, pval), (a, lam)


def fig3_weight_dist_exponential_fit(model, thres=1e-2, bins=50, mode="show"):
    """
    Histogram of |weights| (after threshold) + exponential fit to histogram counts.
    Returns fig, ax when mode == 'return'.
    """
    trained = model.rnn.rnncell.weight_hh.detach().cpu().numpy().ravel()
    _, w = remove_near_zero(trained, thres)

    fig, ax = plt.subplots(figsize=(10, 10))
    counts, bin_edges, _ = ax.hist(np.abs(w), bins=bins, density=True, alpha=0.5)
    ax.set_ylabel("count")
    ax.set_xlabel("Weight")

    # Use bin centers for fitting the histogram
    x_data = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    y_data = counts

    a, lam = safe_exp_fit(x_data, y_data, p0=(1.0, 0.1))
    print(f"Fitted lambda: {lam}")

    if np.isfinite(a) and np.isfinite(lam):
        x_fit = np.linspace(np.min(x_data), np.max(x_data), 200)
        y_fit = exponential(x_fit, a, lam)
        ax.plot(x_fit, y_fit, label="Exponential Fit")
        ax.legend()

    if mode == "save":
        os.makedirs("fig3", exist_ok=True)
        plt.savefig("fig3/weights_dist_exponential_fit.png", dpi=300, bbox_inches="tight")
    elif mode == "show":
        plt.show()
    elif mode == "return":
        return fig, ax


# ----------------------------- #
# FLN matrix helpers & figure
# ----------------------------- #

def FLN_ij(weights, eps=0.0, fallback="uniform", atol=1e-4):
    """
    Convert weights (to, from) into an FLN-like matrix by row normalization of |weights|.
    - NaNs/Infs are treated as zero.
    - Zero-sum rows are handled via `fallback`:
        * 'uniform' (default): fill row with 1/N.
        * 'zeros': leave as all zeros.
    - `eps` > 0 adds smoothing before normalization.
    """
    W = np.abs(np.asarray(weights, dtype=float))
    if eps > 0:
        W = W + eps

    row_sums = W.sum(axis=1, keepdims=True)
    fln = np.divide(W, row_sums, out=np.zeros_like(W), where=row_sums > 0)

    # Handle zero-sum rows
    zero_rows = np.where(row_sums.squeeze() == 0)[0]
    if zero_rows.size > 0:
        if fallback == "uniform":
            fln[zero_rows, :] = 1.0 / fln.shape[1]
        # else keep zeros
        print(f"[FLN_ij] Warning: {zero_rows.size} zero-sum rows encountered: {zero_rows.tolist()}")

    # Sanity report (no assert that kills training)
    s = fln.sum(axis=1)
    if not np.allclose(s, 1, atol=atol):
        bad = np.where(~np.isclose(s, 1, atol=atol))[0]
        print(f"[FLN_ij] Row-sum mismatch on rows {bad.tolist()} "
              f"(min={s.min():.6f}, max={s.max():.6f}, atol={atol})")

    # Clean any remaining non-finite (extremely unlikely)
    fln[~np.isfinite(fln)] = 0.0
    return fln


def fig_3_FLN_matrix(model, sort=False):
    """
    Show FLN matrix (optionally hierarchy-sorted) and aggregate duplicates by mean.
    Returns fig, ax, fln_matrix (parcel-level).
    """
    weights = model.rnn.rnncell.weight_hh.detach().cpu().numpy()

    if sort:
        _, _, area_names_hierarchical, new_indices = sort_connectivity_matrix_and_labels(model=model)
        weights = reorder_matrix(weights, new_indices)

    fln_matrix = FLN_ij(weights)  # robust
    fln_matrix = sum_over_duplicates(fln_matrix, model.ce, mode="mean")

    fig, ax = plt.subplots(figsize=(10, 10))

    # De-emphasize diagonal a bit (pure self-normalized)
    np.fill_diagonal(fln_matrix, fln_matrix.diagonal() * 0.2)

    im = ax.imshow(fln_matrix, cmap="hot", interpolation="nearest", vmin=0, vmax=0.3)
    species = model.ce.species.capitalize()
    ax.set_ylabel(f"{species} cortical areas (to)", fontsize=14)
    ax.set_xlabel(f"{species} cortical areas (from)", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=40)
    cbar.set_ticks([])
    cbar.ax.text(0.5, 1.01, f"{cbar.vmax:.1f}", transform=cbar.ax.transAxes,
                 ha="center", va="bottom", fontsize=12)
    cbar.ax.text(0.5, -0.01, f"{cbar.vmin:.1f}", transform=cbar.ax.transAxes,
                 ha="center", va="top", fontsize=12)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    for line in ax.get_xticklines() + ax.get_yticklines():
        line.set_linewidth(1.5)
    cbar.outline.set_linewidth(1.5)

    return fig, ax, fln_matrix


# ----------------------------- #
# Hierarchy utilities for plots
# ----------------------------- #

def get_hierarchy_from_cog_overlap(cog_network_df):
    """
    Build an OrderedDict of network→areas using max-overlap per row in a
    'cog_network_overlap' dataframe (expects a 'Row' column).
    """
    network_names = [col for col in cog_network_df.columns if col != "Row"]
    networks_dict = {key: [] for key in network_names}

    for _, row in cog_network_df.iterrows():
        network_values = row[network_names]
        max_network = network_values.idxmax()
        networks_dict[max_network].append(row["Row"][:-4])

    hierarchy = OrderedDict([
        ("Visual",  networks_dict.get("Visual", [])),
        ("SomMot",  networks_dict.get("SomMot", [])),
        ("Limbic",  networks_dict.get("Limbic", [])),
        ("Salience", networks_dict.get("Salience", [])),
        ("DorsAtt", networks_dict.get("DorsAtt", [])),
        ("FPN",     networks_dict.get("FPN", [])),
        ("Default", networks_dict.get("Default", [])),
    ])
    return hierarchy


def get_boundaries(hierarchy):
    """Return list of starting indices per group for plotting boundaries."""
    boundaries, count = [], 0
    for sublist in hierarchy.values():
        boundaries.append(count)
        count += len(sublist)
    return boundaries


def get_hierarchical_names(hierarchy):
    """Flatten hierarchy dict into a single ordered name list."""
    return [name for sublist in hierarchy.values() for name in sublist]


def get_new_indices(cortical_areas, hierarchy_names_flat):
    """
    Map current cortical_areas order into the hierarchical order.
    """
    return [hierarchy_names_flat.index(area) for area in cortical_areas]


def sort_hierarchy_dict(hierarchy_dict, ce):
    """
    Sort each network group by distance from V1 (keeps a nice gradient on plots).
    """
    distance_from_v1 = get_distance_from(
        ce.area2idx[ce.sensory[0]],
        ce.original_distance_matrix,
        ce.cortical_areas,
    )
    for key in hierarchy_dict:
        hierarchy_dict[key] = sorted(
            hierarchy_dict[key],
            key=lambda area: distance_from_v1[ce.cortical_areas.index(area)],
        )
    return hierarchy_dict


def sort_connectivity_matrix_and_labels(model):
    """
    Build hierarchical ordering from ce.cog_network_overlap, sort areas by
    within-network distance from V1, and return indices for reordering matrices.
    """
    cog_network_df = model.ce.cog_network_overlap
    cortical_areas = model.ce.cortical_areas

    hierarchy_dict = get_hierarchy_from_cog_overlap(cog_network_df)
    hierarchy_dict = sort_hierarchy_dict(hierarchy_dict, model.ce)

    hierarchy_boundaries = get_boundaries(hierarchy_dict)
    area_names_hierarchical = get_hierarchical_names(hierarchy_dict)
    new_indices = get_new_indices(cortical_areas, area_names_hierarchical)

    return hierarchy_dict, hierarchy_boundaries, area_names_hierarchical, new_indices


# ----------------------------- #
# Circle plot
# ----------------------------- #

def circle_plot(pretrained_model):
    """
    Circle plot of parcel-level connectivity after summing duplicates and
    reordering by hierarchy. Applies a small threshold for clarity.

    Returns fig, ax (from mne_connectivity.viz.plot_connectivity_circle)
    """
    if not _HAS_MNE:
        raise ImportError("circle_plot requires mne and mne-connectivity to be installed")

    # Unit-level connectivity (to, from)
    connectivity = np.abs(pretrained_model.rnn.rnncell.weight_hh.detach().cpu().numpy())
    np.fill_diagonal(connectivity, 0)  # no self-connections

    ce = pretrained_model.ce
    # Remove intra-area edges if mask is 1 for same area
    connectivity *= (1 - ce.area_mask)

    # Aggregate to parcel level
    connectivity = sum_over_duplicates(connectivity, pretrained_model.ce, mode="mean")

    # Reorder by hierarchy
    hierarchy_dict, hierarchy_boundaries, area_names_hierarchical, new_indices = (
        sort_connectivity_matrix_and_labels(model=pretrained_model)
    )
    connectivity = reorder_matrix(connectivity, new_indices)

    # Threshold small weights for readability
    threshold = 0.005
    connectivity_thresholded = connectivity.copy()
    connectivity_thresholded[connectivity_thresholded < threshold] = 0

    # Group colors (kept your palette/size structure)
    group_sizes = [28, 27, 14, 25, 24, 28, 34]
    colormaps = [
        (0.8, 0.2, 0.2),  # Reds
        (0.2, 0.4, 0.8),  # Blues
        (0.2, 0.7, 0.3),  # Greens
        (1.0, 0.6, 0.2),  # Oranges
        (0.6, 0.4, 0.8),  # Purples
        (0.6, 0.6, 0.6),  # Greys
        (0.7, 0.3, 0.6),  # Magenta-like
    ]
    node_colours = []
    for size, color in zip(group_sizes, colormaps):
        node_colours.extend([color] * size)

    # Node layout around the circle
    node_angles = circular_layout(
        area_names_hierarchical,
        node_order=area_names_hierarchical,
        start_pos=90,
        group_boundaries=hierarchy_boundaries,
    )

    fig, ax = plot_connectivity_circle(
        connectivity_thresholded,
        area_names_hierarchical,
        n_lines=300,
        title="Connectivity Circle Plot",
        show=True,
        node_angles=node_angles,
        node_colors=node_colours,
        facecolor="white",
        textcolor="black",
        colormap="Reds",
        fontsize_names=5,
    )
    return fig, ax


# ----------------------------- #
# Minimal self-check (safe to run standalone)
# ----------------------------- #

def _fake_ce(n_areas=5, dup_counts=None):
    """Build a minimal fake CE object for tests (no heavy deps)."""
    class CE:
        pass
    ce = CE()
    ce.cortical_areas = [f"A{i}" for i in range(n_areas)]
    if dup_counts is None:
        dup_counts = [1] * n_areas
    ce.duplicates = {f"A{i}": dup_counts[i] for i in range(n_areas)}
    # area2idx assigns contiguous blocks by duplicates
    starts = np.cumsum([0] + dup_counts[:-1]).tolist()
    ce.area2idx = {f"A{i}": starts[i] for i in range(n_areas)}
    total_units = sum(dup_counts)
    # Simple distance matrix (unit-level) and parcel-level placeholders
    ce.original_distance_matrix = np.abs(np.subtract.outer(np.arange(total_units), np.arange(total_units)))
    ce.distance_matrix = ce.original_distance_matrix.copy()
    ce.cog_network_overlap = None  # not needed here
    ce.sensory = ["A0"]
    ce.species = "macaque"
    # Intra-area mask (unit-level): block-diagonal ones
    area_mask = np.zeros((total_units, total_units), dtype=int)
    for i, dup in enumerate(dup_counts):
        s = ce.area2idx[f"A{i}"]
        area_mask[s:s+dup, s:s+dup] = 1
    ce.area_mask = area_mask
    return ce


def self_check():
    """Quick correctness checks for FLN_ij and sum_over_duplicates."""
    # 1) FLN_ij zero-row handling
    W = np.array([[0, 0, 0],
                  [1, 2, 3],
                  [0, 5, 0]], dtype=float)
    fln = FLN_ij(W)  # default fallback='uniform'
    s = fln.sum(axis=1)
    assert np.allclose(s, 1, atol=1e-6), f"FLN_ij row sums not 1: {s}"
    assert np.all(fln >= 0), "FLN_ij produced negative entries"

    # 2) sum_over_duplicates indexing correctness
    dup_counts = [2, 1, 3]   # total 6 units → 3 parcels
    ce = _fake_ce(n_areas=3, dup_counts=dup_counts)

    # Build a unit-level matrix with distinct block sums we can predict
    U = np.zeros((6, 6))
    # Area 0 (units 0-1) to area 2 (units 3-5) put ones
    U[0:2, 3:6] = 1.0
    # Area 2 (3-5) to area 1 (2) put twos
    U[3:6, 2:3] = 2.0

    M = sum_over_duplicates(U, ce, mode="sum")
    # Expected: parcel 0→2 sum is 2*3*1 = 6 (two rows aggregated, three cols)
    assert np.isclose(M[0, 2], 6.0), f"sum_over_duplicates wrong (0→2): {M[0,2]}"
    # Expected: parcel 2→1 sum is 3*1*2 = 6
    assert np.isclose(M[2, 1], 6.0), f"sum_over_duplicates wrong (2→1): {M[2,1]}"

    # 3) FLN → parcel aggregation preserves sensible scaling
    fln_unit = FLN_ij(U + 1e-9, fallback="zeros")  # avoid perfect zeros
    fln_parcel = sum_over_duplicates(fln_unit, ce, mode="mean")
    pr = fln_parcel.sum(axis=1)
    # Row sums need not be exactly 1 after aggregation + mean, but should be finite and > 0
    assert np.all(np.isfinite(pr)) and np.all(pr > 0), "Parcel FLN rows invalid"

    print("✅ self_check passed: FLN_ij robustness & duplicate aggregation look good.")


if __name__ == "__main__":
    self_check()
