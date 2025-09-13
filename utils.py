import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree


# Mesh & HU Processing

def read_mesh(inp_path):
    """Read nodes and elements from an Abaqus .inp mesh file."""
    with open(inp_path, 'r') as f:
        lines = f.readlines()

    idx_node = next(i for i, line in enumerate(lines) if '*Node' in line)
    idx_elem = next(i for i, line in enumerate(lines) if '*Element' in line)
    idx_end = next(i for i, line in enumerate(lines) if '*End' in line)

    nodes = np.array([list(map(float, line.strip().split(',')))
                      for line in lines[idx_node + 1:idx_elem]], dtype=np.float32)

    elements_raw = [list(map(int, line.strip().split(','))) for line in lines[idx_elem + 1:idx_end]]
    node_xyz = nodes[:, 1:]
    centroids = np.array([np.mean(node_xyz[np.array(e[1:]) - 1], axis=0) for e in elements_raw], dtype=np.float32)
    elements = np.hstack((np.arange(1, len(elements_raw) + 1).reshape(-1, 1), centroids))
    return nodes.tolist(), elements_raw, elements

def hu_to_density(hu, slope, intercept):
    """Convert Hounsfield units (HU) to bone density."""
    return slope / 1000 * hu + intercept / 1000

def density_to_modulus(density, b, c):
    """Convert density to elastic modulus using power law."""
    density = np.where(density < 0.01, 1e-2, density)
    return b * density ** c

# Interpolation & Binning

def fill_nan_with_nearest(grid):
    nan_mask = np.isnan(grid)
    if not np.any(nan_mask):
        return grid
    filled = grid.copy()
    valid_idx = np.argwhere(~nan_mask)
    nan_idx = np.argwhere(nan_mask)
    tree = cKDTree(valid_idx)
    _, nearest_indices = tree.query(nan_idx)
    for ni, nearest in zip(nan_idx, nearest_indices):
        filled[tuple(ni)] = grid[tuple(valid_idx[nearest])]
    return filled

def interpolate_modulus(coords, values, targets, resolution=RESOLUTION, method='linear'):
    rounded = np.round(coords / resolution) * resolution
    x, y, z = [np.unique(rounded[:, i]) for i in range(3)]
    x = np.concatenate([[x[0] - resolution], x, [x[-1] + resolution]])
    y = np.concatenate([[y[0] - resolution], y, [y[-1] + resolution]])
    z = np.concatenate([[z[0] - resolution], z, [z[-1] + resolution]])

    xi = {v: i for i, v in enumerate(x)}
    yi = {v: i for i, v in enumerate(y)}
    zi = {v: i for i, v in enumerate(z)}
    shape = (len(x), len(y), len(z))
    grid = np.full(shape, np.nan, dtype=np.float32)

    for i in range(len(rounded)):
        cx, cy, cz = rounded[i]
        if cx in xi and cy in yi and cz in zi:
            grid[xi[cx], yi[cy], zi[cz]] = values[i]

    interpolator = RegularGridInterpolator((x, y, z), grid, method=method, bounds_error=False, fill_value=np.nan)
    interpolated = interpolator(targets)

    if np.isnan(interpolated).any():
        grid_filled = fill_nan_with_nearest(grid)
        nearest_interp = RegularGridInterpolator((x, y, z), grid_filled, method='nearest', bounds_error=False, fill_value=None)
        interpolated[np.isnan(interpolated)] = nearest_interp(targets[np.isnan(interpolated)])
    return interpolated

def assign_bins(E_values, bin_width=BIN):
    E_values = np.maximum(E_values, 1e-5)
    min_val = max(1e-5, np.floor(np.min(E_values)))
    max_val = np.ceil(np.max(E_values))
    bins = np.arange(min_val, max_val + bin_width, bin_width)
    bin_ids = np.clip(np.digitize(E_values, bins, right=True), 1, len(bins) - 1)
    element_sets = {i: [] for i in range(1, len(bins))}
    for idx, bin_id in enumerate(bin_ids):
        element_sets[bin_id].append(idx + 1)
    return element_sets, bins


# Output Writing

def write_inp_file(out_path, model_name, nodes, elements, element_sets, bins, b, c):
    nodes_coords = np.array([n[1:] for n in nodes])
    rotated_nodes = np.hstack((np.array([n[0] for n in nodes]).reshape(-1, 1), nodes_coords))

    with open(out_path, 'w') as f:
        f.write("*Heading\n")
        f.write(f"** Job name: {model_name}\n")
        f.write("*Preprint, echo=NO, model=NO, history=NO, contact=NO\n")
        f.write(f"*Part, name={model_name}\n*Node\n")
        f.writelines([f"{int(n[0])}, {n[1]}, {n[2]}, {n[3]}\n" for n in rotated_nodes])
        f.write("*Element, type=C3D4\n")
        f.writelines([f"{i+1}, {', '.join(map(str, ele[1:]))}\n" for i, ele in enumerate(elements)])
        for i in element_sets:
            f.write(f"*Elset, elset=Set-{i}\n")
            f.writelines(', '.join(map(str, element_sets[i][j:j+16])) + '\n' for j in range(0, len(element_sets[i]), 16))
        for i in element_sets:
            f.write(f"*Solid Section, elset=Set-{i}, material=MAT_{i}\n")
        f.write("*End Part\n** MATERIALS\n")
        for i in element_sets:
            E = bins[i - 1]
            density = (E / b) ** (1 / c)
            density = np.maximum(density, 1e-5)
            f.write(f"*Material, name=MAT_{i}\n")
            f.write(f"*Density\n{density:.9f}\n")
            f.write(f"*Elastic\n{E:.9f}, {POISSON_RATIO}\n")

def write_summary_csv(out_path, element_sets, bins, w):
    with open(out_path, 'w') as f:
        f.write('Bin, Num_Elements, Avg_E, Avg_Density\n')
        for i in element_sets:
            E = bins[i - 1]
            density = (1 / w[1] * E) ** (1 / w[2])
            f.write(f"{i}, {len(element_sets[i])}, {E:.6f}, {density:.6f}\n")
