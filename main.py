import time
from pathlib import Path
import numpy as np
import pandas as pd
from utils import read_mesh, hu_to_density, density_to_modulus, interpolate_modulus, assign_bins, write_inp_file, write_summary_csv

# Paths & parameters

Dir = Path("data")  # adjust to your repo structure
Mimics_Dir = Dir / "HU" # adjust to your repo structure
Mesh_Dir = Dir / "Mesh" # adjust to your repo structure
Density_Modulus_File = Dir / "Equations.xlsx" # adjust to your repo structure
Eqs_File = Dir / "Calibrations.xlsx" # adjust to your repo structure

RESOLUTION = 0.08 #recomended
BIN_WIDTH = BIN # adjust the interval values based on your problem
POISSON_RATIO = 0.3 #you can change based on your material

# Main workflow
def main():
    density_modulus_eqs = pd.read_excel(Density_Modulus_File).values.tolist()
    calibration_eqs = pd.read_excel(Eqs_File).values.tolist()

    total_start = time.perf_counter()

    for eq in calibration_eqs:
        model_id, slope, intercept = str(eq[0]), eq[1], eq[2]
        mat_dir = Dir / "Materials"
        mat_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing model {model_id}...")
        mesh_file = Mesh_Dir / f"{model_id}_mesh.inp"
        hu_file = Mimics_Dir / f"{model_id}_HU.txt"

        nodes, elements_raw, centroids = read_mesh(mesh_file)
        hu_data = np.loadtxt(hu_file, delimiter=',').astype(np.float32)
        HU = hu_data[:, 3]
        voxel_coords = hu_data[:, :3]
        pqct = hu_to_density(HU, slope, intercept)

        for row in density_modulus_eqs:
            rel_id, b, c = row[:3]
            c_str = f"{c}".replace('.', '_')
            print(f"Processing equation {rel_id} with b={b}, c={c} ...")
            model_name = f"Model_{model_id}"

            Et = density_to_modulus(pqct, b, c)
            Et_interp = interpolate_modulus(voxel_coords, Et, centroids[:, 1:])
            element_sets, bins = assign_bins(Et_interp)

            out_inp = mat_dir / f"{model_id}_b{b}c{c_str}_Mat.inp"
            out_csv = mat_dir / f"{model_id}_b{b}c{c_str}.csv"

            write_inp_file(out_inp, model_name, nodes, elements_raw, element_sets, bins, b, c)
            write_summary_csv(out_csv, element_sets, bins, [rel_id, b, c])

    print(f"\nTotal time: {time.perf_counter() - total_start:.2f} s")

if __name__ == "__main__":
    main()
