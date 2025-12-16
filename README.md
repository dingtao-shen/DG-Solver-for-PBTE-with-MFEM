# DG-Solver-for-PBTE-with-MFEM

## Mesh Loading

### Ways to provide a mesh
- Built-ins: `unit-square`, `unit-square-tri`, `unit-square-quad`, `unit-cube`, `unit-cube-tet`, `unit-cube-hex`.
- External file: MFEM `.mesh` format. Place under `config/mesh/` (recommended).
- Config file: `config/config.yaml` with either
  - block: 
    - `mesh:`  
      - `path: config/mesh/unit-square.mesh`
  - or top-level: `mesh_path: config/mesh/unit-square.mesh`

### Running
```bash
# Use config (default: config/config.yaml)
./build/pbte_demo

# Override with builtin or file
./build/pbte_demo -m unit-cube -o 2
./build/pbte_demo -m config/mesh/unit-cube-hex.mesh -o 1

# Use another config
./build/pbte_demo -c path/to/other.yaml

# Parallel (requires MFEM built with MPI)
mpirun -np 4 ./build/pbte_demo -m unit-cube -o 1 -p
# Force serial even if MPI MFEM is available
./build/pbte_demo -m unit-cube -o 1 -np

# Uniform refinement (applied before building the FE space)
# -r/--refine <levels>
./build/pbte_demo -m unit-cube -o 2 -r 2
```
- `-m/--mesh`: builtin name or mesh file path. If omitted, config is used.
- `-c/--config`: config yaml path (default `config/config.yaml`).
- `-o/--order`: DG polynomial order.
- `-p/--parallel`: enable parallel mesh/space (MPI build required).
- `-np/--no-parallel`: force serial build even if MPI is available.
- `-r/--refine`: uniform refinement levels (integer, 0 means no refinement).

### Output / logging
- After mesh + DG space construction, a summary prints to stdout.
- A log file is written to `output/log/mesh_<source>_p<order>_dim<dim>.txt` by default. Pass a custom path via `BuildDGSpace` if embedding.

### Sample meshes provided
- `config/mesh/unit-square.mesh`: simple 2D triangular unit square.
- `config/mesh/unit-cube-hex.mesh`: simple 3D hexahedral unit cube.

### Parallel build and run (MPI)
- Ensure MFEM is built with MPI; point `-DMFEM_DIR` to the MPI-enabled prefix.
- Example clean rebuild and run (4 ranks, order 2, with two levels of uniform refinement):
```bash
rm -rf build
cmake -B build -S . -DMFEM_DIR=/path/to/mfem-mpi
cmake --build build -j
mpirun -np 4 ./build/pbte_demo -p -o 2 -r 2
```
Only rank 0 prints to stdout and writes `output/log/integrals_all.txt`; all ranks contribute their local integrals to that file.
