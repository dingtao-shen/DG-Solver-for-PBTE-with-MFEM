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
```
- `-m/--mesh`: builtin name or mesh file path. If omitted, config is used.
- `-c/--config`: config yaml path (default `config/config.yaml`).
- `-o/--order`: DG polynomial order.
- `-p/--parallel`: enable parallel mesh/space (MPI build required).
- `-np/--no-parallel`: force serial build even if MPI is available.

### Output / logging
- After mesh + DG space construction, a summary prints to stdout.
- A log file is written to `output/log/mesh_<source>_p<order>_dim<dim>.txt` by default. Pass a custom path via `BuildDGSpace` if embedding.

### Sample meshes provided
- `config/mesh/unit-square.mesh`: simple 2D triangular unit square.
- `config/mesh/unit-cube-hex.mesh`: simple 3D hexahedral unit cube.
