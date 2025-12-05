# DG-Solver-for-PBTE-with-MFEM

## make & run
```
cmake --build build-asan -j && ./build/DG4PBTE -d 2 -s 2 -nx 32 -ny 32 -p 1 -vg 1.0 -knN 0.2 -knR 0.1 -L 1.0 -T 1.0 -ux 0.0 -maxit 20 -rtol 1e-6 -relax 0.3 -bc -Thot 1.2 -Tcold 0.8 -hot 1 -vtk -o iso_bc_attr_demo
```