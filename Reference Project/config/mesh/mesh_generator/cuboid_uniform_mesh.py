import os
import sys


def generate_unit_cuboid_tet_mesh(
    output_path: str = None,
    polynomial_order: int = 1,
    show_gui: bool = False,
    nx: int = 10,
    ny: int = 10,
    nz: int = 10,
    width: float = 1.0,
    length: float = 1.0,
    height: float = 1.0,
) -> str:
    """
    Generate a structured tetrahedral mesh for a cuboid [0, 0, 0] -> [width, length, height].

    The construction is:
    - Uniform Cartesian grid with nx×ny×nz voxels.
    - Each voxel is split into 6 tetrahedra using a consistent marching-tetrahedra pattern
      along the body diagonal from (i,j,k) to (i+1,j+1,k+1). This yields a globally
      conforming, regular tetrahedral mesh.

    Physical groups are created for the volume ("volume") and the six boundary faces:
    "Left" (y=0), "Right" (y=length), "Back" (x=0), "Front" (x=width), 
    "Bottom" (z=0), "Top" (z=height).

    Parameters
    ----------
    output_path : str, optional
        Path to write the mesh (.msh, .vtk, .vtu, etc.). If None, auto-generates name.
    polynomial_order : int
        Finite element polynomial order (1=linear, 2=quadratic,...).
    show_gui : bool
        If True, show Gmsh GUI after mesh is created.
    nx, ny, nz : int
        Number of voxels along x, y, z directions.
    width : float
        Width of cuboid along x direction.
    length : float
        Length of cuboid along y direction.
    height : float
        Height of cuboid along z direction.

    Returns
    -------
    str
        Absolute path to the written mesh file.
    """

    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("nx, ny, nz must be positive integers")

    try:
        import gmsh
    except Exception as exc:
        raise RuntimeError(
            "Gmsh Python API is required. Install with: pip install gmsh"
        ) from exc

    initialized_here = False
    try:
        if not gmsh.isInitialized():
            gmsh.initialize()
            initialized_here = True

        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.ElementOrder", int(polynomial_order))
        # Ensure MSH v2.2 ASCII with 8-byte floats (header: "2.2 0 8")
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)

        model = gmsh.model

        # Create one discrete 3D entity (volume)
        vol_tag = 7
        model.addDiscreteEntity(3, vol_tag)

        # Build nodes on a (nx+1) x (ny+1) x (nz+1) grid in [0,width] x [0,length] x [0,height]
        def node_id(i: int, j: int, k: int) -> int:
            return 1 + i + (nx + 1) * (j + (ny + 1) * k)

        coords = []
        node_tags = []
        for k in range(nz + 1):
            z = k * height / nz
            for j in range(ny + 1):
                y = j * length / ny
                for i in range(nx + 1):
                    x = i * width / nx
                    node_tags.append(node_id(i, j, k))
                    coords.extend([x, y, z])

        model.mesh.addNodes(3, vol_tag, nodeTags=node_tags, coord=coords)

        # Marching tetrahedra template for 6 tets per cube around body diagonal 0->6
        # Local vertex order within a voxel:
        # 0:(0,0,0) 1:(1,0,0) 2:(1,1,0) 3:(0,1,0)
        # 4:(0,0,1) 5:(1,0,1) 6:(1,1,1) 7:(0,1,1)
        tet_local = [
            (0, 5, 1, 6),
            (0, 1, 2, 6),
            (0, 2, 3, 6),
            (0, 3, 7, 6),
            (0, 7, 4, 6),
            (0, 4, 5, 6),
        ]

        def voxel_vertex_global(i: int, j: int, k: int, vid: int) -> int:
            di = [0, 1, 1, 0, 0, 1, 1, 0][vid]
            dj = [0, 0, 1, 1, 0, 0, 1, 1][vid]
            dk = [0, 0, 0, 0, 1, 1, 1, 1][vid]
            return node_id(i + di, j + dj, k + dk)

        tet_type = 4  # linear tetrahedron
        tet_tags = []
        tet_conn = []
        elem_counter = 1
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    for a, b, c, d in tet_local:
                        tet_tags.append(elem_counter)
                        tet_conn.extend(
                            [
                                voxel_vertex_global(i, j, k, a),
                                voxel_vertex_global(i, j, k, b),
                                voxel_vertex_global(i, j, k, c),
                                voxel_vertex_global(i, j, k, d),
                            ]
                        )
                        elem_counter += 1

        model.mesh.addElements(3, vol_tag, [tet_type], [tet_tags], [tet_conn])

        # Add boundary surface entities and triangles for physical groups
        # Triangulation consistent with the marching tetrahedra (two triangles per voxel face)
        tri_type = 2

        def add_face_group(name: str, fixed_axis: str, fixed_index: int) -> int:
            tag = {
                "Left": 1,    # y=0
                "Right": 2,   # y=length
                "Back": 3,    # x=0
                "Front": 4,   # x=width
                "Bottom": 5,  # z=0
                "Top": 6,     # z=height
            }[name]
            model.addDiscreteEntity(2, tag)

            tri_tags = []
            tri_conn = []
            etag = 1

            if fixed_axis == "x":
                i = fixed_index
                for k in range(nz):
                    for j in range(ny):
                        # vertices on the face (local order matches comments above)
                        v0 = node_id(i, j, k)
                        v1 = node_id(i, j + 1, k)
                        v2 = node_id(i, j + 1, k + 1)
                        v3 = node_id(i, j, k + 1)
                        if name == "Back":
                            # [0,3,7] and [0,7,4]
                            tri1 = (v0, v1, v2)
                            tri2 = (v0, v2, v3)
                        else:  # Front (x=width) face
                            # [1,2,6] and [1,6,5]
                            # map via opposite face orientation
                            v0 = node_id(i, j, k)
                            v1 = node_id(i, j, k + 1)
                            v2 = node_id(i, j + 1, k + 1)
                            v3 = node_id(i, j + 1, k)
                            tri1 = (v0, v3, v2)
                            tri2 = (v0, v2, v1)
                        tri_tags += [etag, etag + 1]
                        tri_conn += list(tri1) + list(tri2)
                        etag += 2

            elif fixed_axis == "y":
                j = fixed_index
                for k in range(nz):
                    for i in range(nx):
                        # face vertices on y-constant planes
                        if name == "Left":
                            # [0,4,5] and [0,5,1]
                            v0 = node_id(i, j, k)
                            v1 = node_id(i + 1, j, k)
                            v2 = node_id(i + 1, j, k + 1)
                            v3 = node_id(i, j, k + 1)
                            tri1 = (v0, v3, v2)
                            tri2 = (v0, v2, v1)
                        else:  # Right (y=length)
                            # [3,2,6] and [3,6,7]
                            v0 = node_id(i, j, k)
                            v1 = node_id(i, j, k + 1)
                            v2 = node_id(i + 1, j, k + 1)
                            v3 = node_id(i + 1, j, k)
                            tri1 = (v0, v3, v2)
                            tri2 = (v0, v2, v1)
                        tri_tags += [etag, etag + 1]
                        tri_conn += list(tri1) + list(tri2)
                        etag += 2

            elif fixed_axis == "z":
                k = fixed_index
                for j in range(ny):
                    for i in range(nx):
                        if name == "Bottom":
                            # [0,1,2] and [0,2,3]
                            v0 = node_id(i, j, k)
                            v1 = node_id(i + 1, j, k)
                            v2 = node_id(i + 1, j + 1, k)
                            v3 = node_id(i, j + 1, k)
                            tri1 = (v0, v1, v2)
                            tri2 = (v0, v2, v3)
                        else:  # Top (z1)
                            # [4,5,6] and [4,6,7]
                            v0 = node_id(i, j, k)
                            v1 = node_id(i + 1, j, k)
                            v2 = node_id(i + 1, j + 1, k)
                            v3 = node_id(i, j + 1, k)
                            tri1 = (v0, v2, v1)
                            tri2 = (v0, v3, v2)
                        tri_tags += [etag, etag + 1]
                        tri_conn += list(tri1) + list(tri2)
                        etag += 2

            model.mesh.addElements(2, tag, [tri_type], [tri_tags], [tri_conn])
            phys = model.addPhysicalGroup(2, [tag])
            model.setPhysicalName(2, phys, name)
            return tag

        # Create six boundary face groups with requested names
        add_face_group("Left", "y", 0)      # y=0
        add_face_group("Right", "y", ny)    # y=length
        add_face_group("Back", "x", 0)      # x=0
        add_face_group("Front", "x", nx)    # x=width
        add_face_group("Bottom", "z", 0)    # z=0
        add_face_group("Top", "z", nz)      # z=height

        # Volume physical group
        vol_phys = model.addPhysicalGroup(3, [vol_tag])
        model.setPhysicalName(3, vol_phys, "volume")

        # Write mesh
        if output_path is None:
            output_path = f"cuboid_{nx}x{ny}x{nz}.msh"
        
        ext = os.path.splitext(output_path)[1].lower()
        if ext == "":
            output_path = output_path + ".msh"

        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        gmsh.write(output_path)

        if show_gui:
            gmsh.fltk.initialize()
            gmsh.fltk.run()

        return os.path.abspath(output_path)

    finally:
        try:
            import gmsh  # type: ignore
            if 'initialized_here' in locals() and initialized_here and gmsh.isInitialized():
                gmsh.finalize()
        except Exception:
            pass


def _parse_cli_args(argv):
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a tetrahedral mesh of a unit cuboid using Gmsh",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nx", type=int, default=5, help="Number of voxels along x")
    parser.add_argument("--ny", type=int, default=5, help="Number of voxels along y")
    parser.add_argument("--nz", type=int, default=5, help="Number of voxels along z")
    parser.add_argument("--width", type=float, default=1.0, help="Width of cuboid along x direction")
    parser.add_argument("--length", type=float, default=1.0, help="Length of cuboid along y direction")
    parser.add_argument("--height", type=float, default=1.0, help="Height of cuboid along z direction")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output mesh file path (.msh, .vtk, .vtu, etc.). If not provided, auto-generates name.",
    )
    parser.add_argument(
        "-p",
        "--order",
        type=int,
        default=1,
        help="Polynomial order of mesh elements",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Show Gmsh GUI after meshing",
    )

    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_cli_args(argv if argv is not None else sys.argv[1:])
    out = generate_unit_cuboid_tet_mesh(
        output_path=args.output,
        polynomial_order=args.order,
        show_gui=args.gui,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        width=args.width,
        length=args.length,
        height=args.height,
    )
    print(f"Wrote mesh to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


