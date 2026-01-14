import gmsh
import sys

def generate_cuboid_mesh(length, width, height, mesh_size=1.0):
    """
    Generate a cuboid mesh using Gmsh API.
    
    Args:
        length (float): Length of the cuboid along x-axis
        width (float): Width of the cuboid along y-axis
        height (float): Height of the cuboid along z-axis
        mesh_size (float): Characteristic length of mesh elements
    
    Returns:
        None: The mesh is saved to 'cuboid.msh'
    """
    # Initialize Gmsh
    gmsh.initialize()
    
    # Create a new model
    gmsh.model.add("cuboid")
    
    # Create the cuboid vertices
    p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)
    p2 = gmsh.model.geo.addPoint(length, 0, 0, mesh_size)
    p3 = gmsh.model.geo.addPoint(length, width, 0, mesh_size)
    p4 = gmsh.model.geo.addPoint(0, width, 0, mesh_size)
    p5 = gmsh.model.geo.addPoint(0, 0, height, mesh_size)
    p6 = gmsh.model.geo.addPoint(length, 0, height, mesh_size)
    p7 = gmsh.model.geo.addPoint(length, width, height, mesh_size)
    p8 = gmsh.model.geo.addPoint(0, width, height, mesh_size)
    
    # Create the edges
    # Bottom face
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    
    # Top face
    l5 = gmsh.model.geo.addLine(p5, p6)
    l6 = gmsh.model.geo.addLine(p6, p7)
    l7 = gmsh.model.geo.addLine(p7, p8)
    l8 = gmsh.model.geo.addLine(p8, p5)
    
    # Vertical edges
    l9 = gmsh.model.geo.addLine(p1, p5)
    l10 = gmsh.model.geo.addLine(p2, p6)
    l11 = gmsh.model.geo.addLine(p3, p7)
    l12 = gmsh.model.geo.addLine(p4, p8)
    
    # Create the faces
    # Bottom face
    cl1 = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    bottom = gmsh.model.geo.addPlaneSurface([cl1])
    
    # Top face
    cl2 = gmsh.model.geo.addCurveLoop([l5, l6, l7, l8])
    top = gmsh.model.geo.addPlaneSurface([cl2])
    
    # Front face
    cl3 = gmsh.model.geo.addCurveLoop([l1, l10, -l5, -l9])
    front = gmsh.model.geo.addPlaneSurface([cl3])
    
    # Right face
    cl4 = gmsh.model.geo.addCurveLoop([l2, l11, -l6, -l10])
    right = gmsh.model.geo.addPlaneSurface([cl4])
    
    # Back face
    cl5 = gmsh.model.geo.addCurveLoop([l3, l12, -l7, -l11])
    back = gmsh.model.geo.addPlaneSurface([cl5])
    
    # Left face
    cl6 = gmsh.model.geo.addCurveLoop([l4, l9, -l8, -l12])
    left = gmsh.model.geo.addPlaneSurface([cl6])
    
    # Create the volume
    sl = gmsh.model.geo.addSurfaceLoop([bottom, top, front, right, back, left])
    volume = gmsh.model.geo.addVolume([sl])
    
    # Synchronize the model
    gmsh.model.geo.synchronize()
    
    # Generate the mesh
    gmsh.model.mesh.generate(3)
    
    # Save the mesh
    gmsh.write("./config/mesh/generator/cuboid_medium.msh")
    
    # Finalize Gmsh
    gmsh.finalize()

if __name__ == "__main__":
    # Example usage
    length = 1.0
    width = 1.0
    height = 1.0
    mesh_size = 0.2
    
    generate_cuboid_mesh(length, width, height, mesh_size)
    print(f"Mesh generated successfully with dimensions: {length}x{width}x{height}") 