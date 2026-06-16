from build123d import Box, export_stl

cube = Box(20, 20, 20)
export_stl(cube, "cube.stl")
print(f"wrote cube.stl  bbox={cube.bounding_box()}  volume={cube.volume:.1f} mm^3")
