import os  

g_shapenet_path = "/cluster/project/infk/cvg/students/rzoran/datasets/ShapeNetCore.v2" 
g_texture_path = "/cluster/project/infk/cvg/students/rzoran/datasets/dtd/images"
g_blender_path = r"/cluster/project/infk/cvg/students/rzoran/blender-2.91.0-linux64/blender"
g_background_path = "/cluster/project/infk/cvg/students/rzoran/datasets/LHQ/lhq_256" 
g_outdata_path = "/cluster/project/infk/cvg/students/rzoran/debug"

if not os.path.exists(g_shapenet_path):
	g_shapenet_path = "/home/rong/euler_group/datasets/ShapeNetCore.v2" 
	g_texture_path = "/home/rong/euler_group/datasets/dtd/images"
	g_blender_path = r"/home/rong/euler_group/blender-2.91.0-linux64/blender"
	g_background_path = "/home/rong/euler_group/datasets/LHQ/lhq_256"
	g_outdata_path = "/home/rong/euler_group/debug"
 
if not os.path.exists(g_shapenet_path):
	g_shapenet_path = "/local/home/ronzou/euler/datasets/ShapeNetCore.v2" 
	g_texture_path = "/local/home/ronzou/euler/datasets/dtd/images"
	g_blender_path = r"/local/home/ronzou/euler/blender-2.91.0-linux64/blender" 
	g_background_path = "/local/home/ronzou/euler/datasets/LHQ/lhq_256"
	g_outdata_path = "/local/home/ronzou/euler/debug"
