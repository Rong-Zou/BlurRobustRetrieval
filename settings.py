import os 

g_data_dir = "/cluster/project/infk/cvg/students/rzoran/synthetic_data" 
g_distractor_dir = "/cluster/project/infk/cvg/students/rzoran/synthetic_data_distractor"

if not os.path.exists(g_data_dir):
	g_data_dir = "/home/rong/euler_group/synthetic_data" #"/home/denysr/data/ShapeNetCore.v2.zip"
	g_distractor_dir = "/home/rong/euler_group/synthetic_data_distractor"
	
if not os.path.exists(g_data_dir):
	g_data_dir = "/local/home/ronzou/euler/synthetic_data"
	g_distractor_dir = "/local/home/ronzou/euler/synthetic_data_distractor"
