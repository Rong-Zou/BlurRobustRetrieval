import os
import sys
import json
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from settings import *
import numpy as np
from utils import NoIndent, NoIndentWriter


with open(os.path.join(g_shapenet_path, "taxonomy.json"), "r") as shapenet_file:
    shapenet_taxonomy = json.load(shapenet_file)
    
    
class_dirs = []
for file in sorted(os.listdir(g_shapenet_path)):
    d = os.path.join(g_shapenet_path, file)
    if os.path.isdir(d):
        class_dirs.append(d)
        
assert len(class_dirs) == 55

synsetIds = []

num_instances = []
num_instances_with_textures = []
# save the number of instances and the instance names in a txt file for each class
for class_dir in class_dirs:
    synsetId = os.path.basename(class_dir)
    print(synsetId)
    synsetIds.append(synsetId)
    instance_names = []
    instance_with_textures_names = []
    for file in sorted(os.listdir(class_dir)):
        d = os.path.join(class_dir, file)
        if os.path.isdir(d):
            instance_names.append(file)
            if os.path.isdir(os.path.join(d, "images")):
                instance_with_textures_names.append(file)    
            
    dict = {
        "numInstances": len(instance_names),
        "instanceNames": instance_names
    }
    
    dict_w_tex = {
        "numInstancesWithTextures": len(instance_with_textures_names),
        "ratio": len(instance_with_textures_names) / len(instance_names),
        "instanceWithTexturesNames": instance_with_textures_names
    }
    
    num_instances.append(len(instance_names))
    num_instances_with_textures.append(len(instance_with_textures_names))
    # # save the number of instances and the instance names in a json file for each class
    with open(os.path.join(class_dir, "instances.json"), "w") as instance_file:
        json.dump(dict, instance_file, indent=4)
    with open(os.path.join(class_dir, "instances_with_textures.json"), "w") as instance_file:
        json.dump(dict_w_tex, instance_file, indent=4)

# for each synsetId, find in the shapeNet taxonomy the synsetId and the name of the synset
synsetId_names = []
for synsetId in synsetIds:
    for synset in shapenet_taxonomy:
        if synset["synsetId"] == synsetId:
            print(f"synsetId: {synsetId}, name: {synset['name']}")
            synsetId_names.append((synsetId, synset["name"]))
            break
        
num_instances = np.array(num_instances)
num_instances_with_textures = np.array(num_instances_with_textures)
num_instances_without_textures = num_instances - num_instances_with_textures
ratio = num_instances_with_textures / num_instances

# for each category, save the synsetID, the name, the number of instances, the number of instances with and without textures, the ratio into a dictionary
# then combine all dictionaries into a large dictionary, and save it as a json file
dict = {}
for i in range(len(synsetIds)):
    dict[synsetIds[i]] = {
        "name": synsetId_names[i][1],
        "numInstances": int(num_instances[i]),
        "numInstancesWithTextures": int(num_instances_with_textures[i]),
        "numInstancesWithoutTextures": int(num_instances_without_textures[i]),
        "ratio": float(ratio[i])
    }

with open(os.path.join(g_shapenet_path, "instances_with_textures_info.json"), "w") as instance_file:
    json.dump(dict, instance_file, indent=4)


instance_texture_info = {
    "sunsetIdNames": NoIndent(synsetId_names),
    "numInstances": NoIndent(num_instances.tolist()),
    "numInstancesWithTextures": NoIndent(num_instances_with_textures.tolist()),
    "numInstancesWithoutTextures": NoIndent(num_instances_without_textures.tolist()),
    "withTextureRatio": NoIndent(ratio.tolist())
}

# save the instance_texture_info in a json file
with open(os.path.join(g_shapenet_path, "instance_texture_info_summary.json"), "w") as fp:
    json.dump(instance_texture_info, fp, cls=NoIndentWriter, sort_keys=True, indent=4)
    fp.write('\n')  
