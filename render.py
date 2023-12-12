import os
import sys
import random
import tempfile
from PIL import Image
import PIL 
print(PIL.__version__)
import numpy as np
import random
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import blur_filter as filter
from utils import *

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

seed = 42
random.seed(seed)

np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
"""background_main_tar = './datasets/ILSVRC2012_img_train.tar'
# load json file to get the background subtar names
with open('./datasets/imageNetBackgroundClasses.json') as f:
    background_subtars = json.load(f)

num_background_classes = len(background_subtars)"""


def ensure_blender(blender=None):
    """Ensures that the script is running with Blender.
    If Blender is not running, the script will be restarted with Blender.
    
    Args:
        blender: Path to Blender executable. If None, the script will be restarted with Blender.
        
        Returns: List of arguments passed to the script. 
        Command line arguments after "--" if Blender is running, otherwise None.
    
    """
    try:
        global bpy
        import bpy

        return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    except ImportError:
        print("No bpy")
        if blender:
            import subprocess

            print("Restarting with Blender...")
            sys.stdout.flush()
            sys.stderr.flush()
            subprocess.run([blender, "--background", "-noaudio", "--python", sys.argv[0], "--"] + sys.argv[1:], check=True)
            sys.exit()
        else:
            sys.exit("Failed to import bpy. Please run with Blender.")


def init(frames, resolution, mblur=40, env_light=(0.5, 0.5, 0.5)):
    """Initializes the Blender scene.
    
    Args: 
        frames: Number of frames to render.
        resolution: Resolution of the rendered images.
        mblur: Number of motion blur steps.
        env_light: Color of the environment light.
        
    """
    ensure_blender()
    scene = bpy.context.scene

    # output settings
    scene.frame_start = scene.frame_current = 0
    scene.frame_end = frames - 1
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.film_transparent = True

    # canonical scene
    scene.camera.location = 0, 0, 0
    scene.camera.rotation_euler = 0, 0, 0
    # scene.objects["Light"].location = 0, 0, 0
    # scene.objects["Light"].data.energy = 0
    
    # environment lighting
    bpy.context.scene.world.use_nodes = False
    bpy.context.scene.world.color = env_light
    for area in bpy.context.screen.areas: 
        if area.type == 'VIEW_3D':
            space = area.spaces.active
            if space.type == 'VIEW_3D':
                space.shading.type = 'MATERIAL'
    # remove default cube
    bpy.ops.object.delete()
    # bpy.data.objects.remove(scene.objects["Light"])

    # create material for texture
    mat = bpy.data.materials.new("Texture")
    mat.use_nodes = True
    mat.use_fake_user = True
    nodes = mat.node_tree.nodes
    tex = nodes.new("ShaderNodeTexImage")
    mat.node_tree.links.new(tex.outputs[0], nodes["Principled BSDF"].inputs[0])

    # motion blur parameters
    # bpy.context.scene.eevee.motion_blur_
    #                                     depth_scale
    #                                     max:      Maximum blur distance a pixel can spread over. A value of 0 will disable the camara blur and only use the object motion blur.
    #                                     position: Controls at what point the shutter opens in relation to the current frame.
    #                                     shutter:  Time (in frames) taken between shutter open and close.
    #                                     steps:    Controls the number of steps used by the object accumulation blur and thus its accuracy. More steps means longer render time.

    
    scene.eevee.motion_blur_position = "START"  
    # Start on Frame: Shutter is starting to open at the current frame.
    # Center on Frame: Shutter is fully opened at the current frame.
    # End on Frame: Shutter is fully closed at the current frame.
    
    scene.eevee.motion_blur_steps = mblur # in each render, splits the render into multiple time steps and accumulates the result
    # scene.render.image_settings.color_mode = 'RGBA'
    # bpy.context.scene.render.image_settings.file_format = 'PNG'
    # bpy.context.scene.render.image_settings.color_mode = 'RGBA'

def crop_image(orig_img, cropped_min_x, cropped_max_x, cropped_min_y, cropped_max_y):
    '''Crops an image object of type <class 'bpy.types.Image'>.  For example, for a 10x10 image, 
    if you put cropped_min_x = 2 and cropped_max_x = 6,
    you would get back a cropped image with width 4, and 
    pixels ranging from the 2 to 5 in the x-coordinate

    Note: here y increasing as you down the image.  So, 
    if cropped_min_x and cropped_min_y are both zero, 
    you'll get the top-left of the image (as in GIMP).

    Returns: An image of type  <class 'bpy.types.Image'>
    '''

    num_channels=orig_img.channels
    #calculate cropped image size
    cropped_size_x = cropped_max_x - cropped_min_x
    cropped_size_y = cropped_max_y - cropped_min_y
    #original image size
    orig_size_x = orig_img.size[0]
    orig_size_y = orig_img.size[1]

    cropped_img = bpy.data.images.new(name="cropped_img", width=cropped_size_x, height=cropped_size_y)

    print("Exctracting image fragment, this could take a while...")

    #loop through each row of the cropped image grabbing the appropriate pixels from original
    #the reason for the strange limits is because of the 
    #order that Blender puts pixels into a 1-D array.
    current_cropped_row = 0
    for yy in range(orig_size_y - cropped_max_y, orig_size_y - cropped_min_y):
        #the index we start at for copying this row of pixels from the original image
        orig_start_index = (cropped_min_x + yy*orig_size_x) * num_channels
        #and to know where to stop we add the amount of pixels we must copy
        orig_end_index = orig_start_index + (cropped_size_x * num_channels)
        #the index we start at for the cropped image
        cropped_start_index = (current_cropped_row * cropped_size_x) * num_channels 
        cropped_end_index = cropped_start_index + (cropped_size_x * num_channels)

        #copy over pixels 
        cropped_img.pixels[cropped_start_index : cropped_end_index] = orig_img.pixels[orig_start_index : orig_end_index]

        #move to the next row before restarting loop
        current_cropped_row += 1

    return cropped_img

def render_same_bg(
    out_dir, 
    obj, tex_path, bg_paths, 
    loc, rot, 
    blurs=[(0, -1)], 
    crop_tex=True, 
    use_filter=True, 
    render_subframe_step=4, 
    filter_params=None, 
    logger=None, 
    bg_path_flag = None,
    num_frames=48):
    
    ensure_blender()    
    
    scene = bpy.context.scene

    # load object
    bpy.ops.import_scene.obj(filepath=obj)
    
    obj = bpy.context.selected_objects[0]
    
    cam = bpy.context.scene.camera
        
    cam.data.show_background_images = True
    
    bg = cam.data.background_images.new()
    bpy.context.scene.render.film_transparent = True
    
    # cmpositing
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    

    for node in tree.nodes:
        tree.nodes.remove(node)

    image_node = tree.nodes.new(type="CompositorNodeImage")
    
    scale_node = tree.nodes.new(type="CompositorNodeScale")
    bpy.data.scenes["Scene"].node_tree.nodes["Scale"].space = 'RENDER_SIZE'
    bpy.data.scenes["Scene"].node_tree.nodes["Scale"].frame_method = 'CROP'

    
    render_layers_node = tree.nodes.new('CompositorNodeRLayers')   
    
    alpha_over_node = tree.nodes.new(type="CompositorNodeAlphaOver")

    compositor_node = tree.nodes.new('CompositorNodeComposite')   
    
    links = tree.links
    link1 = links.new(image_node.outputs[0], scale_node.inputs[0])
    link2 = links.new(scale_node.outputs[0], alpha_over_node.inputs[1])
    link3 = links.new(render_layers_node.outputs[0], alpha_over_node.inputs[2])
    link4 = links.new(alpha_over_node.outputs[0], compositor_node.inputs[0])

    # load texture
    if tex_path:
        tex = bpy.data.images.load(os.path.abspath(tex_path))

        if crop_tex:
            res = 32

            # random crop the texture
            # cx = random.randint(0,tex.size[0]-res-1)
            # cy = random.randint(0,tex.size[1]-res-1)
            # tex = crop_image(tex, cx, cx+res, cy, cy+res)

            # center crop the texture
            cx = (tex.size[0] - res) // 2
            cy = (tex.size[1] - res) // 2
            tex = crop_image(tex, cx, cx+res, cy, cy+res)

            
        mat = bpy.data.materials["Texture"]
        mat.node_tree.nodes["Image Texture"].image = tex
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.editmode_toggle()
        bpy.ops.uv.cube_project(scale_to_bounds=True)
        bpy.ops.object.editmode_toggle()
        
    # starting position
    obj.location = loc[0]
    obj.rotation_euler = rot[0]
    # Change scale of object
    obj.scale = [3]*3 # [random.randint(1, 4)]*3
    obj.keyframe_insert("location")
    obj.keyframe_insert("rotation_euler")
    obj.keyframe_insert("scale")
    
    # final position
    obj.location = loc[1]
    obj.rotation_euler = rot[1]
    obj.keyframe_insert("location", frame=scene.frame_end)
    obj.keyframe_insert("rotation_euler", frame=scene.frame_end)

    # linear movement
    for f in obj.animation_data.action.fcurves:
        for k in f.keyframe_points:
            k.interpolation = "LINEAR"

    scene.eevee.use_motion_blur = True
    
    seq_valid = True
    img_valid = True 
    info = {}
    
    if use_filter:
        total_frames = scene.frame_end + 1
        # from 0 to total_frames - 1, generate index every 4 frames
        subframe_idx = np.arange(0, total_frames, render_subframe_step)
        
        # render subframes and store in a temporary directory
        with tempfile.TemporaryDirectory() as tmp:
            for idx in subframe_idx:
                scene.frame_current = idx
                scene.eevee.motion_blur_shutter = 0
                scene.render.filepath = os.path.join(tmp, "{}_subframe".format(idx))
                bpy.ops.render.render(write_still=True)
            
            # filter
            max_occ, min_occ, max_subf, min_subf, seq_valid = filter.obj_occ_judger(tmp, filter_params, logger)
            
    blur_infos = {}   
    bg_path_flag -= 1
    if seq_valid:

        clear_frame = num_frames // 2
        blurs = []

        for i in range(1, num_frames // 2): # i is from 1 to num_frames, i means the number of frames to use for the blur
            # when i is odd, append (clear_frame - i // 2, clear_frame + i // 2) to blurs
            # when i is even, append (clear_frame - i // 2, clear_frame + i // 2 - 1) to blurs
            if i % 2 == 1:
                blurs.append((clear_frame - i // 2, clear_frame + i // 2))
            else:
                blurs.append((clear_frame - i // 2, clear_frame + i // 2 - 1))
        
        W = bpy.context.scene.render.resolution_x
        H = bpy.context.scene.render.resolution_y
        clear_frame_alpha = None
        
        bg_img = Image.open(bg_paths).convert('RGB')
        size_of_bg_img = bg_img.size
        
        bg_img = bg_img.crop((size_of_bg_img[0] // 3 * (bg_path_flag % 3), 
                                size_of_bg_img[1] // 3 * (bg_path_flag // 3), 
                                size_of_bg_img[0] // 3 * (bg_path_flag % 3 + 1), 
                                size_of_bg_img[1] // 3 * (bg_path_flag // 3 + 1)))

        bg_img = np.array(bg_img.resize((W, H)))
        
        current_blur_render = 0
        for i, blur in enumerate(blurs):
            current_blur_render = i
            blur_start, blur_end = [b % (scene.frame_end + 1) for b in blur]
            scene.frame_current = blur_start
            scene.eevee.motion_blur_shutter = blur_end - blur_start # Time (in frames) taken between shutter open and close.
            scene.render.filepath = os.path.join(out_dir, f"{current_blur_render}_blurred")

            bpy.ops.render.render(write_still=True) # no background image

            blurred = np.array(Image.open(os.path.join(out_dir, f"{current_blur_render}_blurred.png")))
            
            alpha = blurred[:, :, 3]
            if i == 0:
                # clear_frame_alpha = average_alpha, value is between 0 to 1
                clear_frame_alpha = alpha[alpha > 0].mean() / 255
                
            current_blur_render = i + 1

            manual_composite = np.zeros_like(blurred)
            manual_composite[:, :, :3] = blurred[:, :, :3] * (alpha / 255)[:, :, np.newaxis] + bg_img[:, :, :3] * (1 - alpha / 255)[:, :, np.newaxis]
            manual_composite[:, :, 3] = alpha
                    
            info, diff_img = filter.get_info(manual_composite, bg_img, clear_frame_alpha=clear_frame_alpha)
            
            info['frames'] = blur
            
            blur_infos[current_blur_render-1] = info

            # save the manual_composite in out_dir
            manual_composite = Image.fromarray(manual_composite[:, :, :3])
            manual_composite.save(os.path.join(out_dir, f"{current_blur_render-1}_rendered.png"))
            
    scene.frame_current = scene.frame_start
    # clean up
    bpy.ops.object.delete()
    for collection in [bpy.data.meshes, bpy.data.materials, bpy.data.images]:
        for block in collection:
            if not block.users:
                collection.remove(block)
    
    return seq_valid, blur_infos


def render_diff_bg(
    out_dir, 
    obj, tex_path, bg_paths, 
    loc, rot, 
    blurs=[(0, -1)],
    crop_tex=True, 
    use_filter=True, 
    render_subframe_step=4, 
    filter_params=None, 
    logger=None,  
    bg_path_flag = None, 
    num_frames=48):

    ensure_blender()    
    
    scene = bpy.context.scene

    # load object
    bpy.ops.import_scene.obj(filepath=obj)
    
    obj = bpy.context.selected_objects[0]
    
    cam = bpy.context.scene.camera
        
    cam.data.show_background_images = True
    
    bg = cam.data.background_images.new()
    bpy.context.scene.render.film_transparent = True
    
    # cmpositing
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    

    for node in tree.nodes:
        tree.nodes.remove(node)

    image_node = tree.nodes.new(type="CompositorNodeImage")
    
    scale_node = tree.nodes.new(type="CompositorNodeScale")
    bpy.data.scenes["Scene"].node_tree.nodes["Scale"].space = 'RENDER_SIZE'
    bpy.data.scenes["Scene"].node_tree.nodes["Scale"].frame_method = 'CROP'

    
    render_layers_node = tree.nodes.new('CompositorNodeRLayers')   
    
    alpha_over_node = tree.nodes.new(type="CompositorNodeAlphaOver")

    compositor_node = tree.nodes.new('CompositorNodeComposite')   
    
    links = tree.links
    link1 = links.new(image_node.outputs[0], scale_node.inputs[0])
    link2 = links.new(scale_node.outputs[0], alpha_over_node.inputs[1])
    link3 = links.new(render_layers_node.outputs[0], alpha_over_node.inputs[2])
    link4 = links.new(alpha_over_node.outputs[0], compositor_node.inputs[0])

    # load texture
    if tex_path:
        tex = bpy.data.images.load(os.path.abspath(tex_path))

        if crop_tex:
            res = 32

            # random crop the texture
            # cx = random.randint(0,tex.size[0]-res-1)
            # cy = random.randint(0,tex.size[1]-res-1)
            # tex = crop_image(tex, cx, cx+res, cy, cy+res)

            # center crop the texture
            cx = (tex.size[0] - res) // 2
            cy = (tex.size[1] - res) // 2
            tex = crop_image(tex, cx, cx+res, cy, cy+res)

            
        mat = bpy.data.materials["Texture"]
        mat.node_tree.nodes["Image Texture"].image = tex
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.editmode_toggle()
        bpy.ops.uv.cube_project(scale_to_bounds=True)
        bpy.ops.object.editmode_toggle()
        
        
    # starting position
    obj.location = loc[0]
    obj.rotation_euler = rot[0]
    # Change scale of object
    obj.scale = [3]*3 # [random.randint(1, 4)]*3
    obj.keyframe_insert("location")
    obj.keyframe_insert("rotation_euler")
    obj.keyframe_insert("scale")
    
    # final position
    obj.location = loc[1]
    obj.rotation_euler = rot[1]
    obj.keyframe_insert("location", frame=scene.frame_end)
    obj.keyframe_insert("rotation_euler", frame=scene.frame_end)

    # linear movement
    for f in obj.animation_data.action.fcurves:
        for k in f.keyframe_points:
            k.interpolation = "LINEAR"

    scene.eevee.use_motion_blur = True
    
    seq_valid = True
    img_valid = True 
    info = {}
    
    if use_filter:
        total_frames = scene.frame_end + 1
        # from 0 to total_frames - 1, generate index every 4 frames
        subframe_idx = np.arange(0, total_frames, render_subframe_step)
        
        # render subframes and store in a temporary directory
        with tempfile.TemporaryDirectory() as tmp:
            for idx in subframe_idx:
                scene.frame_current = idx
                scene.eevee.motion_blur_shutter = 0
                scene.render.filepath = os.path.join(tmp, "{}_subframe".format(idx))
                bpy.ops.render.render(write_still=True)
            
            # filter
            max_occ, min_occ, max_subf, min_subf, seq_valid = filter.obj_occ_judger(tmp, filter_params, logger)
            
    blur_infos = {}   
    bg_path_flag -= 1
    if seq_valid:

        clear_frame = num_frames // 2
        blurs = []

        for i in range(1, num_frames // 2): # i is from 1 to num_frames, i means the number of frames to use for the blur
            # when i is odd, append (clear_frame - i // 2, clear_frame + i // 2) to blurs
            # when i is even, append (clear_frame - i // 2, clear_frame + i // 2 - 1) to blurs
            if i % 2 == 1:
                blurs.append((clear_frame - i // 2, clear_frame + i // 2))
            else:
                blurs.append((clear_frame - i // 2, clear_frame + i // 2 - 1))
        
        W = bpy.context.scene.render.resolution_x
        H = bpy.context.scene.render.resolution_y
        clear_frame_alpha = None

        current_blur_render = 0
        for i, blur in enumerate(blurs):
            
            current_blur_render = i
            blur_start, blur_end = [b % (scene.frame_end + 1) for b in blur]
            scene.frame_current = blur_start
            scene.eevee.motion_blur_shutter = blur_end - blur_start # Time (in frames) taken between shutter open and close.
            scene.render.filepath = os.path.join(out_dir, f"{current_blur_render}_blurred")

            bpy.ops.render.render(write_still=True) # no background image

            blurred = np.array(Image.open(os.path.join(out_dir, f"{current_blur_render}_blurred.png")))
            
            alpha = blurred[:, :, 3]
            if i == 0:
                # clear_frame_alpha = average_alpha, value is between 0 to 1
                clear_frame_alpha = alpha[alpha > 0].mean() / 255
            current_blur_render = i + 1

            bg_img = Image.open(bg_paths[i]).convert('RGB')
            size_of_bg_img = bg_img.size
            
            bg_img = bg_img.crop((size_of_bg_img[0] // 3 * (bg_path_flag % 3), 
                                  size_of_bg_img[1] // 3 * (bg_path_flag // 3), 
                                  size_of_bg_img[0] // 3 * (bg_path_flag % 3 + 1), 
                                  size_of_bg_img[1] // 3 * (bg_path_flag // 3 + 1)))

            bg_img = np.array(bg_img.resize((W, H)))

            manual_composite = np.zeros_like(blurred)
            manual_composite[:, :, :3] = blurred[:, :, :3] * (alpha / 255)[:, :, np.newaxis] + bg_img[:, :, :3] * (1 - alpha / 255)[:, :, np.newaxis]
            manual_composite[:, :, 3] = alpha
                    
            info, _ = filter.get_info(manual_composite, bg_img, clear_frame_alpha=clear_frame_alpha)

            info['frames'] = blur
            
            blur_infos[current_blur_render-1] = info

            # save the manual_composite in out_dir
            manual_composite = Image.fromarray(manual_composite[:, :, :3])
            manual_composite.save(os.path.join(out_dir, f"{current_blur_render-1}_rendered.png"))
            

    scene.frame_current = scene.frame_start
    # clean up
    bpy.ops.object.delete()
    for collection in [bpy.data.meshes, bpy.data.materials, bpy.data.images]:
        for block in collection:
            if not block.users:
                collection.remove(block)
        
    return seq_valid, blur_infos

class Frustum:

    def __init__(
        self, z_range, resolution, max_radius=1.0, dead_zone=0.05, focal_length=50, sensor_size=36
    ):

        self.tan = (1 - dead_zone) * sensor_size / focal_length / 2 # tan of half of the FOV
        self.offset = max_radius / self.tan * (self.tan ** 2 + 1) ** 0.5 #
        self.ratio = resolution[1] / resolution[0]
        self.z_range = z_range

    def gen_point(self, z=None):
        """Generates a random point in the frustum.
        
        Args:
            z: Z value of the point. If None, a random z value will be chosen.
                
        """
        if z is None:
            z_min, z_max = self.z_range
            z = z_min + random.random() * (z_max - z_min)
        x = (random.random() * 2 - 1) * self.tan * (z + self.offset)
        y = (random.random() * 2 - 1) * self.tan * (z * self.ratio + self.offset)
        return x, y, z

    def gen_point_near(self, point, max_delta_z, delta_xy_range):
        """Generates a random point near a given point.
        
        Args:
            point: Point near which to generate a random point.
            max_delta_z: Maximum delta z value.
            delta_xy_range: Range of delta x and y values.
                
        """
        x, y, z = point
        dxy_min, dxy_max = min(delta_xy_range) ** 2, max(delta_xy_range) ** 2
        for _ in range(100000):
            x2, y2, z2 = self.gen_point(z + max_delta_z * (random.random() * 2 - 1))
            if dxy_min <= (x - x2) ** 2 + (y - y2) ** 2 <= dxy_max:
                return x2, y2, z2
        raise ValueError("Failed to generate point in given range. Check input parameters.")

    def gen_point_pair(self, max_delta_z, delta_xy_range):
        """Generates a random point pair.
        
        Args:
            max_delta_z: Maximum delta z value.
            delta_xy_range: Range of delta x and y values.
                
        """
        for _ in range(100):
            a = self.gen_point()
            try:
                b = self.gen_point_near(a, max_delta_z, delta_xy_range)
            except ValueError:
                continue
            return a, b
        # raise ValueError("Failed to generate point pair. Check input parameters.")
        return None