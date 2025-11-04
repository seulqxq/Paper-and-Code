# blender_split_render.py
import bpy, os, mathutils, argparse, sys, glob, json
import numpy as np
from PIL import Image
from mathutils import Vector
import math

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_glb', type=str, default="/home/seulxia/project/PartGen/4dcloth/model/s10/scans/00_000001_uv/00_000001_uv.obj",
                        help='Input 3D model file path or directory (supports .obj, .glb, .gltf formats)')
    parser.add_argument('--out_dir', type=str, default="/home/seulxia/project/PartGen/output",
                        help='Output directory for rendered images and metadata')
    parser.add_argument('--batch', action='store_true', help='Batch process all .obj files in subdirectories')
    parser.add_argument('--res', type=int, default=512, help='Render resolution for both width and height')
    parser.add_argument('--num_views', type=int, default=16, help='Number of Fibonacci-sampled views per part')
    parser.add_argument('--min_part_ratio', type=float, default=0.01, help='Cull parts with volume ratio below this (e.g., 0.05 = 5%)')
    parser.add_argument('--max_parts', type=int, default=20, help='Skip assets with more than this number of parts')
    
    # In Blender, arguments after '--' are meant for the script
    # Extract only those arguments
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
        print(f"DEBUG: Parsing script arguments: {argv}")
    else:
        argv = []
        print("DEBUG: No '--' found, using default arguments")
    
    args = parser.parse_args(argv)
    
    # Set fixed parameters (not configurable via command line)
    args.use_loose_parts = False
    args.use_camera_headlight = False
    args.headlight_strength = 8.0

    print(f"DEBUG: Args parsed: in_glb={args.in_glb}, out_dir={args.out_dir}, batch={args.batch}")
    print(f"DEBUG: Fixed params: use_loose_parts={args.use_loose_parts}")
    return args

def initialize_environment(args):
    """初始化环境和输出目录"""
    IN = args.in_glb
    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)
    return IN, OUT_DIR

def import_glb_and_separate_parts(in_glb_path, use_loose_parts=False):
    """导入GLB/OBJ文件并分离部件"""
    # Clear scene
    print(f"in glb path: {in_glb_path}")
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import model based on file extension
    file_ext = os.path.splitext(in_glb_path)[1].lower()
    if file_ext in ['.obj']:
        print(f"Importing OBJ file: {in_glb_path}")
        bpy.ops.import_scene.obj(filepath=in_glb_path)
    elif file_ext in ['.glb', '.gltf']:
        print(f"Importing GLB/GLTF file: {in_glb_path}")
        bpy.ops.import_scene.gltf(filepath=in_glb_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: .obj, .glb, .gltf")

    if use_loose_parts:
        # Optional fallback: separate connected components
        for obj in list(bpy.context.scene.objects):
            if obj.type != 'MESH':
                continue
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.separate(type='LOOSE')
            bpy.ops.object.mode_set(mode='OBJECT')
            obj.select_set(False)

    # Now all mesh objects are separate loose parts
    all_mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']  # 保存所有mesh对象的引用
    mesh_objs = all_mesh_objs.copy()  # 工作副本，将被过滤
    print("Parts count:", len(mesh_objs))
    
    # 保存原始材质
    original_materials = {}
    for obj in all_mesh_objs:
        # 保存对象的所有材质
        original_materials[obj.name] = [mat for mat in obj.data.materials if mat is not None]
    
    return all_mesh_objs, mesh_objs, original_materials

def mesh_object_volume(obj):
    """
    基于包围盒计算网格对象的体积
    这是最简单、最可靠的方法，适用于组件大小比较
    """
    try:
        # 获取对象的世界空间包围盒
        deps = bpy.context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(deps)
        
        # 计算世界空间的包围盒角点
        bbox_corners = [eval_obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
        
        # 找到包围盒的最小和最大坐标
        min_x = min(v.x for v in bbox_corners)
        max_x = max(v.x for v in bbox_corners)
        min_y = min(v.y for v in bbox_corners)
        max_y = max(v.y for v in bbox_corners)
        min_z = min(v.z for v in bbox_corners)
        max_z = max(v.z for v in bbox_corners)
        
        # 计算包围盒的尺寸
        width = abs(max_x - min_x)
        height = abs(max_y - min_y)
        depth = abs(max_z - min_z)
        
        # 包围盒体积
        bbox_volume = width * height * depth
        
        return float(max(bbox_volume, 0.0))
        
    except Exception as e:
        print(f"包围盒体积计算失败 {obj.name}: {e}")
        # 备用方法：使用本地包围盒
        try:
            bbox_volume = 1.0
            for i in range(3):
                dimension = abs(max(v[i] for v in obj.bound_box) - min(v[i] for v in obj.bound_box))
                bbox_volume *= dimension
            return float(max(bbox_volume, 0.0))
        except:
            return 0.0

# 提供备选的简单体积估算方法（基于边界框和几何复杂度）
def simple_volume_estimate(obj):
    """
    简单但可靠的体积估算方法，基于边界框和几何复杂度
    """
    try:
        # 计算边界框体积
        bbox_volume = 1.0
        for i in range(3):
            bbox_volume *= abs(max(v[i] for v in obj.bound_box) - min(v[i] for v in obj.bound_box))
        
        # 获取几何复杂度指标
        me = obj.data
        vertex_count = len(me.vertices)
        face_count = len(me.polygons)
        
        # 基于几何复杂度调整体积估算
        # 更多的面通常意味着更复杂的几何体，可能有更高的填充率
        complexity_factor = min(1.0, max(0.2, face_count / max(vertex_count, 1)))
        
        return bbox_volume * complexity_factor
    except:
        return 0.0

def calculate_and_filter_parts(mesh_objs, min_part_ratio, max_parts, out_dir):
    """计算部件体积并根据比例和数量约束进行过滤"""
    # 计算体积，如果复杂方法失败则使用简单方法
    print("开始计算组件体积...")
    volumes = {}
    for o in mesh_objs:
        try:
            complex_vol = mesh_object_volume(o)
            simple_vol = simple_volume_estimate(o)
            
            # 如果复杂方法的结果看起来不合理，使用简单方法
            if complex_vol <= 0 or (simple_vol > 0 and complex_vol < simple_vol * 0.01):
                print(f"使用简单方法计算 {o.name} 的体积 (复杂方法结果: {complex_vol:.6f}, 简单方法: {simple_vol:.6f})")
                volumes[o.name] = simple_vol
            else:
                volumes[o.name] = complex_vol
        except Exception as e:
            print(f"体积计算失败 {o.name}: {e}")
            volumes[o.name] = simple_volume_estimate(o)

    total_volume = sum(volumes.values()) if volumes else 0.0

    # 调试信息：显示每个组件的体积计算结果
    print("=== 体积计算调试信息 ===")
    for obj_name, vol in sorted(volumes.items(), key=lambda x: x[1], reverse=True):
        obj = next((o for o in mesh_objs if o.name == obj_name), None)
        if obj:
            # 获取边界框信息作为参考
            bbox_volume = 1.0
            for i in range(3):
                bbox_volume *= abs(max(v[i] for v in obj.bound_box) - min(v[i] for v in obj.bound_box))
            
            # 获取顶点和面数量
            me = obj.data
            vertex_count = len(me.vertices)
            face_count = len(me.polygons)
            
            # 计算体积比率
            ratio = vol / total_volume if total_volume > 0 else 0.0
            
            print(f"{obj_name}:")
            print(f"  计算体积={vol:.6f} ({ratio:.4f}={ratio*100:.2f}%)")
            print(f"  边界框体积={bbox_volume:.6f}")
            print(f"  几何信息: {vertex_count}顶点, {face_count}面")
            print(f"  过滤状态: {'保留' if ratio >= min_part_ratio else '过滤'}")
            print()
            
    print(f"总体积: {total_volume:.6f}")
    print(f"过滤阈值: {min_part_ratio} ({min_part_ratio*100:.1f}%)")
    print("========================")

    if total_volume <= 0.0:
        print("Total volume is zero; skipping filtering.")
        filtered_mesh_objs = mesh_objs
    else:
        # cull parts below ratio
        keep = []
        for o in mesh_objs:
            v = volumes.get(o.name, 0.0)
            ratio = v / total_volume if total_volume > 0 else 0.0
            if ratio >= min_part_ratio:
                keep.append(o)
            else:
                print(f"Cull part {o.name} ratio={ratio:.4f}")
        filtered_mesh_objs = keep
        print("After ratio cull, parts:", len(filtered_mesh_objs))

    # enforce asset-level constraints
    if len(filtered_mesh_objs) > max_parts or len(filtered_mesh_objs) <= 1:
        print(f"Skip asset: parts count {len(filtered_mesh_objs)} violates constraints (max={max_parts}, >1).")
        # write minimal meta.json indicating skip
        try:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "meta.json"), 'w') as f:
                json.dump({"skipped": True, "reason": "parts_count_constraints", "parts": len(filtered_mesh_objs)}, f, indent=2)
        except Exception as e:
            print("Failed to write skip meta:", e)
        sys.exit(0)
    
    return filtered_mesh_objs

def setup_camera_and_rendering(args):
    """设置相机和渲染参数"""
    # Ensure a camera exists; if the glb had a camera, it should be imported as camera object
    cameras = [o for o in bpy.context.scene.objects if o.type == 'CAMERA']
    if cameras:
        cam = cameras[0]
    else:
        # create a simple camera
        cam_data = bpy.data.cameras.new("cam")
        cam = bpy.data.objects.new("cam", cam_data)
        bpy.context.scene.collection.objects.link(cam)
        cam.location = (0, -3, 1)
        cam.rotation_euler = (1.2, 0, 0)
    
    # Set initial clip values, will be adjusted based on scene size
    cam.data.clip_start = 0.01
    cam.data.clip_end = 100.0

    # Use orthographic camera to guarantee full framing independent of distance
    cam.data.type = 'ORTHO'

    # Ensure render engine and base lighting
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    print("DEBUG: Using render engine:", scene.render.engine)
    
    # Disable ambient occlusion and shadows for flat, uniform lighting
    scene.eevee.use_gtao = False  # No ambient occlusion
    scene.eevee.use_bloom = False  # No bloom
    scene.eevee.use_ssr = False  # No screen space reflections

    # Configure GPU rendering
    # Enable GPU compute in Blender preferences
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.refresh_devices()
        prefs.compute_device_type = 'CUDA'  # or 'OPENCL' depending on your GPU
        # Enable all available GPU devices
        for device in prefs.devices:
            if device.type in ['CUDA', 'OPENCL']:
                device.use = True
                print(f"DEBUG: Enabled GPU device: {device.name}")
    except Exception as e:
        print(f"DEBUG: GPU setup warning: {e}")

    # EEVEE automatically uses GPU when available
    # Set up Cycles GPU
    if hasattr(scene, 'cycles'):
        scene.cycles.device = 'GPU'

    # World settings - Use white environment lighting, but make camera see black background
    # This provides the same lighting as white background mode, but with black background in renders
    if not scene.world:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    
    # Clear existing world nodes and rebuild
    world_tree = scene.world.node_tree
    world_tree.nodes.clear()
    
    # Create Light Path node to detect camera rays
    node_light_path = world_tree.nodes.new(type='ShaderNodeLightPath')
    
    # Create two Background nodes
    # One for camera (black), one for lighting (white)
    node_bg_camera = world_tree.nodes.new(type='ShaderNodeBackground')
    node_bg_camera.inputs['Color'].default_value = (0.0, 0.0, 0.0, 1.0)  # Black for camera
    node_bg_camera.inputs['Strength'].default_value = 0.0
    
    node_bg_light = world_tree.nodes.new(type='ShaderNodeBackground')
    node_bg_light.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)  # White for lighting
    node_bg_light.inputs['Strength'].default_value = 1.0  # Same as white background mode
    
    # Create Mix Shader to choose between them
    node_mix = world_tree.nodes.new(type='ShaderNodeMixShader')
    
    # Create World Output
    node_output = world_tree.nodes.new(type='ShaderNodeOutputWorld')
    
    # Connect nodes:
    # Use Light Path "Is Camera Ray" to determine which background to use
    world_tree.links.new(node_light_path.outputs['Is Camera Ray'], node_mix.inputs['Fac'])
    world_tree.links.new(node_bg_light.outputs['Background'], node_mix.inputs[1])  # Not camera ray -> white light
    world_tree.links.new(node_bg_camera.outputs['Background'], node_mix.inputs[2])  # Camera ray -> black background
    world_tree.links.new(node_mix.outputs['Shader'], node_output.inputs['Surface'])
    
    # Remove any existing global lights (not needed with environment lighting)
    for obj in list(scene.objects):
        if obj.type == 'LIGHT' and obj.name.startswith('GlobalLight_'):
            bpy.data.objects.remove(obj, do_unlink=True)

    # Optional camera-aligned headlight
    def get_or_create_headlight():
        obj = bpy.data.objects.get("Camera_Headlight")
        if obj and obj.type == 'LIGHT':
            obj.data.type = 'SUN'
            obj.data.energy = args.headlight_strength
            return obj
        light_data = bpy.data.lights.new(name="Camera_Headlight", type='SUN')
        light_data.energy = args.headlight_strength
        obj = bpy.data.objects.new(name="Camera_Headlight", object_data=light_data)
        bpy.context.scene.collection.objects.link(obj)
        return obj
    
    camera_headlight = get_or_create_headlight() if args.use_camera_headlight else None

    # Setup render
    scene.camera = cam
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'  # 分割任务使用RGB模式（白色背景）
    scene.render.resolution_x = args.res
    scene.render.resolution_y = args.res
    scene.frame_current = 1
    
    # Reduce Blender's verbose output
    scene.render.use_compositing = True
    scene.render.use_sequencer = False
    
    return cam, scene, camera_headlight



# ----------------------
# Helpers: bbox + camera framing
# ----------------------

def get_object_bounds_world(obj):
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    min_v = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    max_v = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    center = (min_v + max_v) * 0.5
    extents = max_v - min_v
    radius = extents.length * 0.5
    return min_v, max_v, center, extents, radius

def get_object_corners_world(obj):
    return [obj.matrix_world @ Vector(c) for c in obj.bound_box]

def get_group_corners_world(objs):
    corners = []
    for o in objs:
        if o.type == 'MESH':
            corners.extend(get_object_corners_world(o))
    return corners

def point_camera_to(cam_obj, target):
    direction = (target - cam_obj.location)
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()

def frame_camera_on_object(cam_obj, obj, margin_scale=1.2, distance_scale=6.0):
    _, _, center, extents, radius = get_object_bounds_world(obj)
    # avoid degenerate sizes
    safe_extents = Vector((max(extents.x, 1e-3), max(extents.y, 1e-3), max(extents.z, 1e-3)))
    max_plane_extent = max(safe_extents.x, safe_extents.z)
    # Orthographic scale defines the width of the view in world units
    cam_obj.data.ortho_scale = float(max_plane_extent * margin_scale)
    # Place camera along -Y looking at center; distance does not affect scale in ORTHO, but must clear clipping
    cam_distance = float(max(safe_extents.length * distance_scale, 0.1))
    cam_obj.location = (center.x, center.y - cam_distance, center.z)
    point_camera_to(cam_obj, center)

def compute_view_setup(obj, margin_scale=1.2, distance_scale=6.0):
    _, _, center, extents, radius = get_object_bounds_world(obj)
    safe_extents = Vector((max(extents.x, 1e-3), max(extents.y, 1e-3), max(extents.z, 1e-3)))
    max_plane_extent = max(safe_extents.x, safe_extents.z)
    ortho_scale = float(max_plane_extent * margin_scale)
    cam_distance = float(max(safe_extents.length * distance_scale, 0.1))
    return center, safe_extents, ortho_scale, cam_distance

def set_camera_by_direction(cam_obj, center, ortho_scale, cam_distance, dir_vec, obj=None, margin_scale=1.2):
    # dir_vec should be a Vector of length > 0; camera placed at center - dir*distance
    dv = Vector((dir_vec[0], dir_vec[1], dir_vec[2]))
    if dv.length == 0:
        dv = Vector((0.0, -1.0, 0.0))
    dv.normalize()

    # Build camera basis (right, up, forward)
    world_up = Vector((0.0, 0.0, 1.0))
    if abs(dv.dot(world_up)) > 0.999:
        world_up = Vector((0.0, 1.0, 0.0))
    right = dv.cross(world_up).normalized()  # perpendicular to forward and up
    up = right.cross(dv).normalized()

    # If object provided, compute orthographic scale to fully include projected bbox in this view
    if obj is not None:
        corners = get_object_corners_world(obj)
        # project onto right/up axes (centered at center)
        xs = []
        ys = []
        for p in corners:
            rel = p - center
            xs.append(rel.dot(right))
            ys.append(rel.dot(up))
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        ortho_scale = float(max(width, height) * margin_scale)

    cam_obj.data.ortho_scale = ortho_scale
    cam_obj.location = (center.x - dv.x * cam_distance,
                        center.y - dv.y * cam_distance,
                        center.z - dv.z * cam_distance)
    point_camera_to(cam_obj, center)

def set_camera_by_direction_group(cam_obj, center, ortho_scale, cam_distance, dir_vec, corners, margin_scale=1.2):
    dv = Vector((dir_vec[0], dir_vec[1], dir_vec[2]))
    if dv.length == 0:
        dv = Vector((0.0, -1.0, 0.0))
    dv.normalize()
    world_up = Vector((0.0, 0.0, 1.0))
    if abs(dv.dot(world_up)) > 0.999:
        world_up = Vector((0.0, 1.0, 0.0))
    right = dv.cross(world_up).normalized()
    up = right.cross(dv).normalized()
    # project all corners to compute width/height in this view
    xs, ys = [], []
    for p in corners:
        rel = p - center
        xs.append(rel.dot(right))
        ys.append(rel.dot(up))
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    cam_obj.data.ortho_scale = float(max(width, height) * margin_scale)
    cam_obj.location = (center.x - dv.x * cam_distance,
                        center.y - dv.y * cam_distance,
                        center.z - dv.z * cam_distance)
    point_camera_to(cam_obj, center)

def restore_original_materials(mesh_objs, original_materials):
    """恢复原始材质,如果没有原始材质则创建默认白色材质"""
    print(f"\n恢复原始材质...")
    
    for obj in mesh_objs:
        # 清除当前材质
        obj.data.materials.clear()
        
        # 尝试恢复原始材质
        if obj.name in original_materials and original_materials[obj.name]:
            for mat in original_materials[obj.name]:
                obj.data.materials.append(mat)
            print(f"  {obj.name}: 恢复 {len(original_materials[obj.name])} 个原始材质")
        else:
            # 如果没有原始材质,创建默认白色材质
            mat_name = f"Default_Material_{obj.name}"
            mat = bpy.data.materials.get(mat_name)
            
            if mat is None:
                mat = bpy.data.materials.new(name=mat_name)
                mat.use_nodes = True
                nodes = mat.node_tree.nodes
                nodes.clear()
                
                # 创建Principled BSDF节点
                bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
                bsdf.location = (0, 0)
                bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1.0)  # 浅灰色
                
                # 创建材质输出节点
                output = nodes.new(type='ShaderNodeOutputMaterial')
                output.location = (200, 0)
                
                # 连接节点
                mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
            
            obj.data.materials.append(mat)
            print(f"  {obj.name}: 创建默认材质(原始材质不存在)")

def assign_colors_to_parts(mesh_objs):
    """为每个part分配不同的颜色材质（使用Emission材质确保纯色渲染）"""
    # 语义分割常用颜色调色板 (RGB格式, 0-1范围)
    # 10种高对比度颜色，易于区分，常用于语义分割任务
    colors = [
        (1.0, 0.0, 0.0),           # 红色 Red
        (0.0, 1.0, 0.0),           # 绿色 Green
        (0.0, 0.0, 1.0),           # 蓝色 Blue
        (1.0, 1.0, 0.0),           # 黄色 Yellow
        (1.0, 0.0, 1.0),           # 洋红色 Magenta
        (0.0, 1.0, 1.0),           # 青色 Cyan
        (1.0, 0.5, 0.0),           # 橙色 Orange
        (0.5, 0.0, 1.0),           # 紫色 Purple
        (1.0, 0.0, 0.5),           # 粉红色 Pink
        (0.5, 1.0, 0.0),           # 浅绿色 Lime
    ]
    
    print(f"\n为 {len(mesh_objs)} 个parts分配颜色材质...")
    
    for i, obj in enumerate(mesh_objs):
        # 获取或创建材质
        mat_name = f"Part_Emission_Material_{i}"
        mat = bpy.data.materials.get(mat_name)
        
        if mat is None:
            mat = bpy.data.materials.new(name=mat_name)
        
        # 启用节点
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        
        # 创建Emission节点（自发光，不受光照影响）
        emission = nodes.new(type='ShaderNodeEmission')
        emission.location = (0, 0)
        
        # 设置颜色
        color = colors[i % len(colors)]
        emission.inputs['Color'].default_value = (*color, 1.0)  # RGBA
        emission.inputs['Strength'].default_value = 1.0  # 发光强度
        
        # 创建材质输出节点
        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (200, 0)
        
        # 连接节点
        mat.node_tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])
        
        # 清除对象的所有材质
        obj.data.materials.clear()
        
        # 分配材质给对象
        obj.data.materials.append(mat)
        
        print(f"  Part {i} ({obj.name}): 分配Emission颜色 RGB{color}")
    
    return colors[:len(mesh_objs)]

def fibonacci_sphere_directions(n):
    # Evenly distributed directions on the sphere using golden angle
    dirs = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        y = 1.0 - (2.0 * (i + 0.5) / n)
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * i
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        dirs.append((x, y, z))
    return dirs

def sync_headlight_to_camera(cam_obj, camera_headlight=None):
    if not camera_headlight:
        return
    camera_headlight.matrix_world = cam_obj.matrix_world.copy()

# ----------------------
# Segmentation mask generation functions
# ----------------------



def generate_incomplete_part_images(visibility_data, out_dir, base_out_dir):
    """
    Generate incomplete part images from mask_part and whole images.
    从mask_part图像和whole图像生成incomplete_part图像（可见部分）
    """
    print("Generating incomplete part images...")
    incomplete_part_data = {}
    
    for cam_idx, vis_data in visibility_data.items():
        print(f"\nProcessing camera {cam_idx} incomplete parts...")
        
        # Load the whole image for this camera
        whole_path = os.path.join(out_dir, f"cam_{cam_idx}_whole.png")
        if not os.path.exists(whole_path):
            print(f"Warning: Whole image not found: {whole_path}")
            continue
            
        try:
            whole_img = Image.open(whole_path)
            if whole_img.mode != 'RGBA':
                whole_img = whole_img.convert('RGBA')
            whole_array = np.array(whole_img)
            print(f"  Loaded whole image: {whole_array.shape}")
            
            incomplete_part_files = []
            
            # Process each part mask
            for part_idx in vis_data['part_indices']:
                mask_file = f"cam_{cam_idx}_mask_{part_idx}.png"
                mask_path = os.path.join(out_dir, mask_file)
                
                if not os.path.exists(mask_path):
                    print(f"Warning: Mask file not found: {mask_path}")
                    continue
                
                try:
                    # Load mask image
                    mask_img = Image.open(mask_path)
                    if mask_img.mode != 'L':
                        mask_img = mask_img.convert('L')
                    mask_array = np.array(mask_img)
                    
                    # Ensure mask and whole image have same dimensions
                    if mask_array.shape[:2] != whole_array.shape[:2]:
                        mask_img = mask_img.resize((whole_array.shape[1], whole_array.shape[0]), Image.NEAREST)
                        mask_array = np.array(mask_img)
                    
                    # Create output image (RGBA format)
                    height, width = whole_array.shape[:2]
                    output_array = np.zeros((height, width, 4), dtype=np.uint8)
                    
                    # Create visibility mask: non-zero pixels indicate visible areas
                    visible_mask = mask_array > 128  # Threshold grayscale values
                    
                    # Apply mask to whole image
                    # Visible area: use whole image colors
                    # Invisible area: black with full opacity
                    output_array[visible_mask] = whole_array[visible_mask]
                    output_array[~visible_mask] = [0, 0, 0, 255]  # Black with full opacity
                    
                    # Save result
                    output_filename = f"cam_{cam_idx}_incomplete_part_{part_idx}.png"
                    output_path = os.path.join(out_dir, output_filename)
                    output_img = Image.fromarray(output_array, 'RGBA')
                    output_img.save(output_path)
                    
                    # Store relative path to base output directory
                    incomplete_part_files.append(os.path.relpath(output_path, base_out_dir))
                    
                    # Statistics
                    visible_pixels = np.count_nonzero(visible_mask)
                    total_pixels = mask_array.size
                    coverage = visible_pixels / total_pixels * 100
                    
                    print(f"    Generated {output_filename}: {coverage:.1f}% visible ({visible_pixels}/{total_pixels} pixels)")
                    
                except Exception as e:
                    print(f"Error processing part {part_idx}: {e}")
            
            if incomplete_part_files:
                incomplete_part_data[cam_idx] = incomplete_part_files
            
        except Exception as e:
            print(f"Error loading whole image for camera {cam_idx}: {e}")
    
    return incomplete_part_data



def prepare_camera_setup(mesh_objs, cam, scene, args):
    """准备相机设置和参数"""
    # Prepare camera directions once
    dirs = fibonacci_sphere_directions(args.num_views)

    # Precompute overall bbox, per-camera fixed scale and distance to enforce consistent scale across parts
    all_corners_for_scale = get_group_corners_world(mesh_objs)
    if all_corners_for_scale:
        min_v_all = Vector((min(v.x for v in all_corners_for_scale), min(v.y for v in all_corners_for_scale), min(v.z for v in all_corners_for_scale)))
        max_v_all = Vector((max(v.x for v in all_corners_for_scale), max(v.y for v in all_corners_for_scale), max(v.z for v in all_corners_for_scale)))
        center_all = (min_v_all + max_v_all) * 0.5
        extents_all = max_v_all - min_v_all
        
        # Adjust camera clip distances based on scene size
        scene_radius = extents_all.length * 0.5
        cam.data.clip_start = max(0.001, scene_radius * 0.001)  # 0.1% of scene radius
        cam.data.clip_end = scene_radius * 15.0  # 15x scene radius to ensure all geometry is captured
        print(f"DEBUG: Scene radius: {scene_radius:.3f}, Clip range: {cam.data.clip_start:.6f} - {cam.data.clip_end:.3f}")
    else:
        center_all = Vector((0.0, 0.0, 0.0))
        extents_all = Vector((1.0, 1.0, 1.0))

    fixed_cam_params = []  # list of (ortho_scale, distance, up_vector, dir_vector)
    for cam_idx in range(len(dirs)):
        dvec = Vector((dirs[cam_idx][0], dirs[cam_idx][1], dirs[cam_idx][2]))
        if dvec.length == 0:
            dvec = Vector((0.0, -1.0, 0.0))
        dvec.normalize()
        world_up = Vector((0.0, 0.0, 1.0))
        if abs(dvec.dot(world_up)) > 0.999:
            world_up = Vector((0.0, 1.0, 0.0))
        right = dvec.cross(world_up).normalized()
        up_vec = right.cross(dvec).normalized()
        # project overall corners to get a shared ortho scale for this view
        xs, ys = [], []
        for p in all_corners_for_scale:
            rel = p - center_all
            xs.append(rel.dot(right))
            ys.append(rel.dot(up_vec))
        width = (max(xs) - min(xs)) if xs else 1.0
        height = (max(ys) - min(ys)) if ys else 1.0
        ortho_scale_fixed = float(max(width, height) * 1.25)
        distance_fixed = float(max(extents_all.length * 6.0, 0.1))
        fixed_cam_params.append((ortho_scale_fixed, distance_fixed, up_vec, dvec))

    # Initialize per-camera logs
    camera_logs = []
    for cam_idx in range(len(dirs)):
        _, _, up_vec, dvec = fixed_cam_params[cam_idx]
        camera_logs.append({
            "cam_id": cam_idx,
            "dir": [float(dvec.x), float(dvec.y), float(dvec.z)],
            "up": [float(up_vec.x), float(up_vec.y), float(up_vec.z)],
            "clip": [float(cam.data.clip_start), float(cam.data.clip_end)],
            "position": None,
            "whole": None,
            "complete_part": [],
            "incomplete_part_mask": [],
        })
    
    return dirs, center_all, fixed_cam_params, camera_logs

def render_individual_parts(mesh_objs, all_mesh_objs, dirs, center_all, fixed_cam_params, 
                           cam, scene, camera_logs, out_dir, base_out_dir, camera_headlight=None):
    """渲染各个部件"""
    i = 0
    for o in mesh_objs:
        # hide ALL mesh objects first (including filtered ones)
        for other in all_mesh_objs:
            other.hide_render = True
        # then show only the current part
        o.hide_render = False
        try:
            # Pre-compute view parameters based on this part
            center, safe_extents, ortho_scale, cam_distance = compute_view_setup(o)

            for cam_idx, d in enumerate(dirs):
                # use fixed per-camera scale and distance, and keep camera position fixed per view (no re-centering)
                ortho_scale_fixed, distance_fixed, _, _ = fixed_cam_params[cam_idx]
                set_camera_by_direction(cam, center_all, ortho_scale_fixed, distance_fixed, d, obj=None, margin_scale=1.25)
                sync_headlight_to_camera(cam, camera_headlight)

                # Render RGB image
                fn = os.path.join(out_dir, f"cam_{cam_idx}_part_{i}.png")
                scene.render.filepath = fn
                bpy.ops.render.render(write_still=True)
                print("Rendered", fn)
                
                # update per-camera position once (same view definition and fixed center)
                camera_logs[cam_idx]["position"] = [float(cam.location.x), float(cam.location.y), float(cam.location.z)]
                # record image path (relative to base output directory)
                rel_fn = os.path.relpath(fn, base_out_dir)
                camera_logs[cam_idx]["complete_part"].append(rel_fn)

        except Exception as e:
            print("Rendering failed for", o.name, e)

        i += 1
    
    return camera_logs

def render_segment_images(mesh_objs, all_mesh_objs, dirs, cam, scene, camera_logs,
                         out_dir, base_out_dir, part_colors, camera_headlight=None):
    """渲染彩色segment图像(所有parts同时可见,每个part不同颜色)"""
    print("\n开始渲染segment图像...")
    
    try:
        # 显示所有带颜色的parts
        for o in all_mesh_objs:
            o.hide_render = True
            if hasattr(o, 'hide_viewport'):
                o.hide_viewport = True
        
        for o in mesh_objs:
            o.hide_render = False
            if hasattr(o, 'hide_viewport'):
                o.hide_viewport = False
        
        # 计算整体边界框
        all_corners = get_group_corners_world(mesh_objs)
        if all_corners:
            min_v = Vector((min(v.x for v in all_corners), min(v.y for v in all_corners), min(v.z for v in all_corners)))
            max_v = Vector((max(v.x for v in all_corners), max(v.y for v in all_corners), max(v.z for v in all_corners)))
            center = (min_v + max_v) * 0.5
            extents = max_v - min_v
            cam_distance = float(max(extents.length * 6.0, 0.1))
            
            for d in dirs:
                cam_idx = dirs.index(d)
                set_camera_by_direction_group(cam, center, 1.0, cam_distance, d, all_corners, margin_scale=1.25)
                sync_headlight_to_camera(cam, camera_headlight)
                
                # 渲染segment图像
                fn = os.path.join(out_dir, f"cam_{cam_idx}_segment.png")
                scene.render.filepath = fn
                bpy.ops.render.render(write_still=True)
                print(f"  渲染segment: {fn}")
                
                # 保存segment路径到camera log (relative to base output directory)
                if cam_idx < len(camera_logs):
                    camera_logs[cam_idx]["whole_segment"] = os.path.relpath(fn, base_out_dir)
    
    except Exception as e:
        print(f"渲染segment失败: {e}")
    
    return camera_logs

def render_whole_model(mesh_objs, all_mesh_objs, dirs, cam, scene, camera_logs, 
                      out_dir, base_out_dir, camera_headlight=None):
    """渲染整体模型视图"""
    try:
        # hide ALL mesh objects first (including filtered ones)
        for o in all_mesh_objs:
            o.hide_render = True
            if hasattr(o, 'hide_viewport'):
                o.hide_viewport = True
        # then unhide only the filtered/kept parts to render the whole model
        for o in mesh_objs:
            o.hide_render = False
            if hasattr(o, 'hide_viewport'):
                o.hide_viewport = False
        # compute union bbox center and corners
        all_corners = get_group_corners_world(mesh_objs)
        if all_corners:
            min_v = Vector((min(v.x for v in all_corners), min(v.y for v in all_corners), min(v.z for v in all_corners)))
            max_v = Vector((max(v.x for v in all_corners), max(v.y for v in all_corners), max(v.z for v in all_corners)))
            center = (min_v + max_v) * 0.5
            extents = max_v - min_v
            # reuse the same distance policy as parts for consistency
            cam_distance = float(max(extents.length * 6.0, 0.1))
            for d in dirs:
                cam_idx = dirs.index(d)
                set_camera_by_direction_group(cam, center, 1.0, cam_distance, d, all_corners, margin_scale=1.25)
                sync_headlight_to_camera(cam, camera_headlight)
                
                # Render RGB image (+ depth if enabled)
                fn = os.path.join(out_dir, f"cam_{cam_idx}_whole.png")
                scene.render.filepath = fn
                bpy.ops.render.render(write_still=True)
                print("Rendered", fn)
                
                # save overall paths into corresponding camera entry before complete_part (relative to base output directory)
                camera_logs[cam_idx]["whole"] = os.path.relpath(fn, base_out_dir)

    except Exception as e:
        print("Rendering overall failed:", e)
    
    return camera_logs

def extract_masks_from_segment(camera_logs, mesh_objs, part_colors, out_dir, base_out_dir):
    """从渲染的segment图像中提取每个part的mask"""
    print("\n从segment图像提取masks...")
    
    visibility_data = {}
    
    # 将0-1范围的颜色转换为0-255范围
    colors_255 = []
    for color in part_colors:
        colors_255.append([int(c * 255) for c in color])
    
    for cam_idx, cam_log in enumerate(camera_logs):
        if "whole_segment" not in cam_log:
            print(f"  相机 {cam_idx}: 未找到segment图像")
            continue
        
        # cam_log["whole_segment"] is already relative to base_out_dir
        segment_path = os.path.join(base_out_dir, cam_log["whole_segment"])
        
        if not os.path.exists(segment_path):
            print(f"  相机 {cam_idx}: segment文件不存在: {segment_path}")
            continue
        
        try:
            # 加载segment图像
            segment_img = Image.open(segment_path)
            if segment_img.mode != 'RGB':
                segment_img = segment_img.convert('RGB')
            segment_array = np.array(segment_img)
            height, width = segment_array.shape[:2]
            
            print(f"\n  相机 {cam_idx}: 处理segment图像 {segment_array.shape}")
            
            # 调试：显示segment图像中的唯一颜色
            unique_colors = np.unique(segment_array.reshape(-1, 3), axis=0)
            print(f"  Segment图像中发现 {len(unique_colors)} 种唯一颜色:")
            for uc in unique_colors[:20]:  # 只显示前20种
                pixel_count = np.sum(np.all(segment_array == uc, axis=2))
                print(f"    RGB{tuple(uc)}: {pixel_count} 像素")
            
            part_masks = {}
            part_indices = []
            mask_files = []
            
            # 为每个part提取mask
            for part_idx in range(len(mesh_objs)):
                if part_idx >= len(colors_255):
                    break
                
                target_color = colors_255[part_idx]
                
                # 创建mask: 找到所有匹配颜色的像素
                # 使用容差来处理渲染可能的轻微颜色变化
                color_tolerance = 10  # RGB容差
                
                r_match = np.abs(segment_array[:, :, 0].astype(int) - target_color[0]) <= color_tolerance
                g_match = np.abs(segment_array[:, :, 1].astype(int) - target_color[1]) <= color_tolerance
                b_match = np.abs(segment_array[:, :, 2].astype(int) - target_color[2]) <= color_tolerance
                
                # 所有通道都匹配的像素
                mask = (r_match & g_match & b_match).astype(np.uint8)
                
                visible_pixels = np.count_nonzero(mask)
                
                # 保存mask图像（无论是否可见都保存，可见区域为白色，不可见区域为黑色）
                part_masks[part_idx] = mask
                part_indices.append(part_idx)
                
                # 保存mask图像 - 命名格式：cam_{cam_idx}_mask_{part_idx}.png
                mask_filename = f"cam_{cam_idx}_mask_{part_idx}.png"
                mask_path = os.path.join(out_dir, mask_filename)
                mask_img = Image.fromarray(mask * 255, mode='L')
                mask_img.save(mask_path)
                
                # Store relative path to base output directory
                mask_files.append(os.path.relpath(mask_path, base_out_dir))
                
                coverage = visible_pixels / (height * width) * 100
                if visible_pixels > 0:
                    print(f"    Part {part_idx} (颜色 RGB{target_color}): {visible_pixels} 像素 ({coverage:.2f}%)")
                else:
                    print(f"    Part {part_idx} (颜色 RGB{target_color}): 未可见 (全黑mask)")
            
            if part_masks:
                visibility_data[cam_idx] = {
                    'masks': part_masks,
                    'part_indices': part_indices,
                    'mask_files': mask_files,
                    'segment_file': cam_log["whole_segment"]
                }
                print(f"  相机 {cam_idx}: 成功提取 {len(part_masks)} 个part masks")
            else:
                print(f"  相机 {cam_idx}: 未提取到任何mask")
        
        except Exception as e:
            print(f"  相机 {cam_idx}: 处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    return visibility_data

def process_segmentation_masks(camera_logs, mesh_objs, part_colors, out_dir, base_out_dir):
    """处理分割掩码生成"""
    visibility_data = {}
    incomplete_part_data = {}
    try:
        print("\n开始从segment图像生成masks...")
        
        # 从segment图像提取masks
        visibility_data = extract_masks_from_segment(camera_logs, mesh_objs, part_colors, out_dir, base_out_dir)
        
        print(f"\n为 {len(visibility_data)} 个相机视图生成了visibility masks")
        
        # 从masks和whole图像生成incomplete part图像
        if visibility_data:
            incomplete_part_data = generate_incomplete_part_images(visibility_data, out_dir, base_out_dir)
            print(f"为 {len(incomplete_part_data)} 个相机视图生成了incomplete part图像")
        
    except Exception as e:
        print(f"Mask生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    return visibility_data, incomplete_part_data

def write_metadata(camera_logs, visibility_data, incomplete_part_data, out_dir):
    """写入元数据文件"""
    # Write meta.json in output directory (using relative paths)
    try:
        json_path = os.path.join(out_dir, "meta.json")
        meta_data = {"cameras": camera_logs}
        
        # Add whole_segment, incomplete_part, and incomplete_part_mask data to each camera entry if visibility data is available
        if visibility_data:
            for cam_idx, vis_data in visibility_data.items():
                if cam_idx < len(camera_logs):
                    # Create a new ordered dictionary to ensure correct field order
                    new_cam_data = {}
                    
                    # Copy existing fields in desired order
                    for key in ["cam_id", "dir", "up", "clip", "position", "whole"]:
                        if key in camera_logs[cam_idx]:
                            new_cam_data[key] = camera_logs[cam_idx][key]
                    
                    # Add whole_segment after whole
                    new_cam_data["whole_segment"] = vis_data["segment_file"]
                    
                    # Add complete_part
                    new_cam_data["complete_part"] = camera_logs[cam_idx]["complete_part"]
                    
                    # Add incomplete_part (between complete_part and incomplete_part_mask)
                    if cam_idx in incomplete_part_data:
                        new_cam_data["incomplete_part"] = incomplete_part_data[cam_idx]
                    
                    # Add incomplete_part_mask
                    new_cam_data["incomplete_part_mask"] = vis_data["mask_files"]
                    
                    # Replace the camera data with the new ordered version
                    camera_logs[cam_idx] = new_cam_data
            
            # Also add detailed visibility masks info as separate section
            meta_data["visibility_masks"] = {}
            for cam_idx, vis_data in visibility_data.items():
                meta_data["visibility_masks"][f"cam_{cam_idx}"] = {
                    "part_indices": vis_data["part_indices"],
                    "mask_files": vis_data["mask_files"],
                    "segment_file": vis_data["segment_file"]
                }
        
        with open(json_path, 'w') as f:
            json.dump(meta_data, f, indent=2)
        print("Wrote", json_path)
    except Exception as e:
        print("Failed to write model JSON:", e)

def process_single_model(model_path, output_dir, args):
    """处理单个模型文件"""
    print(f"\n{'='*60}")
    print(f"Processing: {model_path}")
    print(f"Output to: {output_dir}")
    print(f"{'='*60}\n")
    
    # 临时修改args中的路径
    original_in_glb = args.in_glb
    original_out_dir = args.out_dir
    args.in_glb = model_path
    args.out_dir = output_dir
    
    try:
        # 2. 初始化环境
        IN, OUT_DIR = initialize_environment(args)
        
        # 3. 导入GLB文件并分离部件
        all_mesh_objs, mesh_objs, original_materials = import_glb_and_separate_parts(IN, args.use_loose_parts)
        
        # 4. 计算体积并过滤部件
        mesh_objs = calculate_and_filter_parts(mesh_objs, args.min_part_ratio, args.max_parts, OUT_DIR)
        
        # 5. 恢复原始材质(用于渲染individual parts和whole)
        restore_original_materials(mesh_objs, original_materials)
        
        # 6. 设置相机和渲染
        cam, scene, camera_headlight = setup_camera_and_rendering(args)
        
        # 7. 准备相机设置
        dirs, center_all, fixed_cam_params, camera_logs = prepare_camera_setup(
            mesh_objs, cam, scene, args)
        
        # 8. 渲染各个部件(正常外观)
        # Use original_out_dir as base for relative paths
        camera_logs = render_individual_parts(
            mesh_objs, all_mesh_objs, dirs, center_all, fixed_cam_params, 
            cam, scene, camera_logs, OUT_DIR, original_out_dir, camera_headlight)
        
        # 9. 为parts分配颜色材质(用于segment渲染)
        part_colors = assign_colors_to_parts(mesh_objs)
        
        # 10. 渲染彩色segment(所有parts同时可见,每个part不同颜色)
        camera_logs = render_segment_images(
            mesh_objs, all_mesh_objs, dirs, cam, scene, camera_logs,
            OUT_DIR, original_out_dir, part_colors, camera_headlight)
        
        # 11. 恢复原始材质(用于渲染whole)
        restore_original_materials(mesh_objs, original_materials)
        
        # 12. 渲染整体模型(正常外观)
        camera_logs = render_whole_model(
            mesh_objs, all_mesh_objs, dirs, cam, scene, camera_logs, 
            OUT_DIR, original_out_dir, camera_headlight)
        
        # 13. 从segment图像提取masks并生成incomplete parts
        visibility_data, incomplete_part_data = process_segmentation_masks(
            camera_logs, mesh_objs, part_colors, OUT_DIR, original_out_dir)
        
        # 14. 写入元数据
        write_metadata(camera_logs, visibility_data, incomplete_part_data, OUT_DIR)
        
        print(f"\n✓ Successfully processed: {model_path}\n")
        
    except Exception as e:
        print(f"\n✗ Failed to process {model_path}: {e}\n")
        import traceback
        traceback.print_exc()
    
    finally:
        # 恢复原始路径
        args.in_glb = original_in_glb
        args.out_dir = original_out_dir

def main():
    """主函数 - 整合所有功能模块"""
    # 1. 解析参数
    args = parse_arguments()
    
    # 检查是否为批量处理模式
    if args.batch and os.path.isdir(args.in_glb):
        print(f"\n批量处理模式: 扫描目录 {args.in_glb}")
        
        # 查找所有子目录中的.obj文件（排除smplx文件夹）
        obj_files = []
        skipped_count = 0
        for root, dirs, files in os.walk(args.in_glb):
            # 跳过smplx文件夹
            if 'smplx' in root.lower():
                skipped_count += len([f for f in files if f.endswith('.obj')])
                continue
            
            for file in files:
                if file.endswith('.obj'):
                    obj_path = os.path.join(root, file)
                    obj_files.append(obj_path)
        
        # 按路径排序，保持一致性
        obj_files.sort()
        
        print(f"找到 {len(obj_files)} 个 .obj 文件")
        if skipped_count > 0:
            print(f"跳过 {skipped_count} 个 smplx 文件")
        print()
        
        # 处理每个文件
        for idx, obj_path in enumerate(obj_files, 1):
            # 获取相对路径并保持目录结构
            rel_path = os.path.relpath(obj_path, args.in_glb)
            # 只保留目录结构，不包含文件名
            subdir_path = os.path.dirname(rel_path)
            output_dir = os.path.join(args.out_dir, subdir_path)
            
            print(f"\n[{idx}/{len(obj_files)}] 处理文件: {rel_path}")
            process_single_model(obj_path, output_dir, args)
        
        print(f"\n{'='*60}")
        print(f"批量处理完成! 共处理 {len(obj_files)} 个文件")
        print(f"{'='*60}\n")
    else:
        # 单文件处理模式
        process_single_model(args.in_glb, args.out_dir, args)

if __name__ == "__main__":
    main()
