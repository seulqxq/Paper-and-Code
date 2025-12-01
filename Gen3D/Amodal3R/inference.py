import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import numpy as np
import imageio
from PIL import Image
from amodal3r.pipelines import Amodal3RImageTo3DPipeline
from amodal3r.utils import render_utils, postprocessing_utils
import cv2
import trimesh


def extract_glb(gs, mesh, mesh_simplify=0.95, texture_size=1024, export_path="output.glb"):
    """
    Extract a GLB file from the 3D model.

    Args:
        state (dict): The state of the generated 3D model.
        mesh_simplify (float): The mesh simplification factor.
        texture_size (int): The texture resolution.

    Returns:
        str: The path to the extracted GLB file.
    """

    glb = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False)
    glb.export(export_path)
    return export_path

def save_mesh(mesh_result, filename):
    vertices = mesh_result.vertices.cpu().numpy() if hasattr(mesh_result.vertices, 'cpu') else mesh_result.vertices
    faces = mesh_result.faces.cpu().numpy() if hasattr(mesh_result.faces, 'cpu') else mesh_result.faces
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    if mesh_result.vertex_attrs is not None:
        attrs = mesh_result.vertex_attrs.cpu().numpy() if hasattr(mesh_result.vertex_attrs, 'cpu') else mesh_result.vertex_attrs
        mesh.visual.vertex_colors = attrs
    
    mesh.export(filename)

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = Amodal3RImageTo3DPipeline.from_pretrained("Sm0kyWu/Amodal3R")
pipeline.cuda()


output_dir = "./output/1/"
os.makedirs(output_dir, exist_ok=True)

# can be single image or multiple images
images = [
    Image.open(f"./input/1/000000.png"),
]

masks = [
    Image.open(f"./input/1/000000_mask.png").convert("L"),
]


# Run the pipeline
outputs = pipeline.run_multi_image(
    images,
    masks,
    seed=1,
    # Optional parameters
    sparse_structure_sampler_params={
        "steps": 12,
        "cfg_strength": 7.5,
    },
    slat_sampler_params={
        "steps": 12,
        "cfg_strength": 3,
    },
)

# save as gif
video_gs = render_utils.render_video(outputs['gaussian'][0], bg_color=(1, 1, 1))['color']
video_mesh = render_utils.render_video(outputs['mesh'][0], bg_color=(1, 1, 1))['normal']
video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
imageio.mimsave(os.path.join(output_dir, "sample_multi.gif"), video, fps=30)

# save multi-view gs and mesh
gaussian = outputs['gaussian'][0]
multi_view_gs,_,_ = render_utils.render_multiview(gaussian, nviews=8, bg_color=(1, 1, 1))
multi_view_gs = multi_view_gs['color']
for i in range(8):
    output = multi_view_gs[i]
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, f"{i:03d}_gs.png"), output)

mesh = outputs['mesh'][0]
multi_view_mesh,_,_ = render_utils.render_multiview(mesh, nviews=8, bg_color=(1, 1, 1))
multi_view_mesh = multi_view_mesh['normal']
for i in range(8):
    output = multi_view_mesh[i]
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, f"{i:03d}_mesh.png"), output)

# # save mesh
save_mesh(mesh, os.path.join(output_dir, "mesh.ply"))

# export glb if needed
# glb_path = os.path.join(output_dir, "mesh.glb")
# extract_glb(outputs['gaussian'][0], outputs['mesh'][0], 0.5, 1024, glb_path)
