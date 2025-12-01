import numpy as np
import os
import random
from PIL import Image
import torch
import json
from dataLoader.utils import generate_random_mask, fov_to_ixt

class Amodal3RData(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super(Amodal3RData, self).__init__()
        self.data_root = cfg.data_root
        self.split = cfg.split
        self.img_size = np.array(cfg.img_size)
        self.img_downscale = self.img_size/512
        self.category = "combined"

        if self.category == "combined":
            scenes_name = np.array([f for f in sorted(os.listdir(os.path.join(self.data_root, 'renders'))) if os.path.isdir(f'{self.data_root}/renders/{f}')])
            self.split_num = len(scenes_name)
        elif self.category == "abo":
            with open("./dataset/ABO/abo.txt", "r") as f:
                base_scenes = f.readlines()
            base_scenes = [f.strip() for f in base_scenes]
            scenes_name = np.array([f for f in sorted(os.listdir(os.path.join(self.data_root, 'renders'))) if os.path.isdir(f'{self.data_root}/renders/{f}') and f in base_scenes])
            self.split_num = 4000
        elif self.category == "3df":
            with open("./dataset/3dfuture/3dfuture.txt", "r") as f:
                base_scenes = f.readlines()
            base_scenes = [f.strip() for f in base_scenes]
            scenes_name = np.array([f for f in sorted(os.listdir(os.path.join(self.data_root, 'renders'))) if os.path.isdir(f'{self.data_root}/renders/{f}') and f in base_scenes])
            self.split_num = 8000
        elif self.category == "hssd":
            with open("./dataset/hssd/hssd.txt", "r") as f:
                base_scenes = f.readlines()
            base_scenes = [f.strip() for f in base_scenes]
            scenes_name = np.array([f for f in sorted(os.listdir(os.path.join(self.data_root, 'renders'))) if os.path.isdir(f'{self.data_root}/renders/{f}') and f in base_scenes])
            self.split_num = 6000
            
        self.split_num = 50

        if self.split=='train':
            self.scenes_name = scenes_name[:self.split_num]
        else:
            if self.category == "combined":
                self.scenes_name = scenes_name[-100:]
            else:
                self.scenes_name = scenes_name[self.split_num:]

        print(self.category, " ", self.split, ": ", len(self.scenes_name))


    def build_metas(self, scene):
        json_info = json.load(open(os.path.join(self.data_root, 'renders', scene, f'transforms.json')))
        b2c = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        scene_info = {'ixts': [], 'c2ws': [], 'w2cs':[], 'img_paths': [], 'fovx': [], 'fovy':[], "mask_paths": []}
        positions = []
        for idx, frame in enumerate(json_info['frames']):
            c2w = np.array(frame['transform_matrix'])
            c2w = c2w @ b2c
            fov = frame["camera_angle_x"]
            ixt = fov_to_ixt(np.array([fov, fov]), np.array([512, 512]))
            scene_info['ixts'].append(ixt.astype(np.float32))
            scene_info['c2ws'].append(c2w.astype(np.float32))
            scene_info['w2cs'].append(np.linalg.inv(c2w.astype(np.float32)))
            img_path = os.path.join(self.data_root, 'renders', scene, f'{idx:03d}.png')
            scene_info['img_paths'].append(img_path)
            scene_info['fovx'].append(fov)
            scene_info['fovy'].append(fov)
            positions.append(c2w[:3,3])
            mask_path = os.path.join(self.data_root, 'renders_mask', scene, f'{idx:03d}.png')
            scene_info['mask_paths'].append(mask_path)
            
        scene_info['ss_latent_path'] = os.path.join(self.data_root, 'ss_latents/ss_enc_conv3d_16l8_fp16', scene + '.npz')
        scene_info['slat_path'] = os.path.join(self.data_root, 'latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16', scene + '.npz')
    
        return scene_info
        

    def __getitem__(self, index):
        scene_name = self.scenes_name[index]
        scene_info = self.build_metas(scene_name)
        
        ss_latent = np.load(scene_info['ss_latent_path'])['mean']
        ss_latent = torch.from_numpy(ss_latent).float()
        
        bg_color = np.zeros(3).astype(np.float32)
            
        num_frames = len(scene_info['img_paths'])
        # choose a random frame
        frame_idx = random.randint(0, num_frames-1)
        
        cond_img, cond_img_518_masked, visibility_mask, occlusion_mask = self.read_image(scene_info, frame_idx, bg_color)

        ret = {'cond_img': cond_img, 'cond_img_518_masked': cond_img_518_masked, 'ss_latent': ss_latent, 'slat_path': scene_info['slat_path'], 'visibility_mask': visibility_mask, 'occlusion_mask': occlusion_mask}

        return ret


    def read_image(self, scene, view_idx, bg_color):
        img_path = scene['img_paths'][view_idx]
        img = Image.open(img_path).convert("RGBA")
        img_518 = img.resize((518, 518), Image.Resampling.LANCZOS)

        img_518 = np.array(img_518).astype(np.float32) / 255.
        alpha = img_518[..., -1:]
        img_518 = (img_518[..., :3] * img_518[..., -1:] + bg_color*(1 - img_518[..., -1:])).astype(np.float32)
        
        occlusion_mask = generate_random_mask(height=518, width=518)
        bg_mask = (alpha<0.1)
        visibility_mask = occlusion_mask | bg_mask
        visibility_mask = 1 - visibility_mask
        
        # apply the mask
        img_518_masked = img_518 * visibility_mask

        return img_518.astype(np.float32), img_518_masked.astype(np.float32), visibility_mask.squeeze().astype(np.float32), occlusion_mask.squeeze().astype(np.float32)

    def __len__(self):
        return len(self.scenes_name)

