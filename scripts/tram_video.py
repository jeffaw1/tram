import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import argparse
import numpy as np
import os
import torch
import cv2
from tqdm import tqdm
from glob import glob
import imageio

from lib.vis.traj import *
from lib.models.smpl import SMPL
from lib.vis.renderer_img import Renderer
from lib.utils.rotation_conversions import quaternion_to_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, default='./example_video.mov')
parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

##### Read results from HPS #####
file = args.video
root = os.path.dirname(file)
seq = os.path.basename(file).split('.')[0]

seq_folder = f'{root}/{seq}'
img_folder = f'{seq_folder}/images'
hps_folder = f'{seq_folder}/hps'
imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
hps_files = sorted(glob(f'{hps_folder}/*.npy'))

# Read the first image to calculate the center and focal length
first_img = cv2.imread(imgfiles[0])
height, width = first_img.shape[:2]

# Calculate the image center
img_center = [width / 2, height / 2]
# Calculate the focal length as the diagonal of the image
img_focal = np.sqrt(width**2 + height**2)

smpl = SMPL()
colors = np.loadtxt('data/colors.txt')/255
colors = torch.from_numpy(colors).float()

max_track = len(hps_files)
tstamp =  [t for t in range(len(imgfiles))]
track_verts = {i:[] for i in tstamp}
track_joints = {i:[] for i in tstamp}
track_tid = {i:[] for i in tstamp}

for i in range(max_track):
    hps_file = hps_files[i]

    pred_smpl = np.load(hps_file, allow_pickle=True).item()
    pred_rotmat = pred_smpl['pred_rotmat']
    pred_shape = pred_smpl['pred_shape']
    pred_trans = pred_smpl['pred_trans']
    frame = pred_smpl['frame']

    mean_shape = pred_shape.mean(dim=0, keepdim=True)
    pred_shape = mean_shape.repeat(len(pred_shape), 1)

    pred = smpl(body_pose=pred_rotmat[:,1:], 
                global_orient=pred_rotmat[:,[0]], 
                betas=pred_shape, 
                transl=pred_trans.squeeze(),
                pose2rot=False, 
                default_smpl=True)
    pred_vert = pred.vertices
    pred_j3d = pred.joints[:, :24]

    for j, f in enumerate(frame.tolist()):
        track_tid[f].append(i)
        track_verts[f].append(pred_vert[j])



##### Render video for visualization #####
writer = imageio.get_writer(f'{seq_folder}/tram_output.mp4', fps=30, mode='I', 
                            format='FFMPEG', macro_block_size=1)

img = cv2.imread(imgfiles[0])
height, width = img.shape[:2]

renderer = Renderer(smpl.faces)
renderer.init_renderer(height=height, width=width)

for i in tqdm(range(len(imgfiles))):
    img = cv2.imread(imgfiles[i])
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    verts_list = track_verts[i]
    if len(verts_list) > 0:
        verts = torch.stack(track_verts[i]).cpu().numpy()
        
        # Assuming camera_translation is [0, 0, 2.5] (adjust as needed)
        camera_translation = np.array([0, 0, 0])
        
        rendered_image = renderer(verts, camera_translation, image=img_rgb)
    else:
        rendered_image = img_rgb.copy()
    
    out = np.concatenate([rendered_image], axis=1)
    #out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)  # Convert back to BGR for writing
    writer.append_data(out)

writer.close()
