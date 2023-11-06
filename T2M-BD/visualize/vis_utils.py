from visualize.rotation2xyz import Rotation2xyz
import numpy as np
from trimesh import Trimesh
import os
import torch
from visualize.simplify_loc2rot import joints2smpl
from tqdm import tqdm
import datetime
from utils.motion_process import recover_from_ric

class npy2obj:
    def __init__(self, motions, folder_name, std=None, mean=None, device=0, cuda=True):
        if std is not None and mean is not None:
            motions = motions * std + mean
            motions = recover_from_ric(torch.from_numpy(motions).float(), 22).numpy()
        self.motions = motions
        self.num_frames = self.motions.shape[0]
        self.rot2xyz = Rotation2xyz(device='cpu')
        self.faces = self.rot2xyz.smpl_model.faces
        self.j2s = joints2smpl(num_frames=self.num_frames, device_id=device, cuda=cuda)
        print(f'Running SMPLify, it may take a few minutes.')
        motion_tensor, opt_dict = self.j2s.joint2smpl(self.motions)  # [nframes, njoints, 3]
        self.motions = motion_tensor.cpu().numpy()

        self.vertices = self.rot2xyz(torch.tensor(self.motions), mask=None,
                                     pose_rep='rot6d', translation=True, glob=True,
                                     jointstype='vertices',
                                     # jointstype='smpl',  # for joint locations
                                     vertstrans=True)
        self.root_loc = self.motions[:, -1, :3, :].reshape(1, 1, 3, -1)
        self.vertices += self.root_loc

        date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        results_dir = f'./output/obj/{date}_{folder_name}'
        os.makedirs(results_dir, exist_ok = False)
        print('Saving obj files to [{}]'.format(os.path.abspath(results_dir)))
        for frame_i in tqdm(range(self.num_frames)):
            self.save_obj(os.path.join(results_dir, 'frame{:03d}.obj'.format(frame_i)), frame_i)
        self.save_npy(f'{results_dir}/motions.npy')

    def get_vertices(self, sample_i, frame_i):
        return self.vertices[sample_i, :, :, frame_i].squeeze().tolist()

    def get_trimesh(self, sample_i, frame_i):
        return Trimesh(vertices=self.get_vertices(sample_i, frame_i),
                       faces=self.faces)

    def save_obj(self, save_path, frame_i):
        mesh = self.get_trimesh(0, frame_i)
        with open(save_path, 'w') as fw:
            mesh.export(fw, 'obj')
        return save_path
    
    def save_npy(self, save_path):
        np.save(save_path, self.motions)
    #     data_dict = {
    #         'motion': self.motions[0, :, :, :self.num_frames],
    #         'thetas': self.motions[0, :-1, :, :self.num_frames],
    #         'root_translation': self.motions[0, -1, :3, :self.num_frames],
    #         'faces': self.faces,
    #         'vertices': self.vertices[0, :, :, :self.num_frames],
    #         'text': self.motions['text'][0],
    #         'length': self.num_frames,
    #     }
    #     np.save(save_path, data_dict)
