from __future__ import print_function, division
import argparse
import torch
import os,sys
from os import walk, listdir
from os.path import isfile, join
import numpy as np
import joblib
import smplx
import trimesh
import h5py

sys.path.append('/home/epinyoan/git/joints2smpl')
sys.path.append('/home/epinyoan/git/joints2smpl/src')

from smplify import SMPLify3D
import config


class Joint2SMPL:
    def __init__(self, num_smplify_iters=100):
        self.batchSize=1
        self.num_joints=22
        self.num_smplify_iters=num_smplify_iters
        self.files="test_motion.npy"

        self.device = torch.device("cuda")
        # ---load predefined something
        print(config.SMPL_MODEL_DIR)
        smplmodel = smplx.create(config.SMPL_MODEL_DIR, 
                                model_type="smpl", gender="male", ext="pkl",
                                batch_size=self.batchSize).to(self.device)

        # ## --- load the mean pose as original ---- 
        smpl_mean_file = config.SMPL_MEAN_FILE

        file = h5py.File(smpl_mean_file, 'r')
        self.init_mean_pose = torch.from_numpy(file['pose'][:]).unsqueeze(0).float()
        self.init_mean_shape = torch.from_numpy(file['shape'][:]).unsqueeze(0).float()
        self.cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).to(self.device)
        #
        self.pred_pose = torch.zeros(self.batchSize, 72).to(self.device)
        self.pred_betas = torch.zeros(self.batchSize, 10).to(self.device)
        self.pred_cam_t = torch.zeros(self.batchSize, 3).to(self.device)
        self.keypoints_3d = torch.zeros(self.batchSize, self.num_joints, 3).to(self.device)

        # # #-------------initialize SMPLify
        self.smplify = SMPLify3D(smplxmodel=smplmodel,
                            batch_size=self.batchSize,
                            joints_category="AMASS",
                            num_iters=self.num_smplify_iters,
                            device=self.device)
        #print("initialize SMPLify3D done!")
            
        # --- load data ---
        # data = np.load("/home/epinyoan/git/joints2smpl/demo/demo_data/test_motion.npy")
        # run the whole seqs
        # num_seqs = data.shape[0]
        self.shape_params = torch.ones((1, 10))

    def run(self, pose_params, joint):
        joints = []
        vertices = []
        for idx in range(joint.shape[0]):
        #print(idx)

            # joints3d = data[idx] #*1.2 #scale problem [check first]	
            # self.keypoints_3d[0, :, :] = torch.Tensor(joints3d).to(self.device).float()

            self.pred_betas[0, :] = self.init_mean_shape
            self.pred_pose[0, :] = self.init_mean_pose
            self.pred_cam_t[0, :] = self.cam_trans_zero
            if idx != 0:
                self.pred_pose = pose_params.cuda()
            
            self.pred_pose = self.pred_pose.reshape(1,-1,3)
            self.pred_pose[:,:,1] -= 1
            self.pred_pose = self.pred_pose.reshape(1,-1)
            
            confidence_input =  torch.ones(self.num_joints)
            # make sure the foot and ankle
            print('------')  
            # ----- from initial to fitting -------
            new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, \
            new_opt_cam_t, new_opt_joint_loss = self.smplify(
                                                        self.pred_pose.detach(), # pose_params.cuda(),#
                                                        self.pred_betas.detach(), # self.shape_params.cuda(),#
                                                        self.pred_cam_t.detach(),
                                                        torch.from_numpy(joint[idx:idx+1]).cuda(), #Jtr[:,:22].detach().cuda(), #self.keypoints_3d, #new_opt_joints[:,:22], # torch.from_numpy(joint[:1, 1:23]).cuda(), #
                                                        conf_3d=confidence_input.to(self.device),
                                                        seq_ind=idx
                                                        )
            joints.append(new_opt_joints)
            vertices.append(new_opt_vertices)
        joints = torch.cat(joints, axis=0)
        vertices = torch.cat(vertices, axis=0)
        return joints, vertices