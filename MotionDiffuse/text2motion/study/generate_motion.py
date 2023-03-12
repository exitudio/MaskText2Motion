import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import torch
import numpy as np
import utils.paramUtil as paramUtil
from utils.skeleton import Skeleton
from mylib import get_motion, animate3d, get_SMPL_layer, get6d, coco_bone, t2m_bone, smpl_coco_bone, smpl_smpl_bone, smpl_human36_bone, get_neutral_angle_offset, SkeletonWithHead
from utils.motion_process import recover_from_ric, recover_from_rot, recover_root_rot_pos
from utils.quaternion import *
import pickle
from tqdm import tqdm
##############################
#### See "Add COCO head".ipynb
##############################

coco_head_bone_vis = [(15,22),(22,24),(24,26),(22,23),(23,25)]
smpl_cocohead_bone_vis = t2m_bone + (np.array(coco_head_bone_vis)).tolist()

smpl_data = pickle.load(open('/home/epinyoan/git/smplpytorch/smplpytorch/native/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl', 'rb'), encoding='latin1')
smpl_j_regressor = np.array(smpl_data['J_regressor'].toarray()).astype(np.float32)
coco_head_regressor = np.load('/home/epinyoan/git/Pose2Mesh_RELEASE/data/COCO/J_regressor_coco.npy').astype(np.float32)
coco_head_regressor = coco_head_regressor[:5]

rest_pose = torch.zeros((1,24,3))
verts, rest_joints = get_SMPL_layer(rest_pose.view(1,72), display=False)

coco_head_joints = torch.matmul(torch.from_numpy(coco_head_regressor), verts)
smpl_coco_rest_joints = torch.cat((rest_joints[:,:22], coco_head_joints), axis=1).numpy()

smpl_cocohead_chain = paramUtil.t2m_kinematic_chain.copy()
smpl_cocohead_chain.append([15,22,24,26])
smpl_cocohead_chain.append([22,23,25])

smpl_coco_angle_offset = get_neutral_angle_offset(smpl_cocohead_chain, torch.from_numpy(smpl_coco_rest_joints[0]))


# see training text here: https://github.com/EricGuo5513/HumanML3D/blob/main/HumanML3D/texts.zip
all_text = [
    'a person runs forward very happily.',
    'a person dances and runs',
    'a person dances and walks',
    'a person dances and lies down',
    'a person dances and runs in circle',
    'a person dances and walks in circle',
    'a person dances strongly',
    'a person dances and jumps',
    'a person jumps forward', 
    'a person dances slowly',
    'a person dances slowly and jumps',
    'a person dances diagonally',
    'a person runs diagonally',
    'a person lies down diagonally',
    'a person lies down and run diagonally', # good
    'a person runs in circle', # good
    'a person lies down and walk in circle',
    'a person lies down and run in circle',
    'bending down and walking in a circle',
    'a person crawls',
    'a person crawls and dances',
    'a person crawls and runs',
    'a person walks forward, twirls around',
    'a person runs forward, twirls around',
    'a person runs and dances, twirls around',
    'a person dances and lies down, twirls around',
    'a person dances and jumps, twirls around',
    'a person walks backwards fast.',
    'a person walks backwards slowly.',
    'a person runs backwards fast.',
    'a person runs backwards and dances.',
    'a person runs backwards and jumps.',
    'a person runs backwards and lies down.',
    'a person kicks his leg and runs.',
    'a person kicks his leg and walks.',
    'a person kicks his leg and dances.',
    'a person kicks his leg and crawls, spins around',
    'a person plays basketball',
    'a person plays basketball and walks',
    'a person plays basketball and runs',
    'a person plays basketball and crawls',
    'a person plays basketball and dances, spins around',
    'a person plays basketball and lies down',
    'a person plays basketball and jumps',
    'a person plays basketball and kicks legs',
    'a person plays basketball and runs backward',
    'a person plays basketball, twirls around',
    'a person plays tennis',
    'a person plays tennis and walks',
    'a person plays tennis and runs',
    'a person plays tennis and crawls',
    'a person plays tennis and dances, spins around',
    'a person plays tennis and lies down',
    'a person plays tennis and jumps',
    'a person plays tennis and kicks legs',
    'a person plays tennis and runs backward',
    'a person plays tennis, twirls around',
    'a person throws baseball',
    'a person plays baseball and walks',
    'a person plays baseball and runs',
    'a person plays baseball and crawls',
    'a person plays baseball and dances, spins around',
    'a person plays baseball and lies down',
    'a person plays baseball and jumps',
    'a person plays baseball and kicks legs',
    'a person plays baseball and runs backward',
    'a person plays baseball, twirls around',
    'a person performs arm exercises and walks',
    'a person performs arm exercises and walks',
    'a person performs arm exercises and runs',
    'a person performs arm exercises and crawls',
    'a person performs arm exercises and dances, spins around',
    'a person performs arm exercises and lies down',
    'a person performs arm exercises and jumps',
    'a person performs arm exercises and kicks legs',
    'a person performs arm exercises and runs backward',
    'a person performs arm exercises, twirls around',
    'the person walks forward straight and backwards diagonally', # stand still for some time.
    'run forward and backward', # good
    'a man is jumping rope, then starts to jump with one leg.',
    'a person takes 6 steps forward and slightly towards the left.', # good (walk long and straight line)
    'a person walks forward with confidence.', # good (but similiar to "6 steps forward above")
    
    'a person sitting down gets up the walks forward, then turns around the walks back and sits back down.', # sit down too long
    'someone walks forward two steps and kicks their right leg up to turn around and walk two steps.',
    'a person walks back and forth down a stretch',
    'a person holds both hands up towards their face, as if cradling a baby while running',
    'a person runs by switching their legs around and swinging their arms', 
    'a person skips several steps to the right, several steps to the left and then two steps to the right.',
    'a person skips two steps to the right, walks two step forward and then two steps backward.',

    'the man moves his arms as if swimming', # only arms
    'a person is walking forward while kicking up their heels.',   # only legs
    'a person is stepping down stairs backwards',  # updown stairs
    'a man walks upstair',
    'the man squats and jumps in the air twice.', # move all the time but in one place
    'a person dances happily.',
]
#### WORK ####
# 'a person runs in circle',
# 'bending down and walking in a circle',
# 'the person walks forward straight and backwards diagonally',
# 'run forward and backward',
# 'a man is jumping rope, then starts to jump with one leg.',
# 'a person takes 6 steps forward and slightly towards the left.',
# 'a person walks forward with confidence.',
# 'a person sitting down gets up the walks forward, then turns around the walks back and sits back down.',



# 'the man moves his arms as if swimming', # only arms
# 'a person is walking forward while kicking up their heels.',   # only legs
# 'a person is stepping down stairs backwards',  # updown stairs
# 'the man squats and jumps in the air twice.',
# 'a man is jumping rope, then starts to jump with one leg.',
# 'a person dances happily.',

#### NOT WORK ####
# 'a person run backward'
# 'a person bends their knees and jumps twice straight up into the air.'
# 'a person slowly walked backwards',
# 'person jumping up and down',
# 'a person bends their knees and jumps twice straight up into the air.',
# 'a man walks counter-clockwise, in a circle',
# 'the person walked around them to the right and sat on the ground.',

joint_bone_offset_with_head = np.array([[[ 0.00000000e+00,  9.55121338e-01,  0.00000000e+00],
        [ 5.86734451e-02,  8.73442292e-01, -2.80773938e-02],
        [-5.76139763e-02,  8.63710642e-01, -1.70633420e-02],
        [ 5.95374324e-04,  1.08162463e+00, -3.29593495e-02],
        [ 1.20798409e-01,  4.86864746e-01, -5.75403683e-02],
        [-1.30074486e-01,  4.79168892e-01, -2.18810290e-02],
        [ 4.89652995e-03,  1.22376072e+00, -1.83410645e-02],
        [ 9.82723087e-02,  6.41056597e-02, -1.27340659e-01],
        [-9.73565802e-02,  5.92820942e-02, -1.00050956e-01],
        [-3.42542725e-03,  1.28268707e+00, -6.61644153e-03],
        [ 1.88876316e-01,  2.10969802e-02, -3.00352126e-02],
        [-1.86998084e-01,  1.16647929e-02,  1.13460310e-02],
        [-3.58090624e-02,  1.49391329e+00, -9.10619646e-03],
        [ 5.75786494e-02,  1.40547514e+00, -2.03006566e-02],
        [-9.88451019e-02,  1.38665199e+00, -4.34524193e-03],
        [-3.13895531e-02,  1.57071495e+00,  6.18931204e-02],
        [ 1.86988711e-01,  1.40378714e+00, -5.05600348e-02],
        [-2.14389592e-01,  1.37033343e+00,  9.25319083e-03],
        [ 2.11999640e-01,  1.14693761e+00, -9.89032611e-02],
        [-2.22496182e-01,  1.10441017e+00, -3.94403422e-03],
        [ 2.45336756e-01,  9.04465139e-01, -2.19690800e-02],
        [-2.52992153e-01,  8.57719541e-01,  1.05465584e-01],
        [-2.94180214e-02,  1.63646424e+00,  1.71774805e-01],
        [ 4.28068265e-03,  1.66511786e+00,  1.28545433e-01],
        [-6.37341440e-02,  1.66465974e+00,  1.26472920e-01],
        [ 4.77300659e-02,  1.63412261e+00,  3.30393836e-02],
        [-1.10418573e-01,  1.63457990e+00,  3.17121409e-02]]])
coco_id = [22,24,23,26,25,17,16,19,18,21,20,2,1,5,4,8,7]
def remove_freeze_frames(joint_dataset):
    moving_score = []
    for f in range(1, joint_dataset.shape[0]):
        compared_center = joint_dataset[f] - joint_dataset[f, 0:1]
        compared_center_prev = joint_dataset[f-1] - joint_dataset[f-1, 0:1]
        m = compared_center - compared_center_prev
        moving_score.append(m.max())
    moving_score = np.array(moving_score)
    np.argmax(moving_score>5e-3)
    s = np.flatnonzero(moving_score>5e-3)
    if len(s) == 0:
        return joint_dataset
    return joint_dataset[s[0]:s[-1]+2] # +2 b/c 1. step from F=>T and 2. [a:b+1]

if __name__ == "__main__":

    import glob
    for path in tqdm(sorted(glob.glob('/home/epinyoan/git/HumanML3D/HumanML3D/new_joints/*'))):
        
        joint_dataset = np.load(path)
        joint_dataset = remove_freeze_frames(joint_dataset)
        if joint_dataset.shape[0] < 60:
            continue

        head_quat = np.zeros((joint_dataset.shape[0],5,3))
        joint_with_head = np.concatenate((joint_dataset, head_quat), axis=1)

        # 1. get body quatanion
        tgt_skel = SkeletonWithHead(smpl_coco_angle_offset, smpl_cocohead_chain, 'cpu')
        quat_params = tgt_skel.inverse_kinematics_np(joint_with_head, face_joint_idx=[2, 1, 17, 16], head_chain=smpl_cocohead_chain[5:])

        # 2.1 IK
        smpl_coco_quat = torch.from_numpy(quat_params).float()

        # 2.2 FK
        tgt_offsets = tgt_skel.get_offsets_joints(torch.from_numpy(joint_bone_offset_with_head[0]))
        joint_forward = tgt_skel.forward_kinematics(smpl_coco_quat, root_pos=torch.from_numpy(joint_dataset[:,0]))

        file_name = path.split('/')[-1]
        np.save('HumanML3D/'+file_name, joint_forward[:, coco_id])  
        # animate3d(joint_forward[:, coco_id].numpy(), coco_bone, save_path='HumanML3D_html/'+file_name.split('.')[0]+'.html')




    # for i, text in tqdm(enumerate(all_text)):
    #     for j in range(10):
    #         data = torch.from_numpy(get_motion(text, motion_length=196)).float()
    #         joint_dataset = recover_from_ric(data, joints_num=22).numpy()
    #         joint_dataset = remove_freeze_frames(joint_dataset)
    #         if joint_dataset.shape[0] < 60:
    #             continue

    #         head_quat = np.zeros((joint_dataset.shape[0],5,3))
    #         joint_with_head = np.concatenate((joint_dataset, head_quat), axis=1)

    #         # 1. get body quatanion
    #         tgt_skel = SkeletonWithHead(smpl_coco_angle_offset, smpl_cocohead_chain, 'cpu')
    #         quat_params = tgt_skel.inverse_kinematics_np(joint_with_head, face_joint_idx=[2, 1, 17, 16], head_chain=smpl_cocohead_chain[5:])

    #         # 2.1 IK
    #         smpl_coco_quat = torch.from_numpy(quat_params).float()

    #         # 2.2 FK
    #         tgt_offsets = tgt_skel.get_offsets_joints(torch.from_numpy(joint_bone_offset_with_head[0]))
    #         joint_forward = tgt_skel.forward_kinematics(smpl_coco_quat, root_pos=torch.from_numpy(joint_dataset[:,0]))

    #         file_name = 'output/'+str(i)+'_'+str(j)+'_'+text
    #         np.save(file_name+'.npy', joint_forward[:, coco_id])  
    #         animate3d(joint_forward[:, coco_id].numpy(), coco_bone, save_path='output_html/'+str(i)+'_'+str(j)+'_'+text+'.html')

    #     # smpl_smplcoco_bone = t2m_bone + (np.array(smpl_cocohead_bone_vis)+22).tolist()
    #     # joint_original_smplcoco = np.concatenate((joint_dataset[:,:22], joint_forward.cpu().numpy()), axis=1)
    #     # animate3d(joint_original_smplcoco, smpl_smplcoco_bone, first_total_standard=63, save_path='output/'+str(i)+'_'+text+'.html')

