import sys
sys.path.append('/home/epinyoan/git/smplpytorch/')
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os
import torch
import argparse
from os.path import join as pjoin

import utils.paramUtil as paramUtil
from torch.utils.data import DataLoader
from utils.plot_script import *
from utils.get_opt import get_opt
from datasets.evaluator_models import MotionLenEstimatorBiGRU

from trainers import DDPMTrainer
from models import MotionTransformer
from utils.word_vectorizer import WordVectorizer, POS_enumerator
from utils.utils import *
from utils.motion_process import recover_from_ric, recover_from_rot, recover_root_rot_pos, quaternion_to_cont6d
from utils.skeleton import Skeleton
# from smplpytorch.pytorch.smpl_layer import SMPL_Layer
# from display_utils import display_model
import copy
from utils.quaternion import *
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel


def build_models(opt):
    encoder = MotionTransformer(
        input_feats=opt.dim_pose,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim,
        no_clip=opt.no_clip,
        no_eff=opt.no_eff)
    return encoder


class Args:
    opt_path = '../checkpoints/t2m/t2m_motiondiffuse/opt.txt'
    text = 'the person walks forward straight and backwards diagonally'
    result_path = ''
    gpu_id = 0


def get_motion(text, motion_length=120):
    assert motion_length <= 196
    with torch.no_grad():
        caption = [text]
        m_lens = torch.LongTensor([motion_length]).to(device)
        pred_motions = trainer.generate(caption, m_lens, opt.dim_pose)
        motion = pred_motions[0].cpu().numpy()
        motion = motion * std + mean
        title = args.text + " #%d" % motion.shape[0]
    return motion


def get_range(skeleton, index):
    _min, _max = skeleton[:, :, index].min(), skeleton[:, :, index].max()
    return [_min, _max], _max-_min


t2m_bone = [[0,2], [2,5],[5,8],[8,11],
            [0,1],[1,4],[4,7],[7,10],
            [0,3],[3,6],[6,9],[9,12],[12,15],
            [9,14],[14,17],[17,19],[19,21],
            [9,13],[13,16],[16,18],[18,20]]
kit_bone = [[0, 11], [11, 12], [12, 13], [13, 14], [14, 15], [0, 16], [16, 17], [17, 18], [18, 19], [19, 20], [0, 1], [1, 2], [2, 3], [3, 4], [3, 5], [5, 6], [6, 7], [3, 8], [8, 9], [9, 10]]
coco_bone = [[15, 13],[13, 11],[16, 14],[14, 12],[11, 12],[ 5, 11],[ 6, 12],[ 5, 6 ],[ 5, 7 ],[ 6, 8 ],[ 7, 9 ],[ 8, 10],[ 1, 2 ],[ 0, 1 ],[ 0, 2 ],[ 1, 3 ],[ 2, 4 ],[ 3, 5 ],[ 4, 6 ]]
human36_skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
smpl_coco_bone = t2m_bone + (np.array(coco_bone)+22).tolist()
smpl_human36_bone = t2m_bone + (np.array(human36_skeleton)+22).tolist()
smpl_smpl_bone = t2m_bone + (np.array(t2m_bone)+22).tolist()
kit_kit_bone = kit_bone + (np.array(kit_bone)+21).tolist()

def axis_standard(skeleton):
    skeleton = skeleton.copy()
#     skeleton = -skeleton
    # skeleton[:, :, 0] *= -1
    # xyz => zxy
    skeleton[:, :, [1, 2]] = skeleton[:, :, [2, 1]]
    skeleton[:, :, [0, 1]] = skeleton[:, :, [1, 0]]
    return skeleton

def animate3d(skeleton, BONE_LINK=t2m_bone, first_total_standard=-1, save_path=None, axis_standard=axis_standard):
    # [animation] https://community.plotly.com/t/3d-scatter-animation/46368/6
    
    SHIFT_SCALE = 0
    START_FRAME = 0
    NUM_FRAMES = skeleton.shape[0]
    skeleton = skeleton[START_FRAME:NUM_FRAMES+START_FRAME]
    skeleton = axis_standard(skeleton)
    if BONE_LINK is not None:
        # ground truth
        bone_ids = np.array(BONE_LINK)
        _from = skeleton[:, bone_ids[:, 0]]
        _to = skeleton[:, bone_ids[:, 1]]
        # [f 3(from,to,none) d]
        bones = np.empty(
            (_from.shape[0], 3*_from.shape[1], 3), dtype=_from.dtype)
        bones[:, 0::3] = _from
        bones[:, 1::3] = _to
        bones[:, 2::3] = np.full_like(_to, None)
        display_points = bones
        mode = 'lines+markers'
    else:
        display_points = skeleton
        mode = 'markers'
    # frames = [go.Frame(data=[go.Scatter3d(
    #     x=display_points[k, :, 0],
    #     y=display_points[k, :, 1],
    #     z=display_points[k, :, 2],
    #     mode=mode,
    #     marker=dict(size=3, ))],
    #     traces=[0],
    #     name=f'frame{k}'
    # )for k in range(len(display_points))]
    # print('display_points:', display_points.shape)
    # if first_total_standard == -1:
    #     data=[
    #         go.Scatter3d(x=display_points[0, :, 0], y=display_points[0, :, 1],
    #                      z=display_points[0, :, 2], mode=mode, marker=dict(size=3, )),
    #     ]
    # else:
    #     data=[
    #             go.Scatter3d(x=display_points[0, :63, 0], y=display_points[0, :63, 1],
    #                         z=display_points[0, :63, 2], mode=mode, marker=dict(size=3, color='blue',)),
    #             go.Scatter3d(x=display_points[0, 63:, 0], y=display_points[0, 63:, 1],
    #                         z=display_points[0, 63:, 2], mode=mode, marker=dict(size=3, color='red',)),
    #         ]
    # fig = go.Figure(
    #     data=data, layout=go.Layout(scene=dict(aspectmode='data', camera=dict(eye=dict(x=3, y=0, z=0.1)))))
    
    # follow this thread: https://community.plotly.com/t/3d-scatter-animation/46368/6
    fig = go.Figure(
        data=go.Scatter3d(  x=display_points[0, :first_total_standard, 0], 
                            y=display_points[0, :first_total_standard, 1],
                            z=display_points[0, :first_total_standard, 2], 
                            name='Nodes0',
                            mode=mode, 
                            marker=dict(size=3, color='blue',)), 
                            layout=go.Layout(
                                scene=dict(aspectmode='data', 
                                camera=dict(eye=dict(x=3, y=0, z=0.1)))
                                )
                            )
    if first_total_standard != -1:
        fig.add_traces(data=go.Scatter3d(  
                                x=display_points[0, first_total_standard:, 0], 
                                y=display_points[0, first_total_standard:, 1],
                                z=display_points[0, first_total_standard:, 2], 
                                name='Nodes1',
                                mode=mode, 
                                marker=dict(size=3, color='red',)))

    frames = []
    # frames.append({'data':copy.deepcopy(fig['data']),'name':f'frame{0}'})

    def update_trace(k):
        fig.update_traces(x=display_points[k, :first_total_standard, 0],
            y=display_points[k, :first_total_standard, 1],
            z=display_points[k, :first_total_standard, 2],
            mode=mode,
            marker=dict(size=3, ),
            # traces=[0],
            selector = ({'name':'Nodes0'}))
        if first_total_standard != -1:
            fig.update_traces(x=display_points[k, first_total_standard:, 0],
                y=display_points[k, first_total_standard:, 1],
                z=display_points[k, first_total_standard:, 2],
                mode=mode,
                marker=dict(size=3, ),
                # traces=[0],
                selector = ({'name':'Nodes1'}))

    for k in range(0, len(display_points)):
        update_trace(k)
        frames.append({'data':copy.deepcopy(fig['data']),'name':f'frame{k}'})
    update_trace(0)

    # frames = [go.Frame(data=[go.Scatter3d(
    #     x=display_points[k, :, 0],
    #     y=display_points[k, :, 1],
    #     z=display_points[k, :, 2],
    #     mode=mode,
    #     marker=dict(size=3, ))],
    #     traces=[0],
    #     name=f'frame{k}'
    # )for k in range(len(display_points))]
    
    
    
    fig.update(frames=frames)

    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    sliders = [
        {"pad": {"b": 10, "t": 60},
         "len": 0.9,
         "x": 0.1,
         "y": 0,

         "steps": [
            {"args": [[f.name], frame_args(0)],
             "label": str(k),
             "method": "animate",
             } for k, f in enumerate(fig.frames)
        ]
        }
    ]

    fig.update_layout(
        updatemenus=[{"buttons": [
            {
                "args": [None, frame_args(1000/25)],
                "label": "Play",
                "method": "animate",
            },
            {
                "args": [[None], frame_args(0)],
                "label": "Pause",
                "method": "animate",
            }],

            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "type": "buttons",
            "x": 0.1,
            "y": 0,
        }
        ],
        sliders=sliders
    )
    range_x, aspect_x = get_range(skeleton, 0)
    range_y, aspect_y = get_range(skeleton, 1)
    range_z, aspect_z = get_range(skeleton, 2)

    fig.update_layout(scene=dict(xaxis=dict(range=range_x,),
                                 yaxis=dict(range=range_y,),
                                 zaxis=dict(range=range_z,)
                                 ),
                      scene_aspectmode='manual',
                      scene_aspectratio=dict(
                          x=aspect_x, y=aspect_y, z=aspect_z)
                      )

    fig.update_layout(sliders=sliders)
    fig.show()
    if save_path is not None:
        fig.write_html(save_path)

def visualize_2motions(motion1, motion2, std, mean, dataset_name, length, save_path=None):
    motion1 = motion1 * std + mean
    motion2 = motion2 * std + mean
    if dataset_name == 'kit':
        first_total_standard = 60
        bone_link = kit_kit_bone
        joints_num = 21
        scale = 1/1000
    else:
        raise NameError('dataset:' + dataset_name)
    joint1 = recover_from_ric(torch.from_numpy(motion1).float(), joints_num).numpy()
    joint2 = recover_from_ric(torch.from_numpy(motion2).float(), joints_num).numpy()
    joint_original_forward = np.concatenate((joint1, joint2), axis=1)
    animate3d(joint_original_forward[:length]*scale, 
              BONE_LINK=bone_link, 
              first_total_standard=first_total_standard, 
              save_path=save_path) # 'init.html'

def get_SMPL_layer(pose_params, display=True):
    pose_params = pose_params[:1].reshape((-1,72)).float()
    batch_size = 1
    shape_params = torch.rand(batch_size, 10) * 0.03
    smpl_layer = SMPL_Layer(
            # center_idx=0,
            gender='male',
            model_root='/home/epinyoan/git/smplpytorch/smplpytorch/native/models')
    verts, Jtr = smpl_layer(pose_params, th_betas=shape_params)
    print(smpl_layer.th_J_regressor.shape)
    plt.close()
    if display:
        ax = display_model(
                {'verts': verts.cpu().detach(),
                'joints': Jtr.cpu().detach()},
                model_faces=smpl_layer.th_faces,
                with_joints=True,
                kintree_table=smpl_layer.kintree_table,
                savepath=None,
                show=True)
    return verts, Jtr

def get6d(data):
    joints_num = 22
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)
    return cont6d_params

def get_neutral_angle_offset(_kinematic_tree, joints):
    _parents = [0] * len(joints)
    _parents[0] = -1
    for chain in _kinematic_tree:
        for j in range(1, len(chain)):
            _parents[chain[j]] = chain[j-1]
            
    angle_offset = torch.zeros(joints.shape)
    for i in range(1, angle_offset.shape[0]):
        angle_offset[i] = joints[i] - joints[_parents[i]]
        angle_offset[i] = angle_offset[i]/torch.linalg.norm(angle_offset[i] , ord=2)
    return angle_offset

class SkeletonWithHead(Skeleton):
    def inverse_kinematics_np(self, joints, face_joint_idx, smooth_forward=False, head_chain=None):
        assert len(face_joint_idx) == 4
        '''Get Forward Direction'''
        l_hip, r_hip, sdr_r, sdr_l = face_joint_idx
        across1 = joints[:, r_hip] - joints[:, l_hip]
        across2 = joints[:, sdr_r] - joints[:, sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across**2).sum(axis=-1))[:, np.newaxis]
        # print(across1.shape, across2.shape)

        # forward (batch_size, 3)
        forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        if smooth_forward:
            forward = filters.gaussian_filter1d(forward, 20, axis=0, mode='nearest')
            # forward (batch_size, 3)
        forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]

        '''Get Root Rotation'''
        target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
        root_quat = qbetween_np(forward, target)

        '''Inverse Kinematics'''
        # quat_params (batch_size, joints_num, 4)
        # print(joints.shape[:-1])
        quat_params = np.zeros(joints.shape[:-1] + (4,))
        # print(quat_params.shape)
        root_quat[0] = np.array([[1.0, 0.0, 0.0, 0.0]])
        quat_params[:, 0] = root_quat
        # quat_params[0, 0] = np.array([[1.0, 0.0, 0.0, 0.0]])
        for chain in self._kinematic_tree:
            R = root_quat
            for j in range(len(chain) - 1):
                if head_chain is not None and chain in head_chain:
                    R_loc = qmul_np(qinv_np(R), qinv_np(root_quat))
                else:
                    # (batch, 3)
                    u = self._raw_offset_np[chain[j+1]][np.newaxis,...].repeat(len(joints), axis=0)
                    # print(u.shape)
                    # (batch, 3)
                    v = joints[:, chain[j+1]] - joints[:, chain[j]]
                    v = v / np.sqrt((v**2).sum(axis=-1))[:, np.newaxis]
                    # print(u.shape, v.shape)
                    rot_u_v = qbetween_np(u, v)

                    R_loc = qmul_np(qinv_np(R), rot_u_v)

                quat_params[:,chain[j + 1], :] = R_loc
                R = qmul_np(R, R_loc)

        return quat_params

def display_animate2D(data, w, h):
    def animate(frame_num):
        ax.clear()
        x = data[frame_num,:,0]
        y = -data[frame_num,:,1]
        ax.set_xlim(0, w)
        ax.set_ylim(-h, 0)
        ax.scatter(x,y)
        for i, bone in enumerate(coco_bone):
            ax.plot([x[bone[0]], x[bone[1]]], [y[bone[0]], y[bone[1]]], 'r')
    plt.close()
    fig, ax = plt.subplots()
    animate(0)
    return FuncAnimation(fig, animate, frames=data.shape[0], interval=20)


# move this initialize out of __main__ if you wanna use "get_motion()"
if __name__ == "__main__":
    args = Args()
    device = torch.device('cuda')
    opt = get_opt(args.opt_path, device)
    opt.do_denoise = True

    assert opt.dataset_name == "t2m"
    # opt.data_root = './dataset/HumanML3D'
    # opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
    # opt.text_dir = pjoin(opt.data_root, 'texts')
    # opt.joints_num = 22
    opt.dim_pose = 263
    # dim_word = 300
    dim_pos_ohot = len(POS_enumerator)
    # num_classes = 200 // opt.unit_length
    # result_dict = {}


    mean = np.load('../checkpoints/t2m/t2m_motiondiffuse/meta/mean.npy')
    std = np.load('../checkpoints/t2m/t2m_motiondiffuse/meta/std.npy')

    encoder = build_models(opt).to(device)
    # encoder = MMDataParallel(
    #             encoder, device_ids=[0,1,2,3])
    trainer = DDPMTrainer(opt, encoder)
    trainer.load('../checkpoints/t2m/t2m_motiondiffuse/model/latest.tar')

    trainer.eval_mode()
    trainer.to(opt.device)