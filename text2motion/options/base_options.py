import argparse
import os
import torch
from mmcv.runner import init_dist, get_dist_info
import torch.distributed as dist
import shutil
import datetime

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--name', type=str, default="test", help='Name of this trial')
        self.parser.add_argument('--decomp_name', type=str, default="Decomp_SP001_SM001_H512", help='Name of autoencoder model')

        self.parser.add_argument("--gpu_id", type=int, nargs='+', default=(-1), help='GPU id')
        self.parser.add_argument("--distributed", action="store_true", help='Whether to use DDP training')
        self.parser.add_argument("--data_parallel", action="store_true", help="Whether to use DP training")

        self.parser.add_argument('--dataset_name', type=str, default='t2m', help='Dataset Name')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

        self.parser.add_argument("--unit_length", type=int, default=4, help="Motions are cropped to the maximum times of unit_length")
        self.parser.add_argument("--max_text_len", type=int, default=20, help="Maximum length of text description")

        self.parser.add_argument('--text_enc_mod', type=str, default='bigru')
        self.parser.add_argument('--estimator_mod', type=str, default='bigru')

        self.parser.add_argument('--dim_text_hidden', type=int, default=512, help='Dimension of hidden unit in text encoder')
        self.parser.add_argument('--dim_att_vec', type=int, default=512, help='Dimension of attention vector')
        self.parser.add_argument('--dim_z', type=int, default=128, help='Dimension of latent Gaussian vector')

        self.parser.add_argument('--n_layers_pri', type=int, default=1, help='Number of layers in prior network')
        self.parser.add_argument('--n_layers_pos', type=int, default=1, help='Number of layers in posterior network')
        self.parser.add_argument('--n_layers_dec', type=int, default=1, help='Number of layers in generator')

        self.parser.add_argument('--dim_pri_hidden', type=int, default=1024, help='Dimension of hidden unit in prior network')
        self.parser.add_argument('--dim_pos_hidden', type=int, default=1024, help='Dimension of hidden unit in posterior network')
        self.parser.add_argument('--dim_dec_hidden', type=int, default=1024, help='Dimension of hidden unit in generator')

        self.parser.add_argument('--dim_movement_enc_hidden', type=int, default=512,
                                 help='Dimension of hidden in AutoEncoder(encoder)')
        self.parser.add_argument('--dim_movement_dec_hidden', type=int, default=512,
                                 help='Dimension of hidden in AutoEncoder(decoder)')
        self.parser.add_argument('--dim_movement_latent', type=int, default=512, help='Dimension of motion snippet')

        self.initialized = True


        # EXIT option
        self.parser.add_argument("--project", default="project_name")
        self.parser.add_argument('--debug', type=str2bool, nargs='?', default=True)
        self.parser.add_argument('--corrupt', type=str, default='diffusion', help='diffusion or mask')

    def init_save_folder(self):
        self.opt.raw_name = self.opt.name
        if self.opt.debug:
            self.opt.name = 'TEMP'
        else:
            date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self.opt.name = f'{date}_{self.opt.name}'
            save_folder = f"./{self.opt.checkpoints_dir}/{self.opt.dataset_name}/{self.opt.name}/"
            os.makedirs(save_folder, exist_ok=False)
            for f in ['datasets', 'experiments', 'models', 'options', 'tools', 'trainers', 'utils', 'mask_model']:
                shutil.copytree(f, f'{save_folder}/{f}', ignore=shutil.ignore_patterns('__pycache__'))
            

    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()
        self.init_save_folder()

        self.opt.is_train = self.is_train

        args = vars(self.opt)
        if args["distributed"]:
            init_dist('slurm')
        rank, world_size = get_dist_info()
        if args["distributed"]:
            self.opt.gpu_id = range(world_size)
        elif self.opt.gpu_id != (-1):
            if len(self.opt.gpu_id) == 1:
                torch.cuda.set_device(self.opt.gpu_id[0])
        else:
            assert args["data_parallel"] == False

        if rank == 0:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')
            if self.is_train:
                # save to the disk
                expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.dataset_name, self.opt.name)
                if not os.path.exists(expr_dir):
                    os.makedirs(expr_dir)
                file_name = os.path.join(expr_dir, 'opt.txt')
                with open(file_name, 'wt') as opt_file:
                    opt_file.write('------------ Options -------------\n')
                    for k, v in sorted(args.items()):
                        opt_file.write('%s: %s\n' % (str(k), str(v)))
                    opt_file.write('-------------- End ----------------\n')
        if world_size > 1:
            dist.barrier()
        return self.opt
