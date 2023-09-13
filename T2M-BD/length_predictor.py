import os 
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
from torch.distributions import Categorical
import json
import clip

import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from dataset import dataset_TM_train
from dataset import dataset_TM_eval
from dataset import dataset_tokenize
import models.t2m_trans as trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
from exit.utils import get_model, visualize_2motions, generate_src_mask, uniform, cosine_schedule
from einops import rearrange, repeat
import os
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
# os.environ["CUDA_LAUNCH_BLOCKING"]="1"
# os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"

##### ---- Network ---- #####
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

# https://github.com/openai/CLIP/issues/111
class TextCLIP(torch.nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model
        
    def forward(self,text):
        return self.model.encode_text(text)
clip_model = TextCLIP(clip_model)



class Temp:
    def __init__(self):
        print('mock:: opt')
args = Temp()
args.dataname = 't2m'
args.nb_code = 8192 # 512 # 
args.code_dim = 32 # 512 # 
args.batch_size = 2048
args.down_t = 2
num_workers = 8
codebook_dir = '/home/epinyoan/git/MaskText2Motion/T2M-BD/output/vq/2023-07-19-04-17-17_12_VQVAE_20batchResetNRandom_8192_32/codebook'
train_loader = dataset_TM_train.DATALoader(args.dataname, args.batch_size, args.nb_code, codebook_dir, unit_length=2**args.down_t, num_workers=num_workers)
train_loader_iter = dataset_TM_train.cycle(train_loader)



def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class LengthPredictorCLIP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        nd = 512
        dropout_p = 0.1
        self.output = nn.Sequential(
            nn.Dropout(p=dropout_p, inplace=False),
            nn.Linear(input_size, nd),
            nn.LayerNorm(nd),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(p=dropout_p, inplace=False),
            nn.Linear(nd, nd // 2),
            nn.LayerNorm(nd // 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(p=dropout_p, inplace=False),
            nn.Linear(nd // 2, nd // 4),
            nn.LayerNorm(nd // 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(p=dropout_p, inplace=False),
            nn.Linear(nd // 4, output_size)
        )

    def forward(self, word_embs):
        return self.output(word_embs)
    



from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')
val_loader = dataset_TM_eval.DATALoader(args.dataname, True, 32, w_vectorizer)




def eval_testset():
    correct = 0
    total = 0
    for word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name in tqdm(val_loader):
        text = clip.tokenize(clip_text, truncate=True).cuda()
        feat_clip_text = clip_model(text).float()
        pred_prob_len = len_predictor(feat_clip_text)
        pred_prob_len = pred_prob_len.argsort(dim=-1, descending=True)[:, 0]
        correct += torch.isclose(pred_prob_len*4, m_length.cuda(), atol= 4).sum()
        total += pred_prob_len.shape[0]
    return correct/total





from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./output/length_predictor/7_drpo.1_noise+-1')
len_predictor = LengthPredictorCLIP(512, 50)
len_predictor = torch.nn.DataParallel(len_predictor)
len_predictor.cuda()
crossEntropy = torch.nn.CrossEntropyLoss()
softmax = torch.nn.Softmax(-1)
optimizer = torch.optim.Adam(len_predictor.parameters(), lr=1e-4)

__log_acc_epoch = 20

num_iter = len(train_loader)
for epoch in tqdm(range(3000)):
    avg_loss = 0
    correct = 0
    total = 0
    for clip_text, target, m_tokens_len in train_loader:
        text = clip.tokenize(clip_text, truncate=True).cuda()
        feat_clip_text = clip_model(text).float()
        pred_prob_len = len_predictor(feat_clip_text)
        # pred_prob_len = softmax(pred_prob_len)
        
        noise = (torch.rand(m_tokens_len.shape[0]) * 3).int() - 1
        m_tokens_len = (m_tokens_len + noise).clamp(max=49)
        loss = crossEntropy(pred_prob_len, m_tokens_len.cuda())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(len_predictor.parameters(), 0.5)
        optimizer.step()
        # scheduler.step()
        
        avg_loss += loss/num_iter
        
        if epoch % __log_acc_epoch == 0:
            pred_tok_len = pred_prob_len.argsort(dim=-1, descending=True)[:, 0]
            correct += torch.isclose(pred_tok_len, m_tokens_len.cuda(), atol= 1).sum()
            total += pred_tok_len.shape[0]

    writer.add_scalar('./Loss/all', avg_loss, epoch)
    if epoch % __log_acc_epoch == 0:
        writer.add_scalar('./acc', correct/total, epoch)
        writer.add_scalar('./acc_test', eval_testset(), epoch)