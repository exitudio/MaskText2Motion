import models.vqvae as vqvae
import models.t2m_trans as trans
import torch
import clip
from exit.utils import base_dir

class Temp:
    def __init__(self):
        print('mock:: opt')
args = Temp()
args.dataname = 't2m'
args.nb_code = 8192
args.code_dim = 32
args.output_emb_width = 512
args.down_t = 2
args.stride_t = 2
args.width = 512
args.depth = 3
args.dilation_growth_rate = 3

args.embed_dim_gpt = 1024
args.clip_dim = 512
args.block_size = 51
args.num_layers = 9
args.n_head_gpt = 16
args.drop_out_rate = 0.1
args.ff_rate = 4

args.quantizer = "ema_reset"
args.mu = 0.99

args.resume_pth = f'/{base_dir}/epinyoan/git/MaskText2Motion/T2M-BD/output/vq/2023-07-19-04-17-17_12_VQVAE_20batchResetNRandom_8192_32/net_last.pth'
args.resume_trans = f'/{base_dir}/epinyoan/git/MaskText2Motion/T2M-BD/output/t2m/2023-08-04-23-08-30_HML3D_36_token1stStage_cdim8192_32_lr0.0001_mask.5-1/net_last.pth'

net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                    args.nb_code,
                    args.code_dim,
                    args.output_emb_width,
                    args.down_t,
                    args.stride_t,
                    args.width,
                    args.depth,
                    args.dilation_growth_rate)
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

trans_encoder = trans.Text2Motion_Transformer(vqvae=net,
                                num_vq=args.nb_code, 
                                embed_dim=args.embed_dim_gpt, 
                                clip_dim=args.clip_dim, 
                                block_size=args.block_size, 
                                num_layers=args.num_layers, 
                                n_head=args.n_head_gpt, 
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate)
ckpt = torch.load(args.resume_trans, map_location='cpu')
trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.eval()
trans_encoder.cuda()


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

def run_speed_test(batch, speed_info):
    clip_text, m_length = batch
    k = 0
    
    speed_info.start()
    text = clip.tokenize(clip_text, truncate=True).cuda()
    feat_clip_text = clip_model(text).float()

    m_tokens_len = torch.ceil((m_length)/4)
    pred_len = m_length.cuda()
    pred_tok_len = m_tokens_len
    index_motion = trans_encoder(feat_clip_text, type="sample", m_length=pred_len, rand_pos=False, CFG=-1)

    pred_pose = net(index_motion[k:k+1, :int(pred_tok_len[k].item())], type='decode')
    speed_info.end(clip_text, m_length)