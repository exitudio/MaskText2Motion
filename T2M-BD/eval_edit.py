import torch
import clip

import options.option_transformer as option_trans
import models.vqvae as vqvae
import models.t2m_trans as trans
import warnings
warnings.filterwarnings('ignore')
from exit.utils import get_model
from edit_eval.main_edit_eval import run_all_eval



##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

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
        with torch.no_grad():
            word_emb = self.model.token_embedding(text).type(self.model.dtype)
            word_emb = word_emb + self.model.positional_embedding.type(self.model.dtype)
            word_emb = word_emb.permute(1, 0, 2)  # NLD -> LND
            word_emb = self.model.transformer(word_emb)
            word_emb = self.model.ln_final(word_emb).permute(1, 0, 2).float()
            enctxt = self.model.encode_text(text).float()
        return enctxt, word_emb
clip_model = TextCLIP(clip_model)

net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate)


trans_encoder = trans.Text2Motion_Transformer(net,
                                num_vq=args.nb_code, 
                                embed_dim=args.embed_dim_gpt, 
                                clip_dim=args.clip_dim, 
                                block_size=args.block_size, 
                                num_layers=args.num_layers, 
                                num_local_layer=args.num_local_layer, 
                                n_head=args.n_head_gpt, 
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate)


print ('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

if args.resume_trans is not None:
    print ('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.train()
trans_encoder.cuda()



def call_T2MBD(clip_text, pose, m_length):
    clip_text = [''] * len(clip_text)
    text = clip.tokenize(clip_text, truncate=True).cuda()
    feat_clip_text, word_emb = clip_model(text)

    bs, seq = pose.shape[:2]
    tokens = -1*torch.ones((bs, 50), dtype=torch.long).cuda()
    m_token_length = torch.ceil((m_length)/4).int().cpu().numpy()
    m_token_length_init = (m_token_length * .25).astype(int)
    m_length_init = (m_length * .25).int()
    for k in range(bs):
        l = m_length_init[k]
        l_token = m_token_length_init[k]

        # start tokens
        index_motion = net(pose[k:k+1, :l].cuda(), type='encode')
        tokens[k,:index_motion.shape[1]] = index_motion[0]

        # end tokens
        index_motion = net(pose[k:k+1, m_length[k]-l :m_length[k]].cuda(), type='encode')
        tokens[k, m_token_length[k]-l_token :m_token_length[k]] = index_motion[0]

    mask_id = get_model(trans_encoder).num_vq + 2
    tokens[tokens==-1] = mask_id
    inpaint_index = trans_encoder(feat_clip_text, word_emb, type="sample", m_length=m_length.cuda(), token_cond=tokens)

    pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
    for k in range(bs):
        pred_pose = net(inpaint_index[k:k+1, :m_token_length[k]], type='decode')
        pred_pose_eval[k:k+1, :int(m_length[k].item())] = pred_pose
    
    return pred_pose_eval
run_all_eval(call_T2MBD, args.out_dir, args.exp_name)


# from instantmotion import InstantMotion
# from dataset import dataset_TM_eval
# from utils.word_vectorizer import WordVectorizer
# def call_InstantMotionUpper(clip_text, pose, m_length):
#     return instant_motion_upper.upper_edit(pose, m_length, clip_text)

# w_vectorizer = WordVectorizer('./glove', 'our_vab')
# val_loader = dataset_TM_eval.DATALoader('t2m', True, 32, w_vectorizer)
# instant_motion_upper = InstantMotion(is_upper_edit=True, 
#                                      extra_args = {'mean':val_loader.dataset.mean, 
#                                       'std':val_loader.dataset.std}).cuda()
# run_all_eval(call_InstantMotionUpper, args.out_dir, args.exp_name)