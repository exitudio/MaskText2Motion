from mask_model.util import generate_samples, get_model
from mask_model.motiontransformer import generate_src_mask
import torch

class GenMotion:
    def __init__(self, transformer, quantize, decoder, textEncoder, generator_dim, max_motion_length, device):
        print('GenMotion')
        self.encoder = transformer.eval()
        self.quantize = quantize.eval()
        self.decoder = decoder.eval()
        self.textEncoder = textEncoder.eval()
        self.generator_dim = generator_dim
        self.max_motion_length = max_motion_length
        self.device = device

    def generate(self, caption, m_lens, dim_pose, batch_size=512):
        N = len(caption)
        cur_idx = 0
        self.encoder.eval()
        all_output = []
        while cur_idx < N:
            if cur_idx + batch_size >= N:
                batch_caption = caption[cur_idx:]
                batch_m_lens = m_lens[cur_idx:]
            else:
                batch_caption = caption[cur_idx: cur_idx + batch_size]
                batch_m_lens = m_lens[cur_idx: cur_idx + batch_size]
            output = self.generate_batch(batch_caption, batch_m_lens, dim_pose)
            B = output.shape[0]

            for i in range(B):
                all_output.append(output[i])
            cur_idx += batch_size
        return all_output
    
    def generate_batch(self, all_caption, all_m_lens, dim_pose):
        text_emb = self.textEncoder(all_caption, cond_mask_prob=0)
        text_emb = text_emb.view(text_emb.shape[0], -1, self.generator_dim)
        gen_indices = generate_samples(
            self.encoder, self.device, text_emb, steps=self.max_motion_length)
        x = get_model(self.quantize).embedding(gen_indices.reshape(-1))
        x = x.reshape(*gen_indices.shape, -1)

        B = len(all_caption)
        T = self.max_motion_length
        length = torch.LongTensor([min(T, m_len) for m_len in  all_m_lens]).to(self.device)
        src_mask = generate_src_mask(T, length).to(text_emb.device).unsqueeze(-1)
        num_heads = 8
        src_mask.shape, B
        src_mask_attn = src_mask.view(B, 1, 1, T).repeat(1, num_heads, T, 1)
        x = x * src_mask

        x = self.decoder(x, src_mask = src_mask_attn)
        # [TODO] steps=all_m_lens.max()
        # Filter unused length (to compair to gt)
        return x

    def load(*args):
        print('mocked load()')
        return 0, 0
    def eval_mode(*args):
        print('mocked eval_mode()')
    def to(*args):
        print('mocked to()')

from mask_model.mingpt import GPT
from mmcv.parallel import MMDataParallel
from mask_model.quantize import VectorQuantizer2
from mask_model.motiontransformer import MotionTransformerOnly2
from mask_model.train_transformer import TextEncoder

def build_eval_model(opt, vq_path, transformer_path):
    generator_dim = 256
    text_emb_dim = 512
    codebook_dim = 32
    dim_pose = opt.dim_pose
    latent_dim = 256
    # vq_path = '/home/epinyoan/git/MaskText2Motion/MotionDiffuse/text2motion/checkpoints/kit/2023-03-04-20-39-40_13_vqgan_poseformer_1DisScale/'
    # transformer_path = '/home/epinyoan/git/MaskText2Motion/MotionDiffuse/text2motion/checkpoints/kit/2023-03-10-10-22-39_5_transformer_textCond_fixZeroCond_13_vqgan_poseformer_1DisScale/'

    transformer = GPT(vocab_size=8192, block_size=196+int(text_emb_dim/generator_dim), 
                      n_layer=8, n_head=8, n_embd=generator_dim)
    transformer.load_state_dict(torch.load(transformer_path+'transformer.pth'))
    transformer = MMDataParallel(transformer.cuda(), device_ids=opt.gpu_id)
    transformer.eval()

    #
    quantize = VectorQuantizer2(n_e = 8192, e_dim = codebook_dim)
    quantize.load_state_dict(torch.load(vq_path+'/quantize.pth'))
    quantize = MMDataParallel(quantize.cuda(), device_ids=opt.gpu_id)

    decoder = MotionTransformerOnly2(input_feats=codebook_dim, 
                                    output_feats=dim_pose, 
                                    latent_dim=latent_dim, 
                                    num_layers=8)
    decoder.load_state_dict(torch.load(vq_path+'/decoder.pth'))
    decoder = MMDataParallel(decoder.cuda(), device_ids=opt.gpu_id)

    textEncoder = TextEncoder(output_dim=text_emb_dim).cuda()
    textEncoder.load_state_dict(torch.load(transformer_path+'textEncoder.pth'))

    return GenMotion(transformer, quantize, decoder, textEncoder, generator_dim, opt.max_motion_length, opt.device)