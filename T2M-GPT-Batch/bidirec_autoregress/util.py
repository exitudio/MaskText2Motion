import torch
def get_bidirec_input(target, m_tokens_len):
    half = torch.floor((m_tokens_len+1)/2)

    no_input_mask = torch.ones_like(target, device=m_tokens_len.device).bool()
    no_input_mask.scatter_(-1, half.unsqueeze(-1).long() - 1, False)
    no_input_mask.scatter_(-1, half.unsqueeze(-1).long(), False)
    
    target[:, 1:-1] = target[no_input_mask].view(target.shape[0], -1)
    target[:, 0] = -1 # [INFO] first token = -1
    target.scatter_(1, m_tokens_len[:, None]-1, -2) # [INFO] last token = -2
    return target