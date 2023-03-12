import torch 
import torch.nn.functional as F

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

class MeanMask():
    def __init__(self, src_mask, dim_pose):
        self.denom = src_mask.sum() * dim_pose

    def mean(self, x):
        return x.sum() / self.denom
    

def configure_optimizers(transformer, learning_rate):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95))
        return optimizer

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[..., [-1]]] = -float('Inf')
    return out

def generate_samples(transformer, device, condition_emb=None, num_samples=1, steps = 40):
    c_indices = torch.zeros(num_samples, 1 if condition_emb is None else 0, dtype=torch.int64).to(device)
    top_k = 100
    sample = True

    x = c_indices.clone()
    for k in range(steps):
        logits, _ = transformer(x, condition_emb)
        logits = logits[:, -1, :]
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        if sample:
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(logits, k=1, dim=-1)
        x = torch.cat((x, ix), dim=1)
    x = x[:, c_indices.shape[1]:] # remove condition
    return x

def get_model(model):
    if hasattr(model, 'module'):
        return model.module
    return model

def get_latest_folder_by_name(directory_path, name):
    import os
    import re
    from datetime import datetime
    # Define the regular expression pattern to match the file names
    pattern = r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_%s$" % name

    # Get the list of files in the directory that match the pattern
    matching_files = [f for f in os.listdir(directory_path) if re.match(pattern, f)]

    # Define a key function to extract the datetime from the file name
    def get_datetime_from_filename(file_name):
        return datetime.strptime(file_name.split("_")[0], "%Y-%m-%d-%H-%M-%S")

    # Get the latest file based on the datetime in the file name
    return f'{directory_path}/{max(matching_files, key=get_datetime_from_filename)}'