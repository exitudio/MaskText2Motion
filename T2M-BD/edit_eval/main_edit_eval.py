import torch
from tqdm import tqdm
import utils.eval_trans as eval_trans

def eval_inbetween(eval_wrapper, logger, val_loader, call_model, nb_iter):
    num_repeat = 1
    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    nb_sample = 0
    
    count = 0
    for word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name in tqdm(val_loader):
        bs, seq = pose.shape[:2]
        motion_multimodality_batch = []
        for i in range(num_repeat):
            pred_pose_eval = call_model(clip_text, pose, m_length)
            
            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length)
            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))

            if i == 0:
                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                temp_R, temp_match = eval_trans.calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = eval_trans.calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs
                ### end if
            ### end for
            motion_multimodality.append(torch.cat(motion_multimodality_batch, dim=1))
        # if count > 2:
        #     break
        count += 1
    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = eval_trans.calculate_activation_statistics(motion_annotation_np)
    mu, cov= eval_trans.calculate_activation_statistics(motion_pred_np)

    diversity_real = eval_trans.calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = eval_trans.calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    multimodality = 0
    motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
    if num_repeat > 1:
        multimodality = eval_trans.calculate_multimodality(motion_multimodality, 10)

    fid = eval_trans.calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, \n\
                FID. {fid:.4f} , \n\
                Diversity Real. {diversity_real:.4f}, \n\
                Diversity. {diversity:.4f}, \n\
                R_precision_real. {R_precision_real}, \n\
                R_precision. {R_precision}, \n\
                matching_score_real. {matching_score_real}, \n\
                matching_score_pred. {matching_score_pred}, \n\
                multimodality. {multimodality:.4f}"
    logger.info(msg)
    return fid, diversity, R_precision, matching_score_pred, multimodality