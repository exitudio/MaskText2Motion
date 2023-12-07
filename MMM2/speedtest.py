from utils.word_vectorizer import WordVectorizer
from dataset import dataset_TM_eval
from tqdm import tqdm
from exit.utils import base_dir
from speedtest_models.exit2m import run_speed_test
from speedtest_models.speedtest_main import run_speed_test_all
import os
import numpy as np
speedtest_data_dir = '/home/epinyoan/git/MaskText2Motion/T2M-BD/speedtest_models/speedtest_data.npy'
if not os.path.isfile(speedtest_data_dir):
    speed_test_data = []
    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    val_loader = dataset_TM_eval.DATALoader('t2m', True, 1, w_vectorizer, shuffle=False)
    for word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name in tqdm(val_loader):
        speed_test_data.append([clip_text, m_length, word_embeddings, pos_one_hots, sent_len])
    np.save(speedtest_data_dir, speed_test_data)

run_speed_test_all(run_speed_test, 'exit2m-cross-attn1lyr_15Steps')