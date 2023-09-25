from utils.word_vectorizer import WordVectorizer
from dataset import dataset_TM_eval
from tqdm import tqdm
import datetime
from exit.utils import base_dir

class SpeedInfo:
    def __init__(self, save_path):
        self.record = []
        self.save_path = save_path

    def start(self):
        self.start_time = datetime.datetime.now()
    
    def end(self, text, length):
        self.end_time = datetime.datetime.now()
        self.start_time
        info = {
            'time': (self.end_time-self.start_time).total_seconds(),
            'length': length,
            'text': text
        }
        self.record.append(info)
        print('info:', info)

    def save(self):
        from pathlib import Path
        Path('/'.join(self.save_path.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
        np.save(self.save_path, self.record)
    
    def reset(self):
        self.record = []

speed_info = SpeedInfo(f'{base_dir}/epinyoan/git/MaskText2Motion/T2M-BD/output/speedtest/exit2m.npy')

from speedtest_models.exit2m import run_speed_test
import os
import numpy as np
speedtest_data_dir = f'{base_dir}/epinyoan/git/MaskText2Motion/T2M-BD/speedtest_models/speedtest_data.npy'
if not os.path.isfile(speedtest_data_dir):
    speed_test_data = []
    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    val_loader = dataset_TM_eval.DATALoader('t2m', True, 1, w_vectorizer, shuffle=False)
    for word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name in tqdm(val_loader):
        speed_test_data.append([clip_text, m_length])
    np.save(speedtest_data_dir, speed_test_data)

speed_test_data = np.load(speedtest_data_dir, allow_pickle=True)
for i, batch in enumerate(tqdm(speed_test_data)):
    if i == 0:
        print('========== WARMUP ==========')
        run_speed_test(batch, speed_info)
        run_speed_test(batch, speed_info)
        run_speed_test(batch, speed_info)
        speed_info.reset()
        print('========== START ==========')
    run_speed_test(batch, speed_info)
speed_info.save()
print('done')