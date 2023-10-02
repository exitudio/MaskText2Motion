from tqdm import tqdm
import datetime
import numpy as np
import os
data_permission = os.access('/data/epinyoan', os.R_OK | os.W_OK | os.X_OK)
base_dir = '/data' if data_permission else '/home'

class SpeedInfo:
    def __init__(self, model_name):
        self.model_name = model_name
        self.record = []
        self.time = []
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.save_path = f'{base_dir}/epinyoan/git/MaskText2Motion/T2M-BD/output/speedtest/{date}_{model_name}/'

    def start(self):
        self.start_time = datetime.datetime.now()
    
    def end(self, text, length, pred_len=None):
        self.end_time = datetime.datetime.now()
        self.start_time
        diff_time = (self.end_time-self.start_time).total_seconds()
        info = {
            'time': diff_time,
            'length': length,
            'text': text,
            'pred_len': pred_len
        }
        self.time.append(diff_time)
        self.record.append(info)
        print('info:', info)

    def save(self):
        from pathlib import Path
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        np.save(f'{self.save_path}/all.npy', self.record)
        f = open(f'{self.save_path}/speed.txt', "a")
        f.write("Avg speed:"+str(np.array(self.time).mean()))
        f.write("Total sample:"+str(len(self.time)))
        f.close()
    def reset(self):
        self.record = []
        self.time = []

def run_speed_test_all(run_speed_test, model_name):
    speed_info = SpeedInfo(model_name)
    speed_test_data = np.load('/home/epinyoan/git/MaskText2Motion/T2M-BD/speedtest_models/speedtest_data.npy', allow_pickle=True)
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