from datetime import datetime
from os.path import join as pjoin
import numpy as np
import subprocess
import json
import os


def listdir(src):
    names = os.listdir(src)
    return [pjoin(src, nm) for nm in names]


def index_best_saved_weights(path):
    # calculate a moving average over the validation losses
    # choose the minimum
    # find the closest weight to it
    
    # read the index and loss pairs
    idx_losses = list()
    with open(os.path.join(path, 'val_loss.txt'), 'rt') as txt:
        for line in txt:
            idxstr, lossstr = line.split()
            idx_losses.append((int(idxstr), float(lossstr)))
    # calculate moving average
    mv_avg_validation_losses = list()
    window_sum = np.sum(idx_losses[0:5])
    mv_avg_validation_losses.append((idx_losses[2][0], window_sum / 5.0))
    for il in range(3, len(idx_losses)-2):
        window_sum = window_sum - idx_losses[il - 3][1] + idx_losses[il + 2][1]
        mv_avg_validation_losses.append((idx_losses[il][0], window_sum / 5.0))
    # finding the minimum
    best_option = min(mv_avg_validation_losses, key=lambda x: x[1])
    # find the closest weight
    closest_weight = None
    distance = None
    weight_files = [fn for fn in os.listdir(path) if fn.endswith('.pt')]
    for wf in weight_files:
        w_idx = int(wf.split('_')[-1][0:-3])
        temp = abs(w_idx - best_option[0])
        if distance is None or temp < distance:
            distance = temp
            closest_weight = wf
    return closest_weight


def maximum_index_in_saved_weights(path):
    idx = max([int(nm.split('_')[2][0:-3]) for nm in os.listdir(path) if nm.endswith('.pt')])
    return "model_state_{}.pt".format(idx)


def git_commit_and_gethash(commit_msg=None):
    if not(commit_msg is None):
        subprocess.call(["git", "add", "--all"])
        subprocess.call(["git", "commit", "-m", commit_msg])
    com_hash = subprocess.getoutput("git rev-parse HEAD")
    return com_hash


def clear_weight_files_from_experiment_campaign(path):
    experiment_folders = os.listdir(path)
    for exp_folder in experiment_folders:
        # delete from folders
        view_folders = ["phasesa", "phase2ch", "phase4ch", "phaselvot", "phase_lvhyp_model"]
        for vf in view_folders:
            path_exp_view = os.path.join(path, exp_folder, vf)
            files = os.listdir(path_exp_view)
            for fl in files:
                if fl.endswith(".pt"):
                    path_to_file = os.path.join(path_exp_view, fl)
                    os.remove(path_to_file)


class HyperParameterSaver:

    def __init__(self, path):
        self.path = path
        self.store = dict()
        self.store['current_date'] = str(datetime.now())
    
    def add_parameter(self, name, value):
        self.store[name] = value

    def save(self):
        with open(self.path, "wt") as js:
            json.dump(self.store, js)


class AutoParameterTuner:

    def __init__(self):
        self.configurations = list()
    
    def add_config(self, name, config):
        self.configurations.append([name, config])
    
    def next_configuration(self):
        for conf in self.configurations:
            yield conf


class EarlyStopping:

    def __init__(self, window_size=5, waiting_steps=5):
        self.moving_avg = None
        self.window_size = window_size
        self.waiting_steps = waiting_steps
        self.memory = list()
        self.best_moving_avg = None
        self.steps_until_better_movingavg = 0
    
    def add_loss(self, loss):
        # recalculate the moving average
        if self.moving_avg is None:
            self.memory = [loss] * self.window_size
            self.moving_avg = loss
            self.best_moving_avg = self.moving_avg
        else:
            self.moving_avg = (self.moving_avg * self.window_size - self.memory[0] + loss) / self.window_size
            del self.memory[0]
            self.memory.append(loss)
        # check whether the validation loss is descreasing
        if self.best_moving_avg > self.moving_avg:
            self.best_moving_avg = self.moving_avg
            self.steps_until_better_movingavg = 0
        else:
            self.steps_until_better_movingavg += 1
        # check whether the improvement happened to late
        return self.steps_until_better_movingavg > self.waiting_steps


if __name__ == '__main__':
    clear_weight_files_from_experiment_campaign(r'D:\AI\works\Heart\data\hypertrophy\aim_results\lhyp_results\resnetbased\20210705')
