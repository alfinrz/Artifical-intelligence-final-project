 # importing libraries
import os
import glob
import json
import copy
import numpy as np
import torch
from torch.utils.data import Dataset
from dataset import *

class PedestrianDataset(Dataset): # class for storing pedestrian dataset
    """
    Dataset of pedestrian trajectories.
    """
    def __init__(self, path, seq_len, obs_hist, min_seq_len):
        super().__init__()
        self.path = path
        self.seq_len = seq_len
        self.min_seq_len = min_seq_len
        self.obs_hist = obs_hist

        self.timestamps, self.paths = None, None
        self.dataset_name = None
        self.data_samples = []
        self.dataset_size = 0

        if self.path is not None:
            self.init_dataset()

    def init_dataset(self): # initialize dataset
        expanded_path = os.path.join(self.path, 'data', '*.json')
        paths = glob.glob(expanded_path)
        assert(len(paths) > 0)
        self.timestamps, self.paths = self.order_timestamp_paths(paths)
        assert(len(self.timestamps) == len(self.paths))

        self.set_name(self.path)
        self.create_sequences()
        self.dataset_size = len(self.data_samples)

    def __len__(self):
        return self.dataset_size
        
    def order_timestamp_paths(self, paths): # order timestamp paths
        ordered_detections = []
        for path in paths:
            with open(path, 'r') as file:
                detection = json.load(file)
            ordered_detections.append((detection['timestamp'], path))

        ordered_detections.sort()
        timestamps, new_paths = map(list, zip(*ordered_detections))
        return timestamps, new_paths

    def create_sequences(self): # create sequences
        detections = self.load_detections()
        samples = self.create_samples(detections)
        self.data_samples = list(samples.values())
        self.data_samples = self.slice_sequences()

    def create_samples(self, detections): # create samples
        samples = dict()

        for detection in detections:
            timestamp = detection.timestamp
            for obj in detection.objects():
                if obj.id not in samples:
                    samples[obj.id] = SampleData(obj.id, timestamp)
                sample = samples[obj.id]
                sample.add_position(obj.position)
        return samples

    def load_detections(self): # load detections
        detections = []
        for path in self.paths:
            detection = self.load_detection(path)
            detections.append(detection)
        return detections

    def load_detection(self, path): # load detection
        with open(path, 'r') as file:
            detection_json = json.load(file)
        detection = DetectionData.from_json(detection_json)
        return detection

    def slice_sequences(self): # slice sequences
        sliced_samples = []
        for sample in self.data_samples:
            slices = sample.slice(self.seq_len, self.min_seq_len)
            sliced_samples.extend(slices)
        return sliced_samples

    def __getitem__(self, index): # get item
        sample = copy.deepcopy(self.data_samples[index])

        mask = self.compute_mask(sample)[self.obs_hist:]
        mask = [torch.tensor(x) for x in mask]
        mask = torch.stack(mask)
        self.pad_sequence(sample)

        observed = torch.tensor(sample.trajectory.positions[:self.obs_hist], dtype=torch.float32)
        y_trajectory = np.array(sample.trajectory.positions)
        y_delta = y_trajectory[self.obs_hist:] - y_trajectory[self.obs_hist-1:-1]
        y_delta = torch.tensor(y_delta, dtype=torch.float32)

        return observed, [y_delta, mask]

    def compute_mask(self, sample): # compute mask
        mask = np.ones(self.seq_len)
        mask[len(sample):] = 0.
        return mask

    def pad_sequence(self, sample): # pad sequence
        if len(sample) >= self.seq_len:
            return

        padding_size = (self.seq_len) - len(sample)
        for _ in range(padding_size):
            sample.add_position([0, 0])

    def set_name(self, path): # set name
        info_path = os.path.join(path, 'dataset_info.json')
        with open(info_path, 'r') as file:
            dataset_info = json.load(file)
        self.dataset_name = dataset_info['dataset_name']