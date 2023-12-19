import os
import argparse
import json
import numpy as np
import torch.utils.data as Data
from metrics import *
from ped_dataset import PedestrianDataset

class Config:
    seq_len = 20
    min_seq_len = 10
    obs_hist = 8
    sample = False
    num_samples = 20
    sample_angle_std = 25
    paths = [
        "./data/eth_univ",
        "./data/eth_hotel",
        "./data/ucy_zara01",
        "./data/ucy_zara02",
        "./data/ucy_univ"
    ]

def relative_to_absolute(relative_traj, start_pos):
    relative_traj = relative_traj.permute(1, 0, 2)
    displacement = torch.cumsum(relative_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    absolute_traj = displacement + start_pos
    return absolute_traj.permute(1, 0, 2)

def constant_velocity(observed, sample=False):
    observed_relative = observed[1:] - observed[:-1]
    deltas = observed_relative[-1].unsqueeze(0)
    if sample:
        sampled_angle = np.random.normal(0, Config.sample_angle_std, 1)[0]
        theta = (sampled_angle * np.pi) / 180.
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrix = torch.tensor([[c, s], [-s, c]])
        deltas = torch.t(rotation_matrix.matmul(torch.t(deltas.squeeze(dim=0)))).unsqueeze(0)
    predicted_relative = deltas.repeat(12, 1, 1)
    return predicted_relative

def evaluate(dataset):
    loader = Data.DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    with torch.no_grad():
        avg_displacements = []
        final_displacements = []
        for seq_id, (x, y) in enumerate(loader):
            observed = x.permute(1, 0, 2)
            true_relative, masks = y
            true_relative = true_relative.permute(1, 0, 2)
            sample_avg_disp = []
            sample_final_disp = []
            samples_to_draw = Config.num_samples if Config.sample else 1
            for i in range(samples_to_draw):
                predicted_relative = constant_velocity(observed, sample=Config.sample)
                predicted_absolute = relative_to_absolute(predicted_relative, observed[-1])
                predicted_positions = predicted_absolute.permute(1, 0, 2)
                true_absolute = relative_to_absolute(true_relative, observed[-1])
                true_positions = true_absolute.permute(1, 0, 2)
                avg_displacement = avg_disp(predicted_positions, [true_positions, masks])
                final_displacement = final_disp(predicted_positions, [true_positions, masks])
                sample_avg_disp.append(avg_displacement)
                sample_final_disp.append(final_displacement)
            avg_displacement = min(sample_avg_disp)
            final_displacement = min(sample_final_disp)
            avg_displacements.append(avg_displacement)
            final_displacements.append(final_displacement)
        avg_displacements = np.mean(avg_displacements)
        final_displacements = np.mean(final_displacements)
        return avg_displacements, final_displacements

def load_datasets():
    datasets = []
    total_size = 0
    for path in Config.paths:
        if 'HOME' not in os.environ:
            os.environ['HOME'] = os.environ['USERPROFILE']
        path = path.replace('~', os.environ['HOME'])
        print(f"Loading dataset {path}")
        dataset = PedestrianDataset(path=path, seq_len=Config.seq_len, obs_hist=Config.obs_hist, min_seq_len=Config.min_seq_len)
        datasets.append(dataset)
        total_size += len(dataset)
    print(f"Size of all datasets: {total_size}")
    return datasets

def parse_args():
    parser = argparse.ArgumentParser(description='Runs an evaluation of the Constant Velocity Model.')
    parser.add_argument('--sample', default=Config.sample, action='store_true', help='Turns on the sampling for the CVM (OUR-S).')
    return parser.parse_args()

def main():
    args = parse_args()
    Config.sample = args.sample
    if Config.sample:
        print("Sampling activated.")
    datasets = load_datasets()
    results = []
    for i, dataset in enumerate(datasets):
        print(f"Evaluating dataset {dataset.name}")
        avg_displacements, final_displacements = evaluate(dataset)
        results.append([dataset.name, avg_displacements, final_displacements])
    print("\n== Results for dataset evaluations ==")
    total_avg_disp, total_final_disp = 0, 0
    for name, avg_displacements, final_displacements in results:
        print(f"- Dataset: {name}")
        print(f"ADE: {avg_displacements}")
        print(f"FDE: {final_displacements}")
        total_avg_disp += avg_displacements
        total_final_disp += final_displacements
    print("- Average")
    print(f"*ADE: {total_avg_disp/len(results)}")
    print(f"*FDE: {total_final_disp/len(results)}")

if __name__ == "__main__":
    main()
