# importing libraries
import torch
import torch.nn as nn
import torch.optim as optim

from metrics import *
from pedestrian import *

import matplotlib.pyplot as plt

class RunConfig: # this is a class that contains all the hyperparameters
    sequence_length = 20
    min_sequence_length = 10
    observed_history = 8

    sample = False
    num_samples = 20
    sample_angle_std = 25

    dataset_paths = [
                     "./data/eth_univ",
                     "./data/eth_hotel",
                     "./data/ucy_zara01",
                     "./data/ucy_zara02",
                     "./data/ucy_univ"
                    ]

def load_datasets(): # this function loads the datasets
    datasets = []
    datasets_size = 0
    for dataset_path in RunConfig.dataset_paths:
        if 'HOME' not in os.environ:
            os.environ['HOME'] = os.environ['USERPROFILE']
        dataset_path = dataset_path.replace('~', os.environ['HOME'])
        print("Loading dataset {}".format(dataset_path))
        dataset = PedestrianDataset(path=dataset_path, seq_len=RunConfig.sequence_length, obs_hist=RunConfig.observed_history, \
                              min_seq_len=RunConfig.min_sequence_length)
        datasets.append(dataset)
        datasets_size += len(dataset)
    print("Size of all datasets: {}".format(datasets_size))
    return datasets

class ConstantVelocityModel(nn.Module): # this is a class that contains the constant velocity model
    def __init__(self, input_size, output_size):
        super(ConstantVelocityModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, observed): # this function computes the forward pass
        obs_rel = observed[1:] - observed[:-1]
        deltas = obs_rel[-1].unsqueeze(0)
        y_pred_rel = deltas.repeat(12, 1, 1)
        return self.linear(y_pred_rel.view(y_pred_rel.size(0), -1))

def train_cvm_model(model, train_loader, criterion, optimizer, num_epochs=10): # this function trains the CVM model
    loss_val = []
    
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            observed = batch_x.permute(1, 0, 2)
            y_true_rel, masks = batch_y
            y_true_rel = y_true_rel.permute(1, 0, 2)

            y_pred_rel = model(observed)
            loss = criterion(y_pred_rel, y_true_rel)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss}")
        loss_val.append(avg_loss)
    return loss_val

# Use this function to create a DataLoader for your training dataset
def create_train_loader(trainset, batch_size):
    return torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)

# This function plots the graphs for the results of the training
def plot_training_history(train_loss):
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train_loss, "bo-", label='Training Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()

def main():
    # Load datasets
    datasets = load_datasets()

    # Choose a dataset for training (modify this part as needed)
    train_dataset = datasets[0]
    train_loader = create_train_loader(train_dataset, batch_size=1)

    #Define Input and output size
    input_size = 2
    output_size = 2
    # Initialize the CVM model, criterion, and optimizer
    cvm_model = ConstantVelocityModel(input_size, output_size)
    criterion = nn.MSELoss()  # You may need to choose an appropriate loss function
    optimizer = optim.Adam(cvm_model.parameters(), lr=0.001)  # Adjust the learning rate as needed

    # Train the CVM model and store the result in a variable
    train_loss = train_cvm_model(cvm_model, train_loader, criterion, optimizer, num_epochs=10)

    # Plots the loss graph
    plot_training_history(train_loss)


    # Now you can use the trained model for evaluation or other tasks
if __name__ == "__main__":
    main()
