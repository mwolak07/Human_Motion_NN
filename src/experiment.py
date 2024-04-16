from torch.utils.data import DataLoader
from datetime import datetime
from torch.optim import Adam
from torch import nn
import torch
import os

from h2h_subjects_dataset import H2HSubjectsDataset
from simple_rnn import SimpleRNNClassifier
from train import train_classifier
from utils import data_split


# Set up the experiment.
batch_size = 128
epochs = 2
lr = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
experiment_dir = os.path.join(os.getcwd(), '../experiments', datetime.now().strftime('%d-%m-%Y_%H-%M-%S'))

# Set up the dataset.
# session_files = ['E:/Datasets/CS 4440 Final Project/mat_files_full/26_M4_F5_cropped_data_v2.mat']
session_files = ['E:/Datasets/CS 4440 Final Project/mat_files_full/test_data.mat']
sequence_length = 32
joint_groups_file = 'joint_groups.json'
joint_groups = ['spine', 'left arm', 'right arm', 'left leg', 'right leg']
dataset = H2HSubjectsDataset(session_files, sequence_length, joint_groups_file, joint_groups)

# TODO: DEBUG, remove later.
from torch.utils.data import Subset
dataset = Subset(dataset, range(10))

train_set, val_set, test_set, _ = data_split(dataset=dataset, test_split=0.2, val_split=0.2)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Set up the model.
model = SimpleRNNClassifier(19,
                            19,
                            2,
                            0.25,
                            19,
                            1,
                            2)
optimizer = Adam(model.parameters(), lr)
criterion = nn.CrossEntropyLoss()

# Train the model.
train_classifier(train_loader, val_loader, device, model, optimizer, criterion, epochs, experiment_dir)
