from torch.utils.data import DataLoader
from datetime import datetime
from torchinfo import summary
from torch.optim import Adam
from torch import nn
import torch
import os

from run_classifier import train_classifier, test_classifier
from h2h_two_roles_dataset import H2HTwoRolesDataset
from functional_unit_rnn import FunctionalUnitRNNClassifier
from utils import data_split


# Set up the experiment.
batch_size = 128
epochs = 10
lr = 0.001
grad_norm_clip = 1.0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
now = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
experiment_dir = os.path.normpath(os.path.join(os.getcwd(), os.pardir, 'experiments', now))
if os.path.exists(experiment_dir):
    print(f'Error: {experiment_dir} already exists! Delete this directory if you are sure you want to overwrite it!')
    exit()
os.makedirs(experiment_dir, exist_ok=False)

# Set up the dataset.
session_files = ['E:/Datasets/CS 4440 Final Project/mat_files_full/26_M4_F5_cropped_data_v2.mat']
sequence_length = 25
joint_groups_file = 'joint_groups.json'
joint_groups = ['spine', 'left arm', 'right arm', 'left leg', 'right leg']
print(f'Loading sessions...')
dataset = H2HTwoRolesDataset(session_files, sequence_length, joint_groups_file, joint_groups)
print()
train_set, val_set, test_set, _ = data_split(dataset=dataset, test_split=0.2, val_split=0.2)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Set up the model.
model = FunctionalUnitRNNClassifier(joint_groups_file,
                                    1.0,
                                    1,
                                    1.0,
                                    1,
                                    0.25,
                                    150,
                                    1,
                                    2)
print(f'Model summary:')
summary(model, input_size=(batch_size, sequence_length, 57))
print()
optimizer = Adam(model.parameters(), lr)
criterion = nn.CrossEntropyLoss()

# Train the model.
print(f'Training model on {device} with {len(train_loader) + len(val_loader)} batches of size {batch_size}...')
train_classifier(train_loader, val_loader, device, model, optimizer, criterion, epochs, grad_norm_clip, experiment_dir)
print()
# Test the model.
print(f'Testing model on {device} with {len(test_loader)} batches of size {batch_size}...')
pred_list, acc = test_classifier(test_loader, device, experiment_dir, FunctionalUnitRNNClassifier)
