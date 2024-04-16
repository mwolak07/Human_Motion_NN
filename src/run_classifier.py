from typing import Callable, Tuple, List
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torch.optim import Optimizer
from torch import Tensor
from tqdm import tqdm
from torch import nn
import numpy as np
import torch
import time
import os

from utils import txt_list_append, txt_list_read


def train_classifier(train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
                     model: nn.Module, optimizer: Optimizer, criterion: Callable, epochs: int,
                     save_dir: str, resume: bool = False) -> None:
    """
    Train a classifier model on the given training and validation sets.

    Args:
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        device: Device to perform training and validation on.
        model: Classifier model to be trained.
        optimizer: Optimizer to be used for training.
        criterion: Loss function to be used for training.
        epochs: Number of epochs to train for.
        save_dir: Directory to save checkpoints, logs, and best model.
        resume: Whether to resume training from the checkpoint dir or not.
    """
    # Setup.
    model.to(device)
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    log_dir = os.path.join(save_dir, 'logs')
    os.makedirs(checkpoint_dir, exist_ok=False)
    os.makedirs(log_dir, exist_ok=False)
    epoch = 0
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    # Resume training if needed.
    if resume:
        epoch, model, optimizer, train_loss_list, train_acc_list, val_loss_list, val_acc_list =\
            load_checkpoint(model, optimizer, checkpoint_dir, log_dir)
    # Train for the given number of epochs.
    for epoch in range(epoch, epochs):
        t = time.time()
        # Iterate over the training samples.
        model, optimizer, train_loss, train_acc = train_classifier_step(train_loader, device,
                                                                        model, optimizer, criterion)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        # Iterate over the validation samples.
        val_loss, val_acc = eval_classifier_step(val_loader, device,
                                                 model, criterion)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        # Print stats on a new line.
        print(f'Epoch [{epoch + 1}/{epochs}]: train Loss {train_loss:.4f}, train acc {train_acc:.4f}, '
              f'val loss {val_loss:.4f}, val acc {val_acc:.4f}, time {time.time() - t:.4f}s')
        # Save checkpoint.
        save_checkpoint(epoch, model, optimizer,
                         train_loss, train_acc, val_loss, val_acc,
                         checkpoint_dir, log_dir)
    # Save best model. This is the model with the highest validation accuracy.
    best_epoch = np.argmax(val_acc_list)
    save_model(model, best_epoch, checkpoint_dir, save_dir)
    # Save training plots.
    save_training_plots(train_loss_list, train_acc_list, val_loss_list, val_acc_list, log_dir)


def train_classifier_step(dataloader: DataLoader, device: torch.device,
                          model: nn.Module, optimizer: Optimizer, criterion: Callable) \
        -> Tuple[nn.Module, Optimizer, float, float]:
    """

    Args:
        dataloader: DataLoader for the training set.
        device: Device to perform training and validation on.
        model: Classifier model to be trained.
        optimizer: Optimizer to be used for training.
        criterion: Loss function to be used for training.

    Returns:
        The model with updated weights.
        The optimizer with an updated state.
        The average training loss for the epoch.
        The average training accuracy for the epoch.

    """
    model.train()
    total_loss, total_acc, total_count = 0, 0, 0
    for x, label in tqdm(dataloader, total=len(dataloader)):
        # Sanity check for types.
        assert isinstance(x, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        x = x.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        # Forward pass.
        logits = model(x)
        loss = criterion(logits, label)
        # Backward pass.
        loss.backward()
        optimizer.step()
        # Compute and store metrics.
        total_loss += loss.item()
        acc = (logits.argmax(1) == label).sum()
        total_acc += acc.item()
        total_count += label.size(0)
    # Compute final metrics.
    train_loss = total_loss / total_count
    train_acc = total_acc / total_count
    # Return trained model, updated optimizer, and metrics.
    return model, optimizer, train_loss, train_acc


def eval_classifier_step(dataloader: DataLoader, device: torch.device,
                         model: nn.Module, criterion: Callable) -> Tuple[float, float]:
    """

    Args:
        dataloader: DataLoader for the evaluation dataset.
        device: Device to evaluation on.
        model: Trained classifier model.
        criterion: Loss function to be used for training.

    Returns:
        The average training loss for the epoch.
        The average training accuracy for the epoch.
    """
    model.eval()
    total_loss, total_acc, total_count = 0, 0, 0
    with torch.no_grad():
        for i, (x, label) in enumerate(dataloader):
            # Sanity check for types.
            assert isinstance(x, torch.Tensor)
            assert isinstance(label, torch.Tensor)
            # Initialize batch.
            x = x.to(device)
            label = label.to(device)
            # Forward pass.
            logits = model(x)
            # Compute and store metrics.
            total_loss += criterion(logits, label).item()
            total_acc += (logits.argmax(1) == label).sum().item()
            total_count += label.size(0)
    # Compute final metrics.
    val_loss = total_loss / total_count
    val_acc = total_acc / total_count
    # Return metrics.
    return val_loss, val_acc


def save_checkpoint(epoch: int, model: nn.Module, optimizer: Optimizer,
                     train_loss: float, train_acc: float, val_loss: float, val_acc: float,
                     checkpoint_dir: str, log_dir: str) -> None:
    """
    Saves a checkpoint of the model, optimizer, and statistics.

    Args:
        epoch: The current epoch number.
        model: Classifier model to be trained.
        optimizer: Optimizer to be used for training.
        train_loss: Training loss for the current epoch.
        train_acc: Training accuracy for the current epoch.
        val_loss: Validation loss for the current epoch.
        val_acc: Validation accuracy for the current epoch.
        checkpoint_dir: Path to save model weight checkpoints.
        log_dir: Path to save stats logs.
    """
    # Save the model checkpoint.
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint-{epoch}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc},
        checkpoint_path)
    # Save the stats to log files.
    txt_list_append(os.path.join(log_dir, 'train_loss.txt'), train_loss)
    txt_list_append(os.path.join(log_dir, 'train_acc.txt'), train_acc)
    txt_list_append(os.path.join(log_dir, 'val_loss.txt'), val_loss)
    txt_list_append(os.path.join(log_dir, 'val_acc.txt'), val_acc)


def load_checkpoint(model: nn.Module, optimizer: Optimizer, checkpoint_dir: str, log_dir: str) \
        -> Tuple[int, nn.Module, Optimizer, List[float], List[float], List[float], List[float]]:
    """
    Loads a checkpoint of the model, optimizer, and statistics.

    Args:
        model: Classifier model with uninitialized weights.
        optimizer: Optimizer to with uninitialized state.
        checkpoint_dir: Path where model weight checkpoints are saved.
        log_dir: Path where stats logs are saved.

    Returns:
        A tuple of everything that needs to be restored, as follows:
        - epoch
        - model with weights restored
        - optimizer with state restored
        - list of prior training losses
        - list of prior training accuracies
        - list of prior validation losses
        - list of prior validation accuracies
    """
    # Get the latest model checkpoint.
    checkpoint_paths = os.listdir(checkpoint_dir)
    checkpoint_paths.sort()
    checkpoint_path = checkpoint_paths[-1]
    # Restore the data from the model checkpoint.
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    # Restore the stats.
    train_loss_list = txt_list_read(os.path.join(log_dir, 'train_loss.txt'), float)
    train_acc_list = txt_list_read(os.path.join(log_dir, 'train_acc.txt'), float)
    val_loss_list = txt_list_read(os.path.join(log_dir, 'val_loss.txt'), float)
    val_acc_list = txt_list_read(os.path.join(log_dir, 'val_acc.txt'), float)
    # Return our results.
    return epoch, model, optimizer, train_loss_list, train_acc_list, val_loss_list, val_acc_list


def save_model(model: nn.Module, epoch: int, checkpoint_dir: str, save_dir: str) -> None:
    """
    Loads the model from the given epoch from the checkpoint_dir and saves it to the save_dir.

    Args:
        model: Classifier model to be saved.
        epoch: Epoch number of the checkpoint to load.
        checkpoint_dir: Path to save model weight checkpoints.
        save_dir: Path to the directory to save best model to.
    """
    # Load the checkpoint at that epoch.
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint-{epoch}.pt')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # Save the model in its entirety to the save dir.
    save_path = os.path.join(save_dir, f'model-{epoch}.pt')
    torch.save(model, save_path)


def save_training_plots(train_loss_list: List[float], train_acc_list: List[float],
                         val_loss_list: List[float], val_acc_list: List[float],
                         log_dir: str) -> None:
    """
    Plots the training and validation loss, along with the validation accuracy, and saves the plots to log_dir.

    Args:
        train_loss_list: List of model training loss for each epoch.
        train_acc_list: List of model training accuracy for each epoch.
        val_loss_list: List of model validation loss for each epoch.
        val_acc_list: List of model validation accuracy for each epoch.
        log_dir: Directory in which to save the loss and accuracy plots.
    """
    epochs = list(range(1, len(train_loss_list) + 1))
    # Plot loss.
    loss_path = os.path.join(log_dir, 'loss.png')
    plt.plot(epochs, train_loss_list, label='Training loss', color='blue')
    plt.plot(epochs, val_loss_list, label='Validation loss', color='orange')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss')
    plt.savefig(loss_path)
    plt.clf()
    # Plot accuracy
    accuracy_path = os.path.join(log_dir, 'accuracy.png')
    plt.plot(epochs, train_acc_list, label='Training accuracy', color='blue')
    plt.plot(epochs, val_acc_list, label='Validation accuracy', color='orange')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Accuracy')
    plt.savefig(accuracy_path)


def test_classifier(test_loader: DataLoader, device: torch.device, save_dir: str, model_class: object) \
        -> Tuple[List[Tensor], float]:
    """
    Test the classifier model on the given test set, loading the best model from the given save directory.

    Args:
        test_loader: DataLoader for the test set.
        device: Device to perform testing on.
        save_dir: Directory where the weights for the best model were saved.
        model_class: Model class to be tested. Needs to be defined somewhere for loading to work.
    """
    assert isinstance(model_class, object)
    # Get the model save path.
    model_file = [file for file in os.listdir(save_dir) if '.pt' in file][0]
    save_path = os.path.join(save_dir, model_file)
    # Load the model from the save path.
    model = torch.load(save_path)
    model.to(device)
    # Get the predictions for the test set.
    model.eval()
    total_acc, total_count = 0, 0
    pred_list = []
    with torch.no_grad():
        for i, (x, label) in enumerate(test_loader):
            # Initialize batch.
            x = x.to(device)
            label = label.to(device)
            # Forward pass.
            logits = model(x)
            # Compute and store metrics.
            batch_pred = torch.argmax(logits, dim=1)
            pred_list += [pred for pred in batch_pred]
            total_acc += (batch_pred == label).sum().item()
            total_count += label.size(0)
    test_acc = total_acc / total_count
    print(f'Test Acc: {test_acc:.4f}')
    return pred_list, test_acc
