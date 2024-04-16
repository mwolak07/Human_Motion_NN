from typing import Callable, Dict, List, Any
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torch.optim import Optimizer
from torch import nn
import numpy as np
import torch
import os


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
    # Set checkpoint and log dir.
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    log_dir = os.path.join(save_dir, 'logs')
    # Set initial values.
    epoch = 0
    train_loss = None
    train_loss_list = []
    val_loss_list = []
    val_acc_list = []

    # Resume training.
    if resume:
        checkpoint_data = _load_checkpoint(model, optimizer, checkpoint_dir, log_dir)
        model = checkpoint_data['model']
        optimizer = checkpoint_data['optimizer']
        train_loss = checkpoint_data['train_loss']
        train_loss_list = checkpoint_data['train_loss_list']
        val_loss_list = checkpoint_data['val_loss_list']
        val_acc_list = checkpoint_data['val_acc_list']

    # Train for the given number of epochs.
    for epoch in range(epoch, epochs):
        # Iterate over the training samples.
        model.train()
        for i, (x, label) in enumerate(train_loader):
            # Sanity check for types.
            assert isinstance(x, torch.Tensor)
            assert isinstance(label, torch.Tensor)
            x = x.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            # Forward pass.
            logits = model(x)
            train_loss = criterion(logits, label)
            # Backward pass.
            train_loss.backward()
            optimizer.step()
            # Print loss on same line during training.
            print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}, Loss: {train_loss.item():.4f}',
                  end='\r')

        # Iterate over the validation samples.
        model.eval()
        total_acc, total_count = 0, 0
        with torch.no_grad():
            for i, (x, label) in enumerate(val_loader):
                # Sanity check for types.
                assert isinstance(x, torch.Tensor)
                assert isinstance(label, torch.Tensor)
                # Initialize batch.
                x = x.to(device)
                label = label.to(device)
                # Forward pass.
                logits = model(x)
                # Compute loss and accuracy.
                val_loss = criterion(logits, label)
                total_acc += (logits.argmax(1) == label).sum().item()
                total_count += label.size(0)
                val_acc = total_acc / total_count
                # Store validation loss and accuracy.
                train_loss_list.append(train_loss.item())
                val_loss_list.append(val_loss.item())
                val_acc_list.append(val_acc)
                # Print stats on a new line.
                print(f'Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.4f}')

        # Save checkpoint.
        _save_checkpoint(epoch, model, optimizer, train_loss, val_loss, val_acc, checkpoint_dir, log_dir)

    # Save best model.
    _save_best_model(model, val_acc_list, checkpoint_dir, save_dir)
    # Save training graphs.
    _save_training_plots(train_loss_list, val_loss_list, val_acc_list, log_dir)


def _load_checkpoint(model: nn.Module, optimizer: Optimizer, checkpoint_dir: str, log_dir: str) -> Dict[str, Any]:
    """
    Loads a checkpoint of the model, optimizer, and statistics.

    Args:
        model: Classifier model with uninitialized weights.
        optimizer: Optimizer to with uninitialized state.
        checkpoint_dir: Path where model weight checkpoints are saved.
        log_dir: Path where stats logs are saved.

    Returns:
        A dictionary of everything that needs to be restored, as follows:
        {
            'epoch': epoch,
            'model': model with weights restored,
            'optimizer': optimizer with state restored,
            'train_loss': train_loss,
            'train_loss_list': list of prior training losses,
            'val_loss_list': list of prior validation losses,
            'val_acc_list': list of prior validation accuracies,
        }
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
    train_loss = checkpoint['loss']
    # Restore the stats.
    train_loss_path = os.path.join(log_dir, 'train_loss.txt')
    val_loss_path = os.path.join(log_dir, 'val_loss.txt')
    val_acc_path = os.path.join(log_dir, 'val_acc.txt')
    with open(train_loss_path, 'r') as f:
        train_loss_list = f.read().split('\n')[:-1]
        f.write(f'{train_loss.item()}\n')
    with open(val_loss_path, 'a+') as f:
        val_loss_list = f.read().split('\n')[:-1]
    with open(val_acc_path, 'a+') as f:
        val_acc_list = f.read().split('\n')[:-1]
    return {
        'epoch': epoch,
        'model': model,
        'optimizer': optimizer,
        'train_loss': train_loss,
        'train_loss_list': train_loss_list,
        'val_loss_list': val_loss_list,
        'val_acc_list': val_acc_list,
    }


def _save_checkpoint(epoch: int, model: nn.Module, optimizer: Optimizer,
                     train_loss: Callable, val_loss: Callable, val_acc: float,
                     checkpoint_dir: str, log_dir: str) -> None:
    """
    Saves a checkpoint of the model, optimizer, and statistics.

    Args:
        epoch: The current epoch number.
        model: Classifier model to be trained.
        optimizer: Optimizer to be used for training.
        train_loss: Training loss object for the current epoch.
        val_loss: Validation loss object for the current epoch.
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
        'loss': train_loss},
        checkpoint_path)
    # Save the stats.
    train_loss_path = os.path.join(log_dir, 'train_loss.txt')
    val_loss_path = os.path.join(log_dir, 'val_loss.txt')
    val_acc_path = os.path.join(log_dir, 'val_acc.txt')
    with open(train_loss_path, 'a+') as f:
        f.write(f'{train_loss.item()}\n')
    with open(val_loss_path, 'a+') as f:
        f.write(f'{val_loss.item()}\n')
    with open(val_acc_path, 'a+') as f:
        f.write(f'{val_acc}\n')


def _save_best_model(model: nn.Module, val_acc_list: List[float], checkpoint_dir: str, save_dir: str) -> None:
    """
    Loads the model with the highest validation accuracy from the checkpoint_dir and saves it to the save_dir.

    Args:
        model: Classifier model to be trained.
        val_acc_list: The list of validation accuracies for each epoch.
        checkpoint_dir: Path to save model weight checkpoints.
        save_dir: Path to the directory to save best model to.
    """
    # Get the epoch where the accuracy was the best.
    epoch = np.argmax(val_acc_list)
    # Load the checkpoint at that epoch.
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint-{epoch}.pt')
    checkpoint = torch.load(checkpoint_path)
    model = model.load_state_dict(checkpoint['model_state_dict'])
    # Save the model in its entirety to the save dir.
    save_path = os.path.join(save_dir, f'model-{epoch}.pt')
    model.save(save_path)


def _save_training_plots(train_loss_list: List[float], val_loss_list: List[float], val_acc_list: List[float],
                         log_dir: str) -> None:
    """
    Plots the training and validation loss, along with the validation accuracy, and saves the plots to log_dir.

    Args:
        train_loss_list: List of model training loss for each epoch.
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
    # Plot accuracy
    accuracy_path = os.path.join(log_dir, 'accuracy.png')
    plt.plot(epochs, val_acc_list, label='Validation accuracy', color='orange')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Accuracy')
    plt.savefig(accuracy_path)
