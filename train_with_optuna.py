#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Training with hyperparameter tuning

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from advanced_model import AdvancedAudioClassifier
import wandb
import optuna

# Directory where the preprocessed data is stored
data_dir = 'preprocessed_data'

def load_data_for_fold(fold, train=True):
    """
    Load spectrograms and labels for a specific fold.

    Args:
    - fold (int): The fold number to load.
    - train (bool): Whether to load the training or validation set. Default is True (training set).

    Returns:
    - spectrograms (Tensor): Loaded spectrograms.
    - labels (Tensor): Corresponding labels.
    """
    if train:
        spectrograms = torch.load(os.path.join(f'{data_dir}/fold{fold}', 'spectrograms_augmented.pt'))
    else:
        spectrograms = torch.load(os.path.join(f'{data_dir}/fold{fold}', 'spectrograms.pt'))
    
    labels = torch.tensor(torch.load(os.path.join(f'{data_dir}/fold{fold}', 'labels.pt')), dtype=torch.long)
    return spectrograms, labels

def evaluate_model(model, val_loader, criterion):
    """
    Evaluate the model on the validation set.

    Args:
    - model (nn.Module): The model to evaluate.
    - val_loader (DataLoader): DataLoader for the validation set.
    - criterion (Loss): The loss function.

    Returns:
    - val_loss (float): Average loss on the validation set.
    - val_accuracy (float): Accuracy on the validation set.
    """
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the appropriate device

            outputs = model(inputs)  # Forward pass
            
            loss = criterion(outputs, labels)  # Compute loss

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)  # Get the index of the max log-probability
            total += labels.size(0)  # Update the total number of samples
            correct += predicted.eq(labels).sum().item()  # Update the number of correct predictions

    val_loss = running_loss / total  # Compute average loss over all samples
    val_accuracy = correct / total  # Compute accuracy

    return val_loss, val_accuracy

def train_model(trial, train_loader, val_loader, num_epochs, model, criterion, optimizer, scheduler):
    """
    Train the model and evaluate it on the validation set at each epoch.

    Args:
    - trial (optuna.Trial): Optuna trial object for hyperparameter optimization.
    - train_loader (DataLoader): DataLoader for the training set.
    - val_loader (DataLoader): DataLoader for the validation set.
    - num_epochs (int): Number of epochs to train.
    - model (nn.Module): The model to train.
    - criterion (Loss): The loss function.
    - optimizer (Optimizer): The optimizer.
    - scheduler (LRScheduler): Learning rate scheduler.

    Returns:
    - val_accuracy (float): Validation accuracy after the final epoch.
    """
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the appropriate device
            
            optimizer.zero_grad()  # Zero the parameter gradients
            
            outputs = model(inputs)  # Forward pass
            
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_accuracy = correct / total
        
        # Log training progress to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
        })

        # Evaluate the model
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)
        
        # Log validation progress to wandb
        wandb.log({
            'epoch': epoch + 1,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        })

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}')

        scheduler.step(val_loss)  # Adjust learning rate based on validation loss

    return val_accuracy

def objective(trial):
    """
    Objective function for Optuna hyperparameter tuning.

    Args:
    - trial (optuna.Trial): Optuna trial object.

    Returns:
    - mean_val_accuracy (float): Mean validation accuracy across folds.
    """
    all_val_accuracy = []
    # Sample hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.2, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 512])  

    num_epochs = 15
    
    for fold in range(1, 2):
        X_val, y_val = load_data_for_fold(fold, train=False)
        X_train, y_train = [], []
        for train_fold in range(1, 11):
            if train_fold == fold:
                continue
            X_fold, y_fold = load_data_for_fold(train_fold, train=True)
            X_train.extend(X_fold)
            y_train.extend(y_fold)

        X_train = torch.stack(X_train)
        y_train = torch.tensor(y_train, dtype=torch.long)
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(torch.stack(X_val), y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        model = AdvancedAudioClassifier(dropout_rate).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        
        val_accuracy = train_model(trial, train_loader, val_loader, num_epochs, model, criterion, optimizer, scheduler)
        all_val_accuracy.append(val_accuracy)
    
    return np.mean(all_val_accuracy)

if __name__ == "__main__":
    # Initialize wandb for logging
    wandb.init(project="audio_classification", entity="username")

    # Determine the device to use (MPS, CUDA, or CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Create an Optuna study for hyperparameter optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    print("Best hyperparameters: ", study.best_params)


# In[ ]:





# In[ ]:




