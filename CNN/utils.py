import numpy as np
from datetime import datetime 
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision
import matplotlib.pyplot as plt
import ipdb
import time

def get_accuracy(model, data_loader, device):
    correct_pred = 0 
    n = 0
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:
            X = X.to(device)
            y_true = y_true.to(device)
            y_prob = model(X)[1]
            _, predicted_labels = torch.max(y_prob, 1)
            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()
    return correct_pred.float() / n

def plot_losses(train_losses, valid_losses):
    # temporarily change the style of the plots to seaborn 
    plt.style.use('seaborn')
    train_losses = np.array(train_losses) 
    valid_losses = np.array(valid_losses)
    fig, ax = plt.subplots(figsize = (8, 4.5))
    ax.plot(train_losses, color='blue', label='Training loss') 
    ax.plot(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs", 
            xlabel='Epoch',
            ylabel='Loss')
    ax.legend()
    fig.show()
    # change the plot style to default
    plt.style.use('default')

def validate(valid_loader, model, criterion, device):
    '''
    Function for the validation step of the training loop
    '''
    model.eval()
    running_loss = 0
    for X, y_true in valid_loader:
        X = X.to(device)
        y_true = y_true.to(device)
        # Forward pass and record loss
        y_hat = model(X)[0]
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)
    epoch_loss = running_loss / len(valid_loader.dataset)
    return model, epoch_loss


def get_weights_biases(model):
    parameters = list(model.parameters())
    weights = [weight for weight in parameters[::2]]
    biases = [bias for bias in parameters[1::2]]
    return weights, biases

def biggest_weight(model):
    weights, biases = get_weights_biases(model)
    max_weight = torch.cat([weight.flatten() for weight in weights]).abs().max()
    max_bias = torch.cat([bias.flatten() for bias in biases]).abs().max()
    print("biggest weight={0:.3f} and bias={1:.3f}".format(max_weight, max_bias))