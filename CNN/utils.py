import numpy as np
from datetime import datetime 
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# from torch.utils.tensorboard import SummaryWriter
import torchvision
import matplotlib.pyplot as plt
import time
from functools import partial


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

def get_folded_accuracy(model, data_loader, device):
    correct_pred = 0 
    n = 0
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:
            X = X.to(device)
            y_true = y_true.to(device)
            y_output = model(X)
            y_prob = F.softmax(y_output, dim=1)
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

def validate(valid_loader, model, criterion, device, percentile, tensorboard_writer=None):
    '''
    Function for the validation step of the training loop
    '''
    activations = {}
    def save_activation(name, mod, inp, out):
        if name not in activations.keys():
            activations[name] = out
        else:
            activations[name] = torch.cat((activations[name],out))

    names = []
    handles = []
    max_weights = []
    max_weights_percentile = []
    min_weights = []
    min_weights_percentile = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(partial(save_activation, name)))
            names.append(name)
#             max_weights.append(module.weight.max().cpu().numpy().round(2))
#             min_weights.append(module.weight.min().cpu().numpy().round(2))
            max_weights_percentile.append(np.percentile(module.weight.cpu().numpy(), percentile).round(2))
            min_weights_percentile.append(np.percentile(module.weight.cpu().numpy(), 100-percentile).round(2))

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
    [handle.remove() for handle in handles] # remove forward hooks
    
    if tensorboard_writer == None:
#         str_output = ''.join(["{}: [{},{}]; ".format(names[i], str(min_weights[i]), str(max_weights[i])) for i in range(len(names))])
#         print("\t\tweights: " + str_output)
        str_output = ''.join(["{}: [{},{}]; ".format(names[i], str(min_weights_percentile[i]), str(max_weights_percentile[i])) for i in range(len(names))])
        print("\t" + str(percentile) + "% weights: " + str_output)
#         str_output = ''.join(["{}: {}, ".format(name, torch.quantile(torch.maximum(activation, torch.tensor(0)), percentile/100)) for name, activation in activations.items()])
#         print("\t" + str(percentile) + "% pytorch activations: " + str_output)
        str_output = ''.join(["{}: {}, ".format(name, round(np.percentile(np.maximum(activation.cpu(),0), percentile),3)) for name, activation in activations.items()])
        print("\t" + str(percentile) + "% activations: " + str_output)
    else:
#         [tensorboard_writer.add_scalar(name, weights) for name, weights in zip(names, max_weights_percentile)]
        [tensorboard_writer.add_histogram(name, activation) for name, activation in activations.items()]
        
    return model, epoch_loss

def get_weights_biases(model):
    weights = []
    biases = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            weights.append(module.weight.detach().numpy())
            biases.append(module.bias.detach().numpy())
    return weights, biases

def clip_parameters(model, minimum=-1, maximum=1):
    parameters = list(model.parameters())
    for parameter in parameters:
        parameter = nn.Parameter(torch.clamp(parameter, minimum, maximum))
    return model
