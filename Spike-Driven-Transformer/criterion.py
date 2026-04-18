import torch
import torch.nn as nn


def TET_loss(outputs, labels, criterion, means, lamb):
    T = outputs.size(0)
    Loss_es = 0
    for t in range(T):
        Loss_es += criterion(outputs[t, ...], labels)
    Loss_es = Loss_es / T  # L_TET
    if lamb != 0:
        MMDLoss = torch.nn.MSELoss()
        y = torch.zeros_like(outputs).fill_(means)
        Loss_mmd = MMDLoss(outputs, y)  # L_mse
    else:
        Loss_mmd = 0
    return (1 - lamb) * Loss_es + lamb * Loss_mmd  # L_Total


def firing_rate_cv_loss(spike_outputs, lambda_cv=0.01):
    """
    Maximize the Coefficient of Variation (CV) of firing rates across neurons.
    
    CV = std / mean of firing rates
    Maximizing CV encourages sparse firing patterns where different neurons 
    have different firing rates, improving energy efficiency.
    
    Args:
        spike_outputs: (T, B, ...) - spike tensor (binary: 0 or 1)
                       where T is time steps, B is batch size
        lambda_cv: float - strength of CV loss regularization
    
    Returns:
        cv_loss: scalar tensor
    
    Reference:
        Higher CV indicates more heterogeneous firing patterns,
        which is desirable for event-driven neuromorphic computing.
    """
    T = spike_outputs.shape[0]
    
    # Aggregate spikes across time steps: (T, B, ...) -> (B, ...)
    # firing_rate represents the fraction of time steps where neuron fired
    firing_rate = spike_outputs.sum(dim=0) / T  # Range: [0, 1]
    
    # Flatten to (B, num_neurons) for easier computation
    B = firing_rate.shape[0]
    firing_rate_flat = firing_rate.view(B, -1)  # (B, num_neurons)
    
    # Compute CV for each sample in batch
    mean_fr = firing_rate_flat.mean(dim=1, keepdim=True)  # (B, 1)
    std_fr = firing_rate_flat.std(dim=1, keepdim=True, unbiased=False) + 1e-8  # (B, 1)
    
    # CV = std / mean
    cv = std_fr / (mean_fr + 1e-8)  # (B, 1)
    
    # We want to MAXIMIZE CV, so we return the negative
    cv_loss = -cv.mean()
    
    return lambda_cv * cv_loss


def combined_loss(outputs, labels, criterion, means=1.0, lamb=0.0, 
                  use_cv_loss=False, lambda_cv=0.01, cv_weight=1.0):
    """
    Combined loss function: Classification Loss + CV Regularization Loss
    
    This loss encourages the model to:
    1. Classify correctly (standard cross-entropy)
    2. Generate sparse, heterogeneous firing patterns (CV maximization)
    
    Args:
        outputs: (T, B, C) - model outputs across time steps
        labels: (B,) - target class labels
        criterion: loss function (e.g., CrossEntropyLoss)
        means: float - mean value for MMD loss (TET parameter)
        lamb: float - weight for MMD loss in TET (0-1)
        use_cv_loss: bool - whether to enable CV regularization
        lambda_cv: float - strength of CV loss
        cv_weight: float - relative weight of CV term vs classification loss
    
    Returns:
        total_loss: scalar tensor
    """
    T = outputs.size(0)
    
    # ========== Original TET Loss ==========
    Loss_es = 0
    for t in range(T):
        Loss_es += criterion(outputs[t, ...], labels)
    Loss_es = Loss_es / T  # Temporal aggregation
    
    # MMD loss (optional)
    if lamb != 0:
        MMDLoss = nn.MSELoss()
        y = torch.zeros_like(outputs).fill_(means)
        Loss_mmd = MMDLoss(outputs, y)
    else:
        Loss_mmd = 0
    
    base_loss = (1 - lamb) * Loss_es + lamb * Loss_mmd
    
    # ========== CV Regularization Loss ==========
    if use_cv_loss:
        # Convert continuous outputs to binary spikes
        # Threshold at 0.5 to convert logits/probabilities to spikes
        spike_outputs = (outputs > 0.5).float()
        
        cv_regularization = firing_rate_cv_loss(spike_outputs, lambda_cv)
        total_loss = base_loss + cv_weight * cv_regularization
    else:
        total_loss = base_loss
    
    return total_loss
