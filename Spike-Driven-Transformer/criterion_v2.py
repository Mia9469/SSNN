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


def _spike_cv(spike_tensors):
    """
    Internal helper: compute mean CV of firing rates from a list of spike tensors.

    Each tensor must be (T, B, ...) with binary {0, 1} spike values, where T is
    the number of time steps and B is batch size.

    CV = std(firing_rate) / mean(firing_rate)  across neurons, per sample.
    Averaged over all samples and all provided tensors.

    Returns a scalar tensor, or None if the list is empty.
    """
    cv_values = []
    for spikes in spike_tensors:
        T = spikes.shape[0]
        firing_rate = spikes.float().sum(dim=0) / T  # (B, ...), values in [0, 1]
        B = firing_rate.shape[0]
        fr_flat = firing_rate.view(B, -1)  # (B, N)

        mean_fr = fr_flat.mean(dim=1, keepdim=True)  # (B, 1)
        std_fr = fr_flat.std(dim=1, keepdim=True, unbiased=False) + 1e-8
        # mean_fr is a firing rate in [0,1] so it is non-negative; no need for abs.
        cv = std_fr / (mean_fr + 1e-8)
        cv_values.append(cv.mean())

    if not cv_values:
        return None
    return torch.stack(cv_values).mean()


def firing_rate_cv_loss(spike_tensors, lambda_cv=0.01):
    """
    Maximize the CV of output-layer firing rates.

    ⚠️  GRADIENT WARNING — SNN dead-neuron trap:
        CV = std(fr) / mean(fr).  The gradient of -CV w.r.t. below-average neurons
        is positive (gradient descent pushes their fr toward 0).  Once a LIF neuron
        stops firing the surrogate gradient ≈ 0 and the neuron can never recover.
        This causes catastrophic collapse when training from scratch (mean_fr → 0,
        CE → log(C), accuracy → random).

        Safe use: fine-tuning from a converged checkpoint (neurons are already active).
        Unsafe use: from-scratch training (use weight_cv_loss only instead).

        To apply safely, pass spike tensors with `.detach()` so this term is a
        monitoring metric rather than an active gradient signal.

    Args:
        spike_tensors: list of (T, B, ...) binary spike tensors from output LIF layers
                       (populated from the model's hook dict, key "head_lif")
        lambda_cv: regularization strength

    Returns:
        scalar loss tensor (negative CV scaled by lambda), or 0.0 when list is empty
    """
    cv = _spike_cv(spike_tensors)
    if cv is None:
        return 0.0
    return -lambda_cv * cv


def weight_cv_loss(model, lambda_weight_cv=0.001, layer_types=(nn.Conv2d, nn.Linear)):
    """
    Maximize the CV of weight magnitudes across filters / output neurons.

    CV = std(|w|) / mean(|w|) per filter, averaged over all filters and layers.

    Using the absolute values is critical: weights have both signs, so mean(w) can
    be near zero even for large weights, making std/mean diverge.  std(|w|)/mean(|w|)
    is always well-defined (denominator >= 0) and measures genuine spread.

    Inspired by sparse coding: dictionary atom diversity improves reconstruction;
    hypothesis: weight diversity improves feature specialisation in SNNs.

    Args:
        model: nn.Module
        lambda_weight_cv: regularization strength
        layer_types: which layer classes to apply the regularization to

    Returns:
        scalar loss tensor
    """
    weight_cvs = []
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    for module in model.modules():
        if not isinstance(module, layer_types):
            continue
        if not hasattr(module, 'weight') or module.weight is None:
            continue
        w = module.weight
        if w.dim() < 2:
            continue

        # Flatten each output filter to a vector and work with absolute values.
        w_abs = w.view(w.shape[0], -1).abs()          # (out_channels, fan_in), >= 0
        mean_abs = w_abs.mean(dim=1, keepdim=True) + 1e-8
        std_abs = w_abs.std(dim=1, keepdim=True, unbiased=False) + 1e-8
        cv_w = std_abs / mean_abs                      # (out_channels, 1), well-defined
        weight_cvs.append(cv_w.mean())

    if not weight_cvs:
        return torch.tensor(0.0, device=device, dtype=dtype)

    avg_cv = torch.stack(weight_cvs).mean()
    return -lambda_weight_cv * avg_cv


def activation_cv_loss(spike_tensors, lambda_act_cv=0.005):
    """
    Maximize the CV of hidden-layer spike activations.

    Encourages neurons in intermediate layers to specialize: different neurons
    should fire at different rates, preventing redundant representations.

    ⚠️  Same dead-neuron trap as firing_rate_cv_loss — see its docstring.
        Pass detached spike tensors for safe use during from-scratch training.

    Args:
        spike_tensors: list of (T, B, ...) spike tensors from hidden LIF layers
                       (populated from the model's hook dict, all keys except "head_lif")
        lambda_act_cv: regularization strength

    Returns:
        scalar loss tensor, or 0.0 when list is empty
    """
    cv = _spike_cv(spike_tensors)
    if cv is None:
        return 0.0
    return -lambda_act_cv * cv


def combined_loss(outputs, labels, criterion, use_cv_loss=False,
                  lambda_cv=0.01, cv_weight=1.0, model=None,
                  use_weight_cv=False, lambda_weight_cv=0.001,
                  use_act_cv=False, lambda_act_cv=0.005,
                  hook=None, **kwargs):
    """
    Combined loss: TET + (optional) firing-rate CV + weight CV + hidden-activation CV.

    Spike tensors for the two CV-on-activations terms are sourced from the model's
    hook dict (populated by calling model.forward(x, hook={})). This ensures we
    regularize actual LIF spike outputs rather than class logits.

    Args:
        outputs:           (T, B, C) class logits from the model
        labels:            (B,) ground-truth class indices
        criterion:         base loss (CrossEntropyLoss)
        use_cv_loss:       enable output-layer firing-rate CV regularization
        lambda_cv:         output CV strength
        cv_weight:         additional scale factor for the output CV term
        model:             nn.Module — required when use_weight_cv=True
        use_weight_cv:     enable weight-heterogeneity regularization
        lambda_weight_cv:  weight CV strength
        use_act_cv:        enable hidden-layer activation CV regularization
        lambda_act_cv:     hidden activation CV strength
        hook:              dict returned by model.forward(x, hook={}); keys are
                           layer names, values are detached (T, B, ...) spike tensors.
                           Pass None (default) to skip spike-based CV terms.
        **kwargs:          forwarded to TET_loss as `means` and `lamb`
    """
    base_loss = TET_loss(outputs, labels, criterion,
                         means=kwargs.get('means', 0.0),
                         lamb=kwargs.get('lamb', 0.0))
    total_loss = base_loss

    # --- Output-layer firing-rate CV ---
    if use_cv_loss and hook is not None:
        head_spikes = [v.float() for k, v in hook.items() if k == "head_lif"]
        fr_loss = firing_rate_cv_loss(head_spikes, lambda_cv)
        if isinstance(fr_loss, torch.Tensor):
            total_loss = total_loss + cv_weight * fr_loss

    # --- Weight heterogeneity CV ---
    if use_weight_cv and model is not None:
        total_loss = total_loss + weight_cv_loss(model, lambda_weight_cv)

    # --- Hidden-layer activation CV ---
    if use_act_cv and hook is not None:
        hidden_spikes = [v.float() for k, v in hook.items() if k != "head_lif"]
        act_loss = activation_cv_loss(hidden_spikes, lambda_act_cv)
        if isinstance(act_loss, torch.Tensor):
            total_loss = total_loss + act_loss

    return total_loss
