import torch.nn.functional as F
import torch


def focal_loss(prediction, label, gamma=2.0):
    # prediction - values (not probabilities), shape: (batch, 2)
    # label - values (indices as long), shape: (batch)
    focal_factor = torch.pow((1 - F.softmax(prediction, dim=1)), gamma)
    log_prob = -1.0* focal_factor * F.log_softmax(prediction, dim=1)
    sum_loss = log_prob.gather(1, label.unsqueeze(1))
    return sum_loss.mean()
