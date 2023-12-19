import torch

def average_displacement(predicted, true):
    """ Average displacement error. """
    true, masks = true
    sequence_lengths = masks.sum(1)
    batch_size = len(sequence_lengths)
    squared_distance = (true - predicted)**2
    l2_distance = masks * torch.sqrt(squared_distance.sum(2))
    average_l2_distance = (1. / batch_size) * ((1. / sequence_lengths) * l2_distance.sum(1)).sum()
    return average_l2_distance.item()

def final_displacement(predicted, true):
    """ Final displacement error """
    true, masks = true
    sequence_lengths = masks.sum(1).type(torch.LongTensor) - 1
    batch_size = len(sequence_lengths)
    squared_distance = (true - predicted)**2
    l2_distances = masks * torch.sqrt(squared_distance.sum(2))
    displacement_sum = l2_distances[:, sequence_lengths].sum()
    average_final_l2_displacement = (1. / batch_size) * displacement_sum
    return average_final_l2_displacement.item()