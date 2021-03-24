import numpy as np
import torch

'''
n_events = 2868354

val_idxs = np.arange(0, int(n_events*0.4))
test_idxs = np.arange(int(n_events*0.4), int(n_events*0.5))
train_idxs = np.arange(int(n_events*0.5), n_events)
np.savez("/fast_scratch/WatChMaL/data/IWCD_fulltank_300_segmentation_idx.npz", val_idxs=val_idxs, test_idxs=test_idxs, train_idxs=train_idxs)
'''

labels = np.array([[[0, 2], [1, 2]], [[2, 0], [3, 3]]])
predictions = np.array([[[0, 0], [1, 0]], [[2, 3], [2, 3]]])

labelsTensor = torch.tensor(labels)
predTensor = torch.tensor(predictions)


correct = ((labelsTensor == predTensor) & (labelsTensor != 0) & (labelsTensor != 1)).sum().item()
total = ((labelsTensor != 0) & (labelsTensor != 1)).sum().item()

output = correct/total

print(output)