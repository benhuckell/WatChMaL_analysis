import numpy as np

n_events = 2868354

val_idxs = np.arange(0, int(n_events*0.4))
test_idxs = np.arange(int(n_events*0.4), int(n_events*0.5))
train_idxs = np.arange(int(n_events*0.5), n_events)
np.savez("/fast_scratch/WatChMaL/data/IWCD_fulltank_300_segmentation_idx.npz", val_idxs=val_idxs, test_idxs=test_idxs, train_idxs=train_idxs)