import numpy as np
import torch
import copy

'''
n_events = 2868354

val_idxs = np.arange(0, int(n_events*0.4))
test_idxs = np.arange(int(n_events*0.4), int(n_events*0.5))
train_idxs = np.arange(int(n_events*0.5), n_events)
np.savez("/fast_scratch/WatChMaL/data/IWCD_fulltank_300_segmentation_idx.npz", val_idxs=val_idxs, test_idxs=test_idxs, train_idxs=train_idxs)
'''

labels = np.array([[[[0, 2], [1, 2]], [[2, 0], [3, 2]]],[[[0, 2], [1, 2]], [[2, 0], [3, 3]]],[[[0, 2], [1, 2]], [[2, 0], [3, 3]]],[[[0, 2], [1, 2]], [[2, 0], [3, 3]]]])
predictions = np.array([[[[0, 3], [1, 3]], [[2, 3], [2, 3]]],[[[0, 0], [1, 0]], [[2, 3], [2, 3]]],[[[0, 0], [1, 0]], [[2, 3], [2, 2]]],[[[0, 0], [1, 0]], [[2, 3], [2, 3]]]])

print(labels.shape)

labelsTensor = torch.tensor(labels)
predTensor = torch.tensor(predictions)

#swap labels
swapLabelsTensor = copy.deepcopy(labelsTensor)
swapLabelsTensor[swapLabelsTensor == 2] = -3
swapLabelsTensor[swapLabelsTensor == 3] = 2
swapLabelsTensor[swapLabelsTensor == -3] = 3


regLoss = torch.tensor([0.5, 0.6, 1.2, 3.4]) #regular
swapLoss = torch.tensor([0.4, 0.7, 1.1, -0.2]) #swap

loss = torch.min(regLoss, swapLoss)
lossPos = torch.gt(regLoss, swapLoss)

print(loss)
print(torch.mean(loss).item())
print(lossPos)

accArray = []

for event in range(labelsTensor.shape[0]):
    if(lossPos[event]): #regular loss > swap loss -> use swap
        correct = ((swapLabelsTensor[event]  == predTensor[event] ) & (swapLabelsTensor[event]  != 0)).sum().item()
    else: #regular loss <= swap loss -> use regular
        correct = ((labelsTensor[event] == predTensor[event] ) & (labelsTensor[event]  != 0)).sum().item()

    total = (labelsTensor[event]  != 0).sum().item()
    accArray.append(correct/total)
    
print(accArray)


addTensor = torch.zeros(4,2,1,2)
print(addTensor.shape)

out = torch.cat((labelsTensor,addTensor), dim = 2)
print(out.shape)

out2 = out[:,:,0:2,:]
print(out2.shape)




'''
correct = ((labelsTensor == predTensor) & (labelsTensor != 0)).sum().item()
total = (labelsTensor != 0).sum().item()

output = correct/total

print(output)
'''

#print(torch.gt(labelsTensor,predTensor))


#min



