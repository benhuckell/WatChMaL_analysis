import sys
import numpy as np

class TrueHitsDataset:
    def __init__(self):
        filePath = '/fast_scratch/WatChMaL/data/trueHitsList.npy'
        self.trueHitsList = np.load(filePath, allow_pickle=True)

    def getFullArray(self):
        return self.trueHitsList


if __name__ == "__main__":
    tr = TrueHitsDataset()
    arr = tr.getFullArray()
    print(len(arr))
        
