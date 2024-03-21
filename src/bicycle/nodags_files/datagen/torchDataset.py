from torch.utils.data import Dataset 
import numpy as np

class experimentDataset(Dataset):
    def __init__(self, dataset, intervention_set):
        self.dataset = dataset
        self.intervention_set = intervention_set 

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class experimentDatasetStrat(Dataset):
    def __init__(self, datasets, intervention_sets):
        self.intervention_sets = intervention_sets 
        self.datasets = datasets

        self.make_final_data()

    def make_final_data(self):
        self.final_data = np.vstack(self.datasets)
        masks = list()
        for dataset, intervention_set in zip(self.datasets, self.intervention_sets):
            mask = np.ones(dataset.shape)
            if intervention_set[0] != None:
                mask[:, intervention_set] = 0
            
            masks.append(mask)
        
        self.masks = np.vstack(masks)

    def __len__(self):
        return len(self.final_data)

    def __getitem__(self, idx):
        return self.final_data[idx], self.masks[idx]