import random
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset

def dataset_division(train_dataset, test_dataset, flagTest=0, batch_size=256): 
    class_to_indices = defaultdict(list) # dizionario di liste 
    class_to_indicesTest = defaultdict(list) 

    for idx, (_, label) in enumerate(train_dataset): # divide il dataset in indici in base al label 
        class_to_indices[label].append(idx)     # label come key del dizionario 
        
         # random shuffle of indices 
    for indices in class_to_indices.values(): # restituisce liste di indici per ogni classe 
        random.shuffle(indices) # shuffle di lista 
                
                
    sharding_datasets = [] # list of dataset 
    cardinality_datasets = [] 
    total_indices = [[] for _ in range(100)]
        
    for class_id,indices in class_to_indices.items(): # class_id identifica dizionario, indices è lista 
        n = len(indices)
        chunk_size = n // 100
        remainder = n % 100 
        start = 0
        for i in range(100): 
            extra = 1 if i < remainder else 0
            end = start + chunk_size + extra
            total_indices[i].extend(indices[start:end]) # aggiungiamo indici relativi ad una certa classe - con append diventava lista di indici 
            start = end
    
    for i in range(100): 
        subset = torch.utils.data.Subset(train_dataset, total_indices[i])
        sharding_datasets.append(subset)

    sharding_dataloaders = []
    for dataset in sharding_datasets:
        cardinality_datasets.append(len(dataset)) # record of cardinality of datasets 
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        sharding_dataloaders.append(loader)

    ## test part 
    
    if flagTest == 1: 
        for idx, (_, label) in enumerate(train_dataset): # divide il dataset in indici in base al label su test
            class_to_indicesTest[label].append(idx)    
            
        for indices in class_to_indices.values(): 
            random.shuffle(indices) 
            
        sharding_datasetsTest = []
        cardinality_datasetsTest = [] 
        total_indices = [[] for _ in range(100)]
        
        for class_id,indices in class_to_indicesTest.items(): # class_id identifica dizionario, indices è lista 
            n = len(indices)
            chunk_size = n // 100
            remainder = n % 100 
            start = 0
            for i in range(100): 
                extra = 1 if i < remainder else 0
                end = start + chunk_size + extra
                total_indices[i].extend(indices[start:end]) # aggiungiamo indici relativi ad una certa classe - con append diventava lista di indici 
                start = end
    
        for i in range(100): 
            subset = torch.utils.data.Subset(test_dataset, total_indices[i])
            sharding_datasetsTest.append(subset)

        sharding_dataloadersTest = []
        for dataset in sharding_datasetsTest:
            cardinality_datasetsTest.append(len(dataset)) # record of cardinality of datasets 
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
            sharding_dataloadersTest.append(loader)
        
        return sharding_dataloaders, cardinality_datasets, sharding_dataloadersTest, cardinality_datasetsTest

    return sharding_dataloaders, cardinality_datasets
