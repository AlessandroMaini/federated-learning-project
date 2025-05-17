import random
from collections import defaultdict
import math
from collections import Counter

import torch
from torch.utils.data import DataLoader, Subset

def data_division(train_dataset, test_dataset, type_sharding, flagTest = 0, batch_size=256): 
    # type_sharding = 1 : one class per client
    # type_sharding = 5 : five classes per client
    # type_sharding = 10 : ten classes per client 
    # type_sharding = 50 : fifty classes per client
    # type_sharding = 100: hundred classes per client - return class balance division 

    sharding_datasets = []
    cardinality_datasets = [] 
    sharding_dataloaders = []
    total_indices = [[] for _ in range(100)]
    counter_classes = [0 for _ in range(100)] # tiene conto del numero di classi per dataset 

    if type_sharding == 1: 
        class_to_indices = defaultdict(list) # dizionario di liste 

        for idx, (_, label) in enumerate(train_dataset): # divide il dataset in indici in base al label 
            class_to_indices[label].append(idx)
        
        for class_id, indices in class_to_indices.item(): 
            subset = torch.utils.data.Subset(train_dataset,indices) 
            sharding_datasets.append(subset)
            
        if flagTest == 1: 
            sharding_datasetsTest = []
            cardinality_datasetsTest = [] 
            sharding_dataloadersTest = []
            class_to_indicesTest = defaultdict(list) # dizionario di liste 

            for idx, (_, label) in enumerate(test_dataset): # divide il dataset in indici in base al label 
                class_to_indicesTest[label].append(idx)
            
            for class_id, indices in class_to_indicesTest.item(): 
                subset = torch.utils.data.Subset(test_dataset,indices) 
                sharding_datasetsTest.append(subset)
                
            
    if type_sharding == 5 or type_sharding == 10 or type_sharding == 50 or type_sharding == 100: 
        class_to_indices = defaultdict(list) # dizionario di liste 

        for idx, (_, label) in enumerate(train_dataset): # divide il dataset in indici in base al label 
            class_to_indices[label].append(idx)

        for indices in class_to_indices.values(): # restituisce liste di indici per ogni classe 
            random.shuffle(indices)
            
        if flagTest == 1: 
            sharding_datasetsTest = []
            cardinality_datasetsTest = [] 
            sharding_dataloadersTest = []
            total_indicesTest = [[] for _ in range(100)]
            class_to_indicesTest = defaultdict(list)
            
            for idx, (_, label) in enumerate(test_dataset): # divide il dataset in indici in base al label 
                class_to_indicesTest[label].append(idx)

            for indices in class_to_indicesTest.values(): # restituisce liste di indici per ogni classe 
                random.shuffle(indices)


        for class_id,indices in class_to_indices.items(): # class_id identifica dizionario, indices è lista 
            total = sum(counter_classes) 
            bound =  math.floor(total/100) 
            n = len(indices) # indices of samples associated to a specific class 
            chunk_size = n // type_sharding
            remainder = n % type_sharding
            start = 0
            
            
            # test's part 
            if flagTest == 1: 
                indicesTest = class_to_indicesTest[class_id] # indicesTest
                m = len(indicesTest)
                chunk_sizeTest = m // type_sharding
                remainderTest = m % type_sharding
                startTest = 0

            # valid for both 
            visited_clients = [] # visited clients to avoid repetition - reset at each iteration  
            valid_indices = [i for i, x in enumerate(counter_classes) if x < (bound + 1/2)]
      
            for j in range(type_sharding): 
                random_index = random.choice(valid_indices)
                valid_indices.remove(random_index)
                visited_clients.append(random_index)
                counter_classes[random_index] += 1 

                extra = 1 if j < remainder else 0 # assign one more to the first remainder-elements 
                end = start + chunk_size + extra
                total_indices[random_index].extend(indices[start:end]) # aggiungiamo indici relativi ad una certa classe - con append diventava lista di indici 
                start = end
                
                if flagTest == 1: 
                    extraTest = 1 if j < remainderTest else 0
                    endTest = startTest + chunk_sizeTest + extraTest
                    total_indicesTest[random_index].extend(indicesTest[startTest:endTest]) # aggiungiamo indici relativi ad una certa classe - con append diventava lista di indici 
                    startTest = endTest
                
        for i in range(100): 
            subset = torch.utils.data.Subset(train_dataset, total_indices[i])
            sharding_datasets.append(subset)
            
            if flagTest == 1: 
                subset = torch.utils.data.Subset(test_dataset, total_indicesTest[i])  #dataset test 
                sharding_datasetsTest.append(subset)
              
    
    # general dataloader 
    for dataset in sharding_datasets:
        cardinality_datasets.append(len(dataset)) # record of cardinality of datasets 
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        sharding_dataloaders.append(loader)
        
    if flagTest == 1: 
        for dataset in sharding_datasetsTest:
            cardinality_datasetsTest.append(len(dataset)) # record of cardinality of datasets 
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
            sharding_dataloadersTest.append(loader)
            
        return sharding_dataloaders, cardinality_datasets, sharding_dataloadersTest, cardinality_datasetsTest

    return sharding_dataloaders, cardinality_datasets




def print_classes_and_counts(dataloader, name="Dataloader"):
    class_count = Counter()
    
    for batch in dataloader:
        # Assuming the batch is structured as (images, labels) or (images, labels, ...)
        labels = batch[1]
        class_count.update(labels.tolist())

    print(f"\n📊 {name} — Classes present and their sample counts:")
    for cls in sorted(class_count):
        print(f"  Class {cls}: {class_count[cls]} samples")
