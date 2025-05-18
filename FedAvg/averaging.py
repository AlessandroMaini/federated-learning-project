from train import train

def clientUpdate(client_dataloader, model, current_state, optimizer, device, epochs, criterion): # function to excute training on a single dataloader
    model.load_state_dict(current_state)
    for epoch in range(epochs): 
        train(model, client_dataloader, optimizer, criterion, device)

    return model.state_dict() # ritorna i parametri del modello aggiornati 

def get_trainable_keys(model): # ritorna keys di model_state trainabili (model_state è un dizionario)
    return {name for name, param in model.named_parameters() if param.requires_grad}
    

def averaging(clients_dataloaders, cardinality_dataloaders, model, optimizer, device, epochs, criterion): 
    total_sum = sum(cardinality_dataloaders) 
    weights = [x / total_sum for x in cardinality_dataloaders] # weight of sum 
    current_state = model.state_dict() 

    local_states = []
    
    for dataloader in clients_dataloaders: 
        client_state = clientUpdate(dataloader, model, current_state, optimizer, device, epochs, criterion)
        local_states.append(client_state)

    trainable_keys = get_trainable_keys(model)


    # Calcola la media pesata del parametro
    for key in trainable_keys:
        avg_param = sum(local_states[i][key] * weights[i] for i in range(10))
        current_state[key] = avg_param
  
    model.load_state_dict(current_state)

  
