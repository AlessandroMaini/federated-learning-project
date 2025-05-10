import torch.optim as optim
from eval import evaluate
from train import train

def run_grid_search(train_loader, val_loader, model_fn, criterion, configs, device):
    best_acc = 0
    best_cfg = None
    results = {'lr': [], 'momentum': [], 'val_loss': [], 'val_acc': []}

    for cfg in configs:
        model = model_fn(device)
        optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

        best_loss = float('inf')
        patience_counter = 0

        print(f"Training with lr={cfg['lr']}, momentum={cfg['momentum']}...")
        for epoch in range(5):  # Small number of epochs for quick grid search
            train(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            
            print(f"  Epoch {epoch+1}: Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")
            
            if val_loss < best_loss - 1e-2:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 3:
                print("  Early stopping")
                break

        results['lr'].append(cfg['lr'])
        results['momentum'].append(cfg['momentum'])
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_cfg = cfg


    print(f"Best val acc: {best_acc:.4f} with lr={best_cfg['lr']}, momentum={best_cfg['momentum']}")
    return best_cfg, results