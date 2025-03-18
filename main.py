import torch
import numpy as np
from torch import nn, optim
from sklearn.metrics import roc_auc_score, f1_score
from dataloader import get_dataloaders
from model import TabTransformer, BaselineDNN

def train(model, train_loader, val_loader, criterion, optimizer, epochs=10, device='cpu'):
    model = model.to(device)
    best_auc = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            categorical = batch['categorical'].to(device)
            numerical = batch['numerical'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            
            if isinstance(model, TabTransformer):
                outputs = model(categorical, numerical)
            else:
                inputs = torch.cat([categorical.float(), numerical], dim=1)
                outputs = model(inputs)
                
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        val_metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val AUC: {val_metrics['auc']:.4f} | Val F1: {val_metrics['f1']:.4f} | Val Acc: {val_metrics['acc']:.4f}")
        
        # Save best model
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            torch.save(model.state_dict(), 'best_model.pth')
            
    return model

def evaluate(model, loader, device='cpu'):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            categorical = batch['categorical'].to(device)
            numerical = batch['numerical'].to(device)
            targets = batch['target'].to(device)
            
            if isinstance(model, TabTransformer):
                outputs = model(categorical, numerical)
            else:
                inputs = torch.cat([categorical.float(), numerical], dim=1)
                outputs = model(inputs)
            probabilities = torch.sigmoid(outputs)
            all_preds.append(probabilities.cpu())
            all_targets.append(targets.cpu())
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    preds_class = (all_preds > 0.5).float()
    
    return {
        'acc': (preds_class == all_targets).float().mean().item(),
        'f1': f1_score(all_targets, preds_class),
        'auc': roc_auc_score(all_targets, all_preds)
    }

if __name__ == "__main__":
    # Config
    DATA_PATH = "diabetes_prediction_dataset.csv"
    BATCH_SIZE = 64
    EPOCHS = 20
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Get data
    train_loader, val_loader, test_loader,  encoders = get_dataloaders(DATA_PATH, BATCH_SIZE)
    
    # Model setup
    categorical_dims = [len(enc.classes_[0]) for enc in encoders.values()]
    numerical_dim = 7  # From numerical columns
    
    # Choose model
    model = TabTransformer(
        categorical_dims=categorical_dims,
        numerical_dim=numerical_dim,
        hidden_dim=64,
        n_heads=4,
        n_layers=3
    )
    
    # For baseline DNN:
    # input_dim = sum(categorical_dims) + numerical_dim
    # model = BaselineDNN(input_dim)
    
    # Handle class imbalance
    pos_weight = torch.tensor([5.0]).to(DEVICE)  # Adjust based on your dataset
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Train
    trained_model = train(
        model, train_loader, val_loader, 
        criterion, optimizer, EPOCHS, DEVICE
    )
    
    # Final evaluation
    test_metrics = evaluate(trained_model, val_loader, DEVICE)
    print("\nFinal Test Performance:")
    print(f"AUC: {test_metrics['auc']:.4f} | F1: {test_metrics['f1']:.4f} | Accuracy: {test_metrics['acc']:.4f}")