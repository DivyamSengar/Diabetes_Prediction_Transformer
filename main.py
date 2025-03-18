import torch
import numpy as np
import argparse
from torch import nn, optim
from sklearn.metrics import roc_auc_score, f1_score
from dataloader import get_dataloaders
from model import TabTransformer, BaselineDNN
from utils import plot_training_history

def train(model, train_loader, val_loader, criterion, optimizer, epochs=10, device='cpu'):
    model = model.to(device)
    best_auc = 0
    train_losses = []
    val_metrics_history = []
    
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
        
        avg_train_loss = train_loss/len(train_loader)
        train_losses.append(avg_train_loss)
            
        # Validation
        val_metrics = evaluate(model, val_loader, device)
        val_metrics_history.append(val_metrics)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val AUC: {val_metrics['auc']:.4f} | Val F1: {val_metrics['f1']:.4f} | Val Acc: {val_metrics['acc']:.4f}")
        
        # Save best model
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_metrics': val_metrics_history,
            }, f'best_{type(model).__name__}.pth')
    
    # Plot training history
    plot_training_history(train_losses, val_metrics_history, f'{type(model).__name__}_history.png')
            
    return model, train_losses, val_metrics_history

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
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)
                
            all_preds.append(probs.cpu())
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
    parser = argparse.ArgumentParser(description='Train diabetes prediction models')
    parser.add_argument('--model', type=str, choices=['transformer', 'dnn'], default='transformer',
                      help='Model to train (transformer or dnn)')
    parser.add_argument('--epochs', type=int, default=20,
                      help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size for training')
    parser.add_argument('--data_path', type=str, default='diabetes.csv',
                      help='Path to the diabetes dataset')
    parser.add_argument('--augment', action='store_true',
                      help='Use SMOTE data augmentation')
    args = parser.parse_args()
    
    # Config
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    
    # Get data
    train_loader, val_loader, test_loader, encoders = get_dataloaders(
        args.data_path, args.batch_size, args.augment
    )
    
    # Model setup
    categorical_dims = [len(encoders['smoking_history'].classes_)]
    numerical_dim = 7  # From numerical columns
    
    # Choose model
    if args.model == 'transformer':
        model = TabTransformer(
            categorical_dims=categorical_dims,
            numerical_dim=numerical_dim,
            hidden_dim=64,
            n_heads=4,
            n_layers=3
        )
        print("Training TabTransformer model")
    else:
        # For DNN, we just need total input dimension (1 for each feature)
        input_dim = len(categorical_dims) + numerical_dim  # 1 categorical + 7 numerical = 8 features
        model = BaselineDNN(
            input_dim=input_dim,
            hidden_dims=[32, 16],  # Smaller network than transformer
            num_classes=1
        )
        print("Training Baseline DNN model")
    
    # Handle class imbalance
    pos_weight = torch.tensor([5.0]).to(DEVICE)  # Adjust based on your dataset
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Train
    trained_model, train_losses, val_metrics = train(
        model, train_loader, val_loader, 
        criterion, optimizer, args.epochs, DEVICE
    )
    
    # Final evaluation
    test_metrics = evaluate(trained_model, test_loader, DEVICE)
    print("\nFinal Test Performance:")
    print(f"AUC: {test_metrics['auc']:.4f} | F1: {test_metrics['f1']:.4f} | Accuracy: {test_metrics['acc']:.4f}")