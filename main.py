import torch
import numpy as np
from torch import nn, optim
from sklearn.metrics import roc_auc_score, f1_score
from dataloader import get_dataloaders
from model import TabTransformer, BaselineDNN, EnsembleModel, EnhancedTabTransformer
import argparse

def train(model, train_loader, val_loader, criterion, optimizer, epochs=10, device='cpu'):
    model = model.to(device)
    best_auc = 0
    best_acc = 0
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
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
            elif isinstance(model, BaselineDNN):
                inputs = torch.cat([categorical.float(), numerical], dim=1)
                outputs = model(inputs)
            elif isinstance(model, EnsembleModel):
                outputs = model(categorical, numerical)
            else:
                raise ValueError("Unknown model type for training.")

            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
        scheduler.step()

        # Validation
        val_metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val AUC: {val_metrics['auc']:.4f} | Val F1: {val_metrics['f1']:.4f} | Val Acc: {val_metrics['acc']:.4f}")

        if val_metrics['acc'] > best_acc:
            best_acc = val_metrics['acc']
            torch.save(model.state_dict(), 'best_model.pth')

    return model

def evaluate(model, loader, device='cpu'):
    model.eval()
    all_preds =[]
    all_targets =[]

    with torch.no_grad():
        for batch in loader:
            categorical = batch['categorical'].to(device)
            numerical = batch['numerical'].to(device)
            targets = batch['target'].to(device)

            if isinstance(model, TabTransformer):
                outputs = model(categorical, numerical)
            elif isinstance(model, BaselineDNN):
                inputs = torch.cat([categorical.float(), numerical], dim=1)
                outputs = model(inputs)
            elif isinstance(model, EnsembleModel):
                outputs = model(categorical, numerical)
            else:
                raise ValueError("Unknown model type for evaluation.")

            probabilities = torch.sigmoid(outputs)
            all_preds.append(probabilities.cpu())
            all_targets.append(targets.cpu())
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # preds_class = (all_preds > 0.5).float()
    best_thresh = find_optimal_threshold(model, val_loader, DEVICE)
    preds_class = (all_preds > best_thresh).float()

    return {
        'acc': (preds_class == all_targets).float().mean().item(),
        'f1': f1_score(all_targets, preds_class),
        'auc': roc_auc_score(all_targets, all_preds)
    }
# Add to model.py or keep in main.py
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.BCEWithLogitsLoss()(inputs, targets)
        pt = torch.exp(-bce_loss)
        return self.alpha * (1-pt)**self.gamma * bce_loss
def find_optimal_threshold(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            categorical = batch['categorical'].to(device)
            numerical = batch['numerical'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model(categorical, numerical)
            all_preds.append(torch.sigmoid(outputs).cpu())
            all_targets.append(targets.cpu())
    
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    
    thresholds = np.linspace(0.3, 0.7, 50)
    f1s = [f1_score(targets, preds > t) for t in thresholds]
    return thresholds[np.argmax(f1s)]
# Add new functions
def calculate_val_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            categorical = batch['categorical'].to(device)
            numerical = batch['numerical'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model(categorical, numerical)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss/len(loader)

def plot_losses(train_losses, val_losses):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training/Validation Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_curves.png')
    plt.close()
if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Train and evaluate different models.")
    parser.add_argument("--model_type", type=str, choices=['baseline', 'tabular', 'ensemble'], default='ensemble', help="Choose the model type to run (baseline, tabular, ensemble)")
    args = parser.parse_args()

    # Config
    DATA_PATH = "diabetes_prediction_dataset.csv"
    BATCH_SIZE = 64
    EPOCHS = 20
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Get data
    train_loader, val_loader, test_loader, one_hot_columns = get_dataloaders(DATA_PATH, BATCH_SIZE)

    # Model setups
    numerical_dim = 7  # From numerical columns

    if args.model_type == 'tabular':
        categorical_dims = len(one_hot_columns)
        model = TabTransformer(
            categorical_dims=categorical_dims,
            numerical_dim=numerical_dim,
            hidden_dim=64,
            n_heads=8,
            n_layers=6
        )
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    elif args.model_type == 'baseline':
        input_dim = len(one_hot_columns) + numerical_dim
        model = BaselineDNN(input_dim)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    elif args.model_type == 'enhanced':
        categorical_dims = len(one_hot_columns)
        model = EnhancedTabTransformer(
        categorical_dims=len(one_hot_columns),
        numerical_dim=numerical_dim,
        hidden_dim=64,
        n_heads=8,  # Matches your previous experiment
        n_layers=6   # Matches your previous experiment
    )
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    elif args.model_type == 'ensemble':
        categorical_dims_tab = len(one_hot_columns)
        model_tab = TabTransformer(
            categorical_dims=categorical_dims_tab,
            numerical_dim=numerical_dim,
            hidden_dim=64,
            n_heads=8,
            n_layers=6
        )
        input_dim_dnn = len(one_hot_columns) + numerical_dim
        model_baseline = BaselineDNN(input_dim_dnn)
        model = EnsembleModel(model_tab, model_baseline)
        optimizer = optim.AdamW([
            {'params': model.model1.parameters()},
            {'params': model.model2.parameters()},
            {'params': model.meta.parameters()}
        ], lr=1e-3, weight_decay=1e-4)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    # # Replace pos_weight calculation in main.py
    # class_counts = train_loader.dataset.df['diabetes'].value_counts().to_list()
    # pos_weight = torch.tensor([class_counts[0]/class_counts[1]]).to(DEVICE)  # Auto-calculate
    # Handle class imbalance
    pos_weight = torch.tensor([5.0]).to(DEVICE)  # Adjust based on your dataset
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = FocalLoss(alpha=0.25, gamma=2)
    # Train model
    trained_model = train(
        model, train_loader, val_loader,
        criterion, optimizer, EPOCHS, DEVICE
    )

    # Final evaluation
    trained_model.load_state_dict(torch.load('best_model.pth'))
    test_metrics = evaluate(trained_model, val_loader, DEVICE)
    print("\nFinal Test Performance:")
    print(f"AUC: {test_metrics['auc']:.4f} | F1: {test_metrics['f1']:.4f} | Accuracy: {test_metrics['acc']:.4f}")