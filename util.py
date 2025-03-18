import matplotlib.pyplot as plt

def plot_training_history(train_losses, val_losses, val_metrics, save_path='training_history.png'):
    """Plot training history including losses and validation metrics."""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot validation metrics
    ax2.plot(epochs, [x['auc'] for x in val_metrics], 'r-', label='AUC')
    ax2.plot(epochs, [x['f1'] for x in val_metrics], 'g-', label='F1')
    ax2.plot(epochs, [x['acc'] for x in val_metrics], 'b-', label='Accuracy')
    ax2.set_title('Validation Metrics')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
