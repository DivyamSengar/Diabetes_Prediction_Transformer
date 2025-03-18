# Diabetes Prediction with Transformers

This project implements two models for diabetes prediction using tabular data:
1. A TabTransformer model that uses self-attention mechanisms
2. A baseline Deep Neural Network (DNN)

## Setup

1. Install the required packages:
```bash
pip install torch pandas numpy sklearn matplotlib imblearn
```

2. Place your diabetes dataset (CSV file) in the project directory.

## Usage

Train a model using the following command:

```bash
python main.py --model [transformer|dnn] --epochs N --batch_size B --data_path path/to/data.csv
```

### Arguments:
- `--model`: Choose between 'transformer' or 'dnn' (default: transformer)
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size for training (default: 64)
- `--data_path`: Path to the diabetes dataset CSV file (default: 'diabetes.csv')
- `--augment`: Use SMOTE data augmentation to handle class imbalance (optional)

### Example:
```bash
# Train TabTransformer for 30 epochs
python main.py --model transformer --epochs 30

# Train baseline DNN with data augmentation
python main.py --model dnn --augment
```

## Output

The training process will:
1. Save the best model checkpoint based on validation AUC
2. Generate training history plots showing:
   - Training loss over epochs
   - Validation metrics (AUC, F1, Accuracy) over epochs
3. Display final test performance metrics

## Project Structure

- `main.py`: Main training script with command-line interface
- `model.py`: Model architectures (TabTransformer and BaselineDNN)
- `dataloader.py`: Data preprocessing and loading utilities
- `utils.py`: Plotting and utility functions