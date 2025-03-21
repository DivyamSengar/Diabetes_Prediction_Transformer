# Exploring Diabetes Prediction with Tabular Transformers

This project explores the application of transformer-based architectures for predicting diabetes from tabular health data. We compare a custom TabTransformer against a traditional Deep Neural Network (DNN) baseline and an Ensemble Model to evaluate the effectiveness of self-attention mechanisms in capturing complex inter-feature relationships within structured data.

The TabTransformer, trained with a Focal Loss criterion, demonstrated superior performance, achieving an F1-score of **0.7907**, AUC of **0.9779**, and an accuracy of **96.57%** on the validation set. This highlights the potential of transformer architectures to generalize well on complex, class-imbalanced medical datasets, as is consistent with the real world.

---

## 📋 Key Features

* **Advanced Model Architectures:** Implements and compares a Baseline DNN, a TabTransformer, and an Ensemble model.
* **Robust Preprocessing:** Features a data pipeline that handles categorical and numerical data, including one-hot encoding for the `smoking_history` feature.
* **Class Imbalance Handling:** Utilizes techniques like a weighted sampler and a Focal Loss function to effectively manage the highly imbalanced dataset.
* **Threshold Optimization:** Implements a dynamic thresholding technique to find the optimal classification threshold that maximizes the F1-score on a validation set.
* **Comprehensive Evaluation:** Models are evaluated on Accuracy, AUC, and F1-score, with a strong emphasis on the F1-score as the primary metric due to class imbalance.

---

## 🧠 Model Architectures

The project investigates three different architectures to tackle the prediction task:

1. **Baseline DNN**
   A standard Feedforward Neural Network (FFNN) that serves as our performance baseline. The network processes a flattened vector of one-hot encoded categorical features and scaled numerical features. The architecture consists of two hidden layers (`[64, 32]`) with ReLU activations and Dropout.

2. **TabTransformer**
   A transformer-based model inspired by the original TabTransformer paper.

   * **Embeddings:** Categorical features are projected into a high-dimensional space using a linear layer, and numerical features are processed with a separate linear projection.
   * **Self-Attention:** A stack of multi-head self-attention layers processes the combined feature embeddings, allowing the model to learn deep, contextual interactions between features.
   * **Hyperparameters:** The model uses a hidden dimension of 64, 8 attention heads, and 6 transformer layers.

3. **Ensemble Model**
   This model was designed to leverage the unique strengths of the two architectures above. It processes data in parallel through the Baseline DNN and the TabTransformer. The outputs from both models are then fed into a final linear "meta-classifier" to produce the final prediction. However, our experiments showed that this approach did not yield performance gains over the standalone TabTransformer.

---

## 📊 Performance Results

The final evaluation metrics on the validation set clearly demonstrate the superiority of the TabTransformer architecture. The model's ability to capture complex dependencies between features was crucial for its success.

| Model          | Accuracy   | F1 Score   | AUC        |
| -------------- | ---------- | ---------- | ---------- |
| Baseline DNN   | 0.6985     | 0.3594     | 0.9699     |
| Ensemble Model | 0.9608     | 0.7644     | 0.9737     |
| TabTransformer | **0.9657** | **0.7907** | **0.9779** |

---

## 🚀 Getting Started

Follow these steps to set up the environment and run the project.

### 1. Prerequisites

* Python 3.8+
* PyTorch
* Git

### 2. Installation

First, clone the repository to your local machine:

```bash
git clone <your-repo-link>
cd <your-repo-directory>
```

Next, it's recommended to create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

Install the required dependencies. You can create a `requirements.txt` file with the following content:

```
torch
numpy
pandas
scikit-learn
matplotlib
imblearn
```

Then, install the packages:

```bash
pip install -r requirements.txt
```

### 3. Dataset

Download the Diabetes Prediction Dataset from Kaggle. Place the `diabetes_prediction_dataset.csv` file in the root directory of the project.

---

## ⚙️ Usage

The main script `main.py` is used for training and evaluating the models. You can select the model to run using the `--model_type` command-line argument.

Train the TabTransformer (Default & Best Performing):

```bash
python main.py --model_type tabular
```

Train the Baseline DNN:

```bash
python main.py --model_type baseline
```

Train the Ensemble Model:

```bash
python main.py --model_type ensemble
```

After training completes, the best model checkpoint (`best_model.pth`) will be saved, and a plot of the training curves (`training_metrics.png`) will be generated. The final evaluation results on the test set will be printed to the console.

