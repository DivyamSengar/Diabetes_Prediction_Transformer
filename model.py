import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class TabTransformer(nn.Module):
    def __init__(self, categorical_dims, numerical_dim, hidden_dim=64, 
                 n_heads=4, n_layers=3, num_classes=1):
        super().__init__()
        
        # Categorical embeddings
        # self.embeddings = nn.ModuleList([
        #     nn.Embedding(dim, hidden_dim) for dim in categorical_dims
        # ])
        self.categorical_proj = nn.Linear(categorical_dims, hidden_dim)
        # Numerical processing
        self.numerical_proj = nn.Linear(numerical_dim, hidden_dim)
        print(self.numerical_proj)
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, categorical, numerical):
        # Embed categorical features
        # cat_embedded = [emb(categorical[:, i]) for i, emb in enumerate(self.embeddings)]
        # cat_embedded = torch.stack(cat_embedded, dim=1)  # [batch, num_cat, hidden]
        cat_embedded = self.categorical_proj(categorical).unsqueeze(1)
        # Process numerical features
        num_proj = self.numerical_proj(numerical).unsqueeze(1)  # [batch, 1, hidden]
        
        # Combine features
        combined = torch.cat([cat_embedded, num_proj], dim=1)  # [batch, num_cat+1, hidden]
        
        # Transformer
        transformer_out = self.transformer(combined)  # [batch, seq_len, hidden]
        
        # Pooling
        pooled = transformer_out.mean(dim=1)  # [batch, hidden]
        
        # Classification
        # return torch.sigmoid(self.classifier(pooled)).squeeze()
        return self.classifier(pooled).squeeze()

class BaselineDNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], num_classes=1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = dim
            
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)
        
    def forward(self, x):
        features = self.features(x)
        return torch.sigmoid(self.classifier(features)).squeeze()
    
# Add this class to model.py
class EnsembleModel(nn.Module):
    def __init__(self, model1, model2):
        super().__init__()
        self.model1 = model1  # TabTransformer
        self.model2 = model2  # BaselineDNN
        self.meta = nn.Linear(2, 1)  # Combine outputs

    def forward(self, categorical, numerical):
        out1 = self.model1(categorical, numerical)
        inputs = torch.cat([categorical, numerical], dim=1)
        out2 = self.model2(inputs)
        combined = torch.stack([out1, out2], dim=1)
        return self.meta(combined).squeeze()