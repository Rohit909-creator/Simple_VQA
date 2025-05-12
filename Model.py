import torch
import clip
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import json
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

with open("./dataset_cache/answer_to_idx.json", 'r') as f:
    s = f.read()
    answer_to_idx = json.loads(s)

# print(f"Total unique answers: {len(answer_to_idx)}")

class FusionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        
        # Add layer normalization for inputs
        self.image_layernorm = nn.LayerNorm(embed_dim)
        self.text_layernorm = nn.LayerNorm(embed_dim)
        
        # Multi-head attention with dropout
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # Output layer normalization after attention
        self.out_layernorm = nn.LayerNorm(embed_dim)
        
        # Enhanced MLP with dropout
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),  # Wider hidden layer
            nn.GELU(),  # Using GELU activation (better than ReLU for transformers)
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Final layer normalization
        self.final_layernorm = nn.LayerNorm(embed_dim)
        
    def forward(self, image_embs, text_embs):
        # Apply layer normalization to inputs
        image_embs = self.image_layernorm(image_embs)
        text_embs = self.text_layernorm(text_embs)
        
        # Attention with residual connection
        attn_out, _ = self.attn(text_embs, image_embs, image_embs)
        attn_out = text_embs + attn_out  # Residual connection
        attn_out = self.out_layernorm(attn_out)  # Layer normalization
        
        # MLP with residual connection
        mlp_out = self.mlp(attn_out)
        out = attn_out + mlp_out  # Residual connection
        out = self.final_layernorm(out)  # Final layer normalization
        
        return out

class CrossModalFusion(nn.Module):
    """Cross-modal fusion module to better combine image and text features"""
    def __init__(self, embed_dim):
        super().__init__()
        self.image_proj = nn.Linear(embed_dim, embed_dim)
        self.text_proj = nn.Linear(embed_dim, embed_dim)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        
    def forward(self, image_embs, text_embs):
        # Project features
        img_proj = self.image_proj(image_embs)
        txt_proj = self.text_proj(text_embs)
        
        # Concatenate for gating
        concat = torch.cat([img_proj, txt_proj], dim=-1)
        gate = self.gate(concat)
        
        # Element-wise gated fusion
        fused = gate * img_proj + (1 - gate) * txt_proj
        return fused

class EnhancedFusor(pl.LightningModule):
    def __init__(self, embedding_size=512, num_heads=8, output_size=None, 
                 dropout=0.1, lr=3e-4, weight_decay=0.01, device='cuda'):
        super().__init__()
        
        # Auto-detect output size if not specified
        if output_size is None:
            with open("./dataset_cache/answer_to_idx.json", 'r') as f:
                s = f.read()
                answer_to_idx = json.loads(s)
                output_size = len(answer_to_idx)
        
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.lr = lr
        self.weight_decay = weight_decay
        
        # Load CLIP model and freeze it
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model.requires_grad_(requires_grad=False)
        
        # Feature projection layers with dropout
        self.text_projection = nn.Sequential(
            nn.LayerNorm(embedding_size),
            nn.Linear(embedding_size, embedding_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.image_projection = nn.Sequential(
            nn.LayerNorm(embedding_size),
            nn.Linear(embedding_size, embedding_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Cross-modal fusion before attention
        self.cross_fusion = CrossModalFusion(embedding_size)
        
        # Attention fusion layer
        self.fusion = FusionLayer(embedding_size, num_heads, dropout)
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size, embedding_size*2),
            nn.LayerNorm(embedding_size*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size*2, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, output_size)
        )

        # Label smoothing cross entropy loss
        self.loss_func = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Save hyperparameters for easy checkpoint loading
        self.save_hyperparameters()

    def forward(self, image_embs, text_embs):
        # Apply projection layers
        text_features = self.text_projection(text_embs)
        image_features = self.image_projection(image_embs)
        
        # Initial cross-modal fusion
        fused_features = self.cross_fusion(image_features, text_features)
        
        # Apply attention fusion
        attn_out = self.fusion(image_features, text_features)
        
        # Add the initial fusion with attention output
        combined = fused_features + attn_out
        
        # Classification
        logits = self.classifier(combined)
        return logits
        
    def encode_image(self, preprocessed_image):
        with torch.no_grad():
            image_features = self.model.encode_image(preprocessed_image)
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    def encode_text(self, tokenized_text):
        with torch.no_grad():
            text_features = self.model.encode_text(tokenized_text)
            # Normalize features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def training_step(self, batch, batch_idx):
        x1, x2, targets = batch
        pred = self(x1, x2)
        loss = self.loss_func(pred, targets)
        
        # Calculate accuracy
        preds = torch.argmax(pred, dim=-1)
        targets_idx = torch.argmax(targets, dim=-1)
        acc = (preds == targets_idx).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return {'loss': loss, 'acc': acc}
    
    def validation_step(self, batch, batch_idx):
        x1, x2, targets = batch
        pred = self(x1, x2)
        loss = self.loss_func(pred, targets)
        
        # Calculate accuracy
        preds = torch.argmax(pred, dim=-1)
        targets_idx = torch.argmax(targets, dim=-1)
        acc = (preds == targets_idx).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return {'val_loss': loss, 'val_acc': acc}
    
    def configure_optimizers(self):
        # AdamW optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=2,
            min_lr=1e-6,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }


def create_data():
    # Load data
    X1 = torch.load("./dataset_cache/images_embs.pt", weights_only=True)
    X2 = torch.load("./dataset_cache/text_embs.pt", weights_only=True)
    print('Shape of data:', X1.shape, X2.shape, X1.dtype, X2.dtype)
    
    # Convert to float32 if needed
    X1 = X1.to(torch.float32)
    X2 = X2.to(torch.float32)
    
    # Normalize embeddings (important for consistent scale)
    X1 = F.normalize(X1, p=2, dim=1)
    X2 = F.normalize(X2, p=2, dim=1)
    
    # Load targets
    Y = torch.load("./dataset_cache/targets.pt", weights_only=True)
    
    # Convert to one-hot encoding
    eye = torch.eye(len(answer_to_idx))[Y.squeeze()]
    
    # Print some stats
    print(f"Total samples: {len(Y)}")
    print(f"Feature dimensions: Image={X1.shape[1]}, Text={X2.shape[1]}")
    print(f"Target classes: {len(answer_to_idx)}")
    
    return TensorDataset(X1, X2, eye)

def get_dataloaders(batch_size=32):
    dataset = create_data()
    
    # Create train/val/test split (80/10/10)
    train_size = int(0.8 * len(dataset))
    # val_size = int(0.2 * len(dataset))
    val_size = len(dataset)-train_size
    # test_size = len(dataset) - train_size - val_size
    
    # Create splits with fixed seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # test_loader = DataLoader(
    #     test_dataset, 
    #     batch_size=batch_size, 
    #     shuffle=False,
    #     num_workers=4,
    #     pin_memory=True
    # )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    # print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader

# Train model
def train_model(max_epochs=30, batch_size=64):
    # Create model and dataloaders
    model = EnhancedFusor(
        embedding_size=512, 
        num_heads=8,
        dropout=0.2,
        lr=5e-4,
        weight_decay=0.01
    )
    
    train_loader, val_loader = get_dataloaders(batch_size)

    # Create trainer with additional callbacks
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        enable_progress_bar=True,
        num_nodes=1,
        enable_checkpointing=True,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='val_loss',
                filename='{epoch}-{val_loss:.2f}-{val_acc:.2f}',
                save_top_k=3,
                mode='min'
            ),
            # pl.callbacks.EarlyStopping(
            #     monitor='val_loss',
            #     patience=5,
            #     mode='min'
            # ),
        ],
        gradient_clip_val=1.0,  # Prevent exploding gradients
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    # Test the model
    # test_result = trainer.test(model, test_loader)
    # print(f"Test results: {test_result}")
    
    return model, trainer

def inference_example(checkpoint_path=None):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create reverse mapping from index to answer
    idx2answers = {v: k for k, v in answer_to_idx.items()}
    
    # Load model from checkpoint or create new
    if checkpoint_path:
        model = EnhancedFusor.load_from_checkpoint(
            checkpoint_path=checkpoint_path
        ).to(device)
        print(f"Loaded model from {checkpoint_path}")
    else:
        model = EnhancedFusor().to(device)
        print("Created new model (not trained)")
    
    # Set model to evaluation mode
    model.eval()
    
    # Load and preprocess image
    _, preprocess = clip.load("ViT-B/32", device=device)
    image_path = r"C:\Users\Rohit Francis\Desktop\Codes\Datasets\VQA\dataset\images\image1.png"  # Adjust path as needed
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    # Create text query
    # query = "what is on the left side of white oven ?"
    query = "how many garbage_bin is here in the image ?"
    text = clip.tokenize([query]).to(device)
    
    print(f"Query: {query}")
    
    # Get embeddings
    with torch.no_grad():
        image_embs = model.encode_image(image).to(torch.float32)
        text_embs = model.encode_text(text).to(torch.float32)
    
    # Get prediction
    with torch.no_grad():
        out = model(image_embs, text_embs)
        
    # Get top-5 predictions
    top_probs, top_indices = torch.topk(F.softmax(out, dim=-1), k=5)
    
    # Display results
    print("\nTop 5 predictions:")
    for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
        answer = idx2answers.get(idx.item(), "unknown")
        print(f"{i+1}. {answer} ({prob.item()*100:.2f}%)")


if __name__ == "__main__":
    # Uncomment to train the model
    # train_model(max_epochs=30, batch_size=64)
    inference_example("./lightning_logs/version_4/checkpoints/epoch=7-val_loss=3.66-val_acc=0.32.ckpt")
    # For inference with trained model (replace with your checkpoint path)
    # inference_example(checkpoint_path="./lightning_logs/version_0/checkpoints/epoch=1-step=686.ckpt")