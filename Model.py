import torch
import clip
from PIL import Image
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import json

with open("./dataset_cache/answer_to_idx.json", 'r') as f:
    s = f.read()
    answer_to_idx = json.loads(s)

print(len(answer_to_idx))
class FusionLayer(nn.Module):
    
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        
        self.layernorm = nn.LayerNorm(embed_dim)
        
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=0.0, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )
        
    def forward(self, image_embs, text_embs):
        
        attn_out = self.attn(text_embs, image_embs, image_embs)[0]
        attn_out = self.layernorm(attn_out) + attn_out
        out = self.mlp(attn_out)
        return out
    
    

class Fusor(pl.LightningModule):
    
    def __init__(self, embedding_size, num_heads, output_size, device='cuda', lr = 0.001):
        super().__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model.requires_grad_(requires_grad=False)
        self.lr = lr
        self.text_linear_map = nn.Linear(embedding_size, embedding_size)
        self.image_linear_map = nn.Linear(embedding_size, embedding_size)
        
        self.fusion = FusionLayer(embedding_size, num_heads)
        
        self.linear_map = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(embedding_size, output_size),
            # nn.ReLU()
        )

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, image_embs, text_embs):
        
        text_embs = self.text_linear_map(text_embs)
        image_embs = self.image_linear_map(image_embs)

        attn_out = self.fusion(image_embs, text_embs)
        
        out = self.linear_map(attn_out)
        return out
        
    def encode_image(self, preprocessed_image):
        image_features = self.model.encode_image(preprocessed_image)        
        return image_features
    
    def encode_text(self, tokenized_text):
        text_features = self.model.encode_text(tokenized_text)
        return text_features
    
    def training_step(self, batch, batch_idx):
        x1, x2, targets = batch
        pred = self(x1, x2)
        l = self.loss_func(pred, targets)
        # train_losses.append(l.item())
        self.log('train loss', l, prog_bar=True)
        
        # Calculate accuracy
        # preds = torch.argmax(pred, dim=-1)
        # acc = (preds == targets).float().mean()
        # self.log("train_acc", acc, prog_bar=True)
        return l
    
    def validation_step(self, batch, batch_idx):
        x1, x2, targets = batch
        pred = self(x1, x2)
        l = self.loss_func(pred, targets)
        # train_losses.append(l.item())
        # preds = torch.argmax(pred, dim=-1)
        # acc = (preds == targets).float().mean()
        
        self.log("val_loss", l, prog_bar=True)
        # self.log("val_acc", acc, prog_bar=True)
        
        
        return l
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": None,
            # "monitor": "val_loss"
        }


def create_data():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data and move to device
    X1 = torch.load("./dataset_cache/images_embs.pt", weights_only=True)  # Add weights_only=True to avoid warning

    X2 = torch.load("./dataset_cache/text_embs.pt", weights_only=True)
    print('Shape of data:', X1.shape, X2.shape, X1.dtype, X2.dtype)
    
    X1 = X1.to(torch.float32)
    X2 = X2.to(torch.float32)
    
    Y = torch.load("./dataset_cache/targets.pt", weights_only=True)  # Add weights_only=True to avoid warning
    eye = torch.eye(len(answer_to_idx))[Y]
    # print(eye.shape, eye[0][Y[0]-4:Y[0]+3], eye[1][Y[1]-4:Y[1]+3], Y[:2])
    
    # Y_list = Y.tolist()
    # print(len(set(Y)))
    
    # Keep data on CPU for now (will move to GPU during training)
    return TensorDataset(X1, X2, eye)

def get_dataloaders(batch_size=32):
    dataset = create_data()
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    # Create a generator that matches your device
    generator = torch.Generator()
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# # Initialize and train model
def train_model():
    # input_size, hidden_size, output_size = 256, 512, 2
    # model = FaceRecognizer(2562)
    model = Fusor(512, 4 ,len(answer_to_idx))
    train_loader, val_loader = get_dataloaders()

    trainer = pl.Trainer(max_epochs=20,
                        enable_progress_bar=True,  # Disable default tqdm ba
                        num_nodes=1,
                        enable_checkpointing=True
                        )
    
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":

    device = 'cpu'

    idx2answers = {answer_to_idx[key]:key for key in answer_to_idx}
    
    model = Fusor.load_from_checkpoint(checkpoint_path="./lightning_logs/version_0/checkpoints/epoch=1-step=686.ckpt", embedding_size=512, num_heads = 4, output_size=len(answer_to_idx)).to(device)
    # model = Fusor(embedding_size=512, num_heads=4, output_size=len(answer_to_idx), device=device)
    _, preprocess = clip.load("ViT-B/32", device=device)
    image = preprocess(Image.open(r"C:\Users\Rohit Francis\Desktop\Codes\Datasets\VQA\dataset\images\image5.png")).unsqueeze(0).to(device)
    text = clip.tokenize(["is there a table in the image ?"]).to(device)
    print(text)
    image_embs = model.encode_image(image).to(torch.float32)
    text_embs = model.encode_text(text).to(torch.float32)
    
    out = model(image_embs, text_embs)
    idx = torch.argmax(out, dim=-1)
    print(idx)
    print(idx2answers[idx.item()])
    
    # print(image_embs.shape, text_embs.shape)
    
    # out = model(image_embs, text_embs)

    # print(out.shape)
    
    # train_model()
    
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-B/32", device=device)

    # image = preprocess(Image.open("Clip.jpeg")).unsqueeze(0).to(device)
    # text = clip.tokenize(["a clip", "a dog", "a cat"]).to(device)

    # with torch.no_grad():
    #     image_features = model.encode_image(image)
    #     text_features = model.encode_text(text)
        
    #     logits_per_image, logits_per_text = model(image, text)
    #     probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]