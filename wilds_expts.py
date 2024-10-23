import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import wandb
from transformers import CLIPProcessor, CLIPModel

class VisualTextAlignedVAE(nn.Module):
    def __init__(self, visual_dim, text_dim, content_dim=64, domain_dim=32, num_classes=10, num_domains=10):
        super().__init__()
        self.content_dim = content_dim
        self.domain_dim = domain_dim
        total_latent_dim = content_dim + domain_dim

        # Visual encoder
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Text encoder
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Shared latent space
        self.fc_mu = nn.Linear(512, total_latent_dim)
        self.fc_logvar = nn.Linear(512, total_latent_dim)

        # Decoder (reconstructs visual features)
        self.decoder = nn.Sequential(
            nn.Linear(total_latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, visual_dim)
        )

        # Classifiers
        self.content_classifier = nn.Linear(content_dim, num_classes)
        self.domain_classifier = nn.Linear(domain_dim, num_domains)

    def encode(self, visual_features, text_features):
        visual_encoded = self.visual_encoder(visual_features)
        text_encoded = self.text_encoder(text_features)
        combined = torch.cat([visual_encoded, text_encoded], dim=1)
        return self.fc_mu(combined), self.fc_logvar(combined)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, visual_features, text_features):
        mu, logvar = self.encode(visual_features, text_features)
        z = self.reparameterize(mu, logvar)
        visual_recon = self.decode(z)
        
        z_content = z[:, :self.content_dim]
        z_domain = z[:, self.content_dim:]
        
        content_pred = self.content_classifier(z_content)
        domain_pred = self.domain_classifier(z_domain)
        
        return visual_recon, content_pred, domain_pred, mu, logvar, z_content, z_domain

class LLMFeatureExtractor:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)

    def extract_features(self, images, prompts):
        inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        return outputs.image_embeds, outputs.text_embeds

def generate_prompt(metadata):
    # Example prompt generation based on metadata
    return f"An image of a product from the {metadata['domain']} category with a {metadata['label']} sentiment."

def loss_function(recon_visual, visual_features, content_pred, content_true, domain_pred, domain_true, mu, logvar, z_content, z_domain, beta=1.0, lambda_domain=0.1):
    recon_loss = nn.MSELoss(reduction='sum')(recon_visual, visual_features)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    content_loss = nn.CrossEntropyLoss()(content_pred, content_true)
    domain_loss = nn.CrossEntropyLoss()(domain_pred, domain_true)
    
    # Encourage independence between subspaces
    independence_loss = torch.abs(torch.sum(z_content @ z_domain.t()))
    
    return (recon_loss + 
            beta * kld_loss + 
            content_loss + 
            lambda_domain * domain_loss + 
            0.1 * independence_loss)

def train(model, llm_extractor, train_loader, optimizer, device, epoch):
    model.train()
    train_loss = 0
    for batch in train_loader:
        x, y_true, metadata = batch
        x, y_true = x.to(device), y_true.to(device)
        domain_labels = metadata['domain'].to(device)

        # Generate prompts and extract LLM features
        prompts = [generate_prompt(m) for m in metadata]
        visual_features, text_features = llm_extractor.extract_features(x, prompts)
        visual_features, text_features = visual_features.to(device), text_features.to(device)
        
        optimizer.zero_grad()
        recon_visual, content_pred, domain_pred, mu, logvar, z_content, z_domain = model(visual_features, text_features)
        loss = loss_function(recon_visual, visual_features, content_pred, y_true, domain_pred, domain_labels, mu, logvar, z_content, z_domain)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
    train_accuracy = (content_pred.argmax(dim=1) == y_true).float().mean().item()
    wandb.log({
        "train_loss": train_loss / len(train_loader.dataset),
        "train_accuracy": train_accuracy,
        
    })
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

def test(model, llm_extractor, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    all_labels = []
    all_preds = []
    z_content_list, z_domain_list, labels_list, domain_list = [], [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
            x, y_true, metadata = batch
            x, y_true = x.to(device), y_true.to(device)
            domain_labels = metadata['domain'].to(device)

            # Generate prompts and extract LLM features
            prompts = [generate_prompt(m) for m in metadata]
            visual_features, text_features = llm_extractor.extract_features(x, prompts)
            visual_features, text_features = visual_features.to(device), text_features.to(device)
            
            recon_visual, content_pred, domain_pred, mu, logvar, z_content, z_domain = model(visual_features, text_features)
            loss = loss_function(recon_visual, visual_features, content_pred, y_true, domain_pred, domain_labels, mu, logvar, z_content, z_domain)
            test_loss += loss.item()
            pred = content_pred.argmax(dim=1)
            correct += pred.eq(y_true).sum().item()
            
            all_labels.extend(y_true.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            
            z_content_list.append(z_content.cpu().numpy())
            z_domain_list.append(z_domain.cpu().numpy())
            labels_list.append(y_true.cpu().numpy())
            domain_list.append(domain_labels.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')
    print(f'====> Test set accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": accuracy
       
    })
    
    return np.concatenate(z_content_list), np.concatenate(z_domain_list), np.concatenate(labels_list), np.concatenate(domain_list)

def visualize_latent_space(z_content, z_domain, labels, domains, epoch):
    # Visualize content subspace
    tsne_content = TSNE(n_components=2, random_state=42)
    z_content_2d = tsne_content.fit_transform(z_content)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_content_2d[:, 0], z_content_2d[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title(f'Content Latent Space (Epoch {epoch})')
    plt.savefig(f'content_latent_space_epoch_{epoch}.png')
    wandb.log({"content_latent_space": wandb.Image(plt)})
    plt.close()
    
    # Visualize domain subspace
    tsne_domain = TSNE(n_components=2, random_state=42)
    z_domain_2d = tsne_domain.fit_transform(z_domain)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_domain_2d[:, 0], z_domain_2d[:, 1], c=domains, cmap='Set1')
    plt.colorbar(scatter)
    plt.title(f'Domain Latent Space (Epoch {epoch})')
    plt.savefig(f'domain_latent_space_epoch_{epoch}.png')
    wandb.log({"domain_latent_space": wandb.Image(plt)})
    plt.close()

def main():
    wandb.init(project="visual-text-aligned-vae")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load WILDS dataset (e.g., FMOW)
    dataset = get_dataset(dataset="fmow", download=True)
    train_data = dataset.get_subset("train", transform=dataset.get_transform("train"))
    test_data = dataset.get_subset("test", transform=dataset.get_transform("test"))
    
    train_loader = get_train_loader("standard", train_data, batch_size=32)
    test_loader = get_eval_loader("standard", test_data, batch_size=32)
    
    # Initialize model and LLM feature extractor
    visual_dim = 512  # CLIP's image embedding dimension
    text_dim = 512    # CLIP's text embedding dimension
    model = VisualTextAlignedVAE(visual_dim=visual_dim, text_dim=text_dim, 
                                 num_classes=dataset.n_classes, num_domains=dataset.n_domains).to(device)
    llm_extractor = LLMFeatureExtractor()
    optimizer = optim.Adam(model.parameters())
    wandb.watch(model)
    
    # Training loop
    for epoch in range(1, 51):
        train(model, llm_extractor, train_loader, optimizer, device, epoch)
        z_content, z_domain, labels, domains = test(model, llm_extractor, test_loader, device)
        
        if epoch % 10 == 0:
            visualize_latent_space(z_content, z_domain, labels, domains, epoch)
    
    print("Training completed.")
    wandb.finish()

if __name__ == "__main__":
    main()