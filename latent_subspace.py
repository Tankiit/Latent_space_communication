import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class RotatedMNIST(Dataset):
    def __init__(self, mnist_dataset, rotation_range=(-180, 180)):
        self.mnist_dataset = mnist_dataset
        self.rotation_range = rotation_range

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        image, label = self.mnist_dataset[idx]
        angle = np.random.uniform(*self.rotation_range)
        rotated_image = transforms.functional.rotate(image, angle)
        return rotated_image, label, angle

class InvariantVAE(nn.Module):
    def __init__(self, latent_dim=20, invariant_dim=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.invariant_dim = invariant_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

        # Classifier for invariant features
        self.classifier = nn.Linear(invariant_dim, 10)

        # Rotation predictor for variant features
        self.rotation_predictor = nn.Linear(latent_dim - invariant_dim, 1)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        
        # Split latent space into invariant and variant parts
        z_invariant = z[:, :self.invariant_dim]
        z_variant = z[:, self.invariant_dim:]
        
        class_pred = self.classifier(z_invariant)
        rotation_pred = self.rotation_predictor(z_variant)
        
        return x_recon, class_pred, rotation_pred, mu, logvar, z_invariant, z_variant

def loss_function(recon_x, x, class_pred, class_true, rotation_pred, rotation_true, mu, logvar, z_invariant, z_variant, beta=1.0, gamma=10.0):
    recon_loss = nn.MSELoss(reduction='sum')(recon_x, x)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    classification_loss = nn.CrossEntropyLoss()(class_pred, class_true)
    rotation_loss = nn.MSELoss()(rotation_pred.squeeze(), rotation_true)
    
    # Encourage independence between invariant and variant subspaces
    independence_loss = torch.abs(torch.sum(z_invariant @ z_variant.t()))
    
    return recon_loss + beta * kld_loss + classification_loss + gamma * rotation_loss + 0.1 * independence_loss

def train(model, train_loader, optimizer, device, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target, angle) in enumerate(train_loader):
        data, target, angle = data.to(device), target.to(device), angle.to(device)
        optimizer.zero_grad()
        recon_batch, class_pred, rotation_pred, mu, logvar, z_invariant, z_variant = model(data)
        loss = loss_function(recon_batch, data, class_pred, target, rotation_pred, angle, mu, logvar, z_invariant, z_variant)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, angle in test_loader:
            data, target, angle = data.to(device), target.to(device), angle.to(device)
            recon_batch, class_pred, rotation_pred, mu, logvar, z_invariant, z_variant = model(data)
            test_loss += loss_function(recon_batch, data, class_pred, target, rotation_pred, angle, mu, logvar, z_invariant, z_variant).item()
            pred = class_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')
    print(f'====> Test set accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.2f}%)')

def visualize_latent_space(model, test_loader, device):
    model.eval()
    z_invariant_list = []
    z_variant_list = []
    labels_list = []
    angles_list = []
    
    with torch.no_grad():
        for data, target, angle in test_loader:
            data = data.to(device)
            _, _, _, _, _, z_invariant, z_variant = model(data)
            z_invariant_list.append(z_invariant.cpu().numpy())
            z_variant_list.append(z_variant.cpu().numpy())
            labels_list.append(target.numpy())
            angles_list.append(angle.numpy())
    
    z_invariant = np.concatenate(z_invariant_list)
    z_variant = np.concatenate(z_variant_list)
    labels = np.concatenate(labels_list)
    angles = np.concatenate(angles_list)
    
    # Visualize invariant subspace
    tsne_invariant = TSNE(n_components=2, random_state=42)
    z_invariant_2d = tsne_invariant.fit_transform(z_invariant)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_invariant_2d[:, 0], z_invariant_2d[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title('Invariant Latent Space')
    plt.savefig('invariant_latent_space.png')
    plt.close()
    
    # Visualize variant subspace
    tsne_variant = TSNE(n_components=2, random_state=42)
    z_variant_2d = tsne_variant.fit_transform(z_variant)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_variant_2d[:, 0], z_variant_2d[:, 1], c=angles, cmap='hsv')
    plt.colorbar(scatter)
    plt.title('Variant Latent Space')
    plt.savefig('variant_latent_space.png')
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = MNIST('data', train=True, download=False, transform=transform)
    mnist_test = MNIST('data', train=False, download=False, transform=transform)
    
    # Create rotated MNIST datasets
    rotated_train = RotatedMNIST(mnist_train)
    rotated_test = RotatedMNIST(mnist_test)
    
    # Create data loaders
    train_loader = DataLoader(rotated_train, batch_size=128, shuffle=True)
    test_loader = DataLoader(rotated_test, batch_size=128, shuffle=False)
    
    # Initialize model
    model = InvariantVAE().to(device)
    optimizer = optim.Adam(model.parameters())
    
    # Training loop
    for epoch in range(1, 51):
        train(model, train_loader, optimizer, device, epoch)
        test(model, test_loader, device)
        
        if epoch % 10 == 0:
            visualize_latent_space(model, test_loader, device)
    
    print("Training completed.")

if __name__ == "__main__":
    main()