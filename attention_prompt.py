import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np
from transformers import AutoModel, AutoTokenizer

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
        return rotated_image, label, torch.tensor(angle, dtype=torch.float32)

class InterpretableAttentionLLMRotatedMNIST(nn.Module):
    def __init__(self, llm_model_name="bert-base-uncased", num_heads=8):
        super().__init__()
        
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.llm = AutoModel.from_pretrained(llm_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        
        self.feature_extractor = nn.Linear(self.llm.config.hidden_size, 128 * num_heads)
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=128, num_heads=num_heads)
        
        self.rotation_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.digit_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        
        self.feature_interpreter = nn.Linear(128, 128)  # Maps attended features to interpretable space

    def forward(self, image, prompt):
        # Encode image
        x = self.cnn_encoder(image)  # Shape: [batch_size, 128, H, W]
        batch_size, C, H, W = x.shape
        x = x.view(batch_size, C, -1).permute(2, 0, 1)  # Shape: [H*W, batch_size, C]

        # Encode prompt
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        prompt_embedding = self.llm(**prompt_inputs).last_hidden_state  # Shape: [batch_size, seq_len, hidden_size]

        # Extract attention queries from prompt
        attention_queries = self.feature_extractor(prompt_embedding).view(batch_size, -1, 128).permute(1, 0, 2)

        # Multi-head attention
        attended_features, attention_weights = self.multi_head_attention(attention_queries, x, x)
        attended_features = attended_features.mean(dim=0)  # Shape: [batch_size, C]

        # Domain-Specific prediction (Rotation)
        rotation_pred = self.rotation_predictor(attended_features)

        # Domain-Independent prediction (Digit)
        digit_pred = self.digit_classifier(attended_features)

        # Interpretable latent space
        interpretable_features = self.feature_interpreter(attended_features)

        return digit_pred, rotation_pred, interpretable_features, attention_weights

def generate_prompt(angle, digit):
    features = [
        "orientation", "curvature", "stroke width", "open loops", "closed loops",
        "vertical lines", "horizontal lines", "diagonal lines", "intersections"
    ]
    selected_feature = np.random.choice(features)
    return f"Focus on the {selected_feature} of the digit {digit} rotated by {angle:.2f} degrees."

def train(model, train_loader, optimizer, device, epoch):
    model.train()
    for batch_idx, (data, target, angle) in enumerate(train_loader):
        data, target, angle = data.to(device), target.to(device), angle.to(device).float()
        
        prompts = [generate_prompt(a.item(), t.item()) for a, t in zip(angle, target)]
        
        optimizer.zero_grad()
        digit_pred, rotation_pred, interpretable_features, attention_weights = model(data, prompts)
        
        digit_loss = nn.CrossEntropyLoss()(digit_pred, target)
        rotation_loss = nn.MSELoss()(rotation_pred.squeeze(), angle)
        
        # Add interpretability loss (e.g., sparsity in interpretable features)
        interpretability_loss = torch.norm(interpretable_features, p=1)
        
        total_loss = digit_loss + rotation_loss + 0.01 * interpretability_loss
        total_loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {total_loss.item():.6f}')

def test(model, test_loader, device):
    model.eval()
    digit_correct = 0
    rotation_mse = 0
    
    with torch.no_grad():
        for data, target, angle in test_loader:
            data, target, angle = data.to(device), target.to(device), angle.to(device).float()
            
            prompts = [generate_prompt(a.item(), t.item()) for a, t in zip(angle, target)]
            
            digit_pred, rotation_pred, interpretable_features, attention_weights = model(data, prompts)
            
            digit_correct += digit_pred.argmax(dim=1).eq(target).sum().item()
            rotation_mse += nn.MSELoss(reduction='sum')(rotation_pred.squeeze(), angle).item()

    digit_accuracy = 100. * digit_correct / len(test_loader.dataset)
    rotation_mse /= len(test_loader.dataset)
    
    print(f'Test set: Digit Accuracy: {digit_accuracy:.2f}%, Rotation MSE: {rotation_mse:.4f}')
    return digit_accuracy, rotation_mse, interpretable_features, attention_weights

def analyze_interpretable_features(interpretable_features, attention_weights, prompts):
    # This function would analyze the interpretable features and attention weights
    # For demonstration, we'll just print some basic statistics
    print("Interpretable Feature Statistics:")
    print(f"Mean: {interpretable_features.mean().item()}")
    print(f"Std Dev: {interpretable_features.std().item()}")
    
    print("\nAttention Weight Statistics:")
    print(f"Mean: {attention_weights.mean().item()}")
    print(f"Std Dev: {attention_weights.std().item()}")
    
    # In a real scenario, you might:
    # 1. Visualize attention weights as heatmaps overlaid on the original images
    # 2. Cluster interpretable features to find common patterns
    # 3. Correlate interpretable features with prompt content
    # 4. Analyze how different prompts affect attention and interpretable features

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = MNIST('./data', train=False, download=True, transform=transform)
    
    # Create rotated MNIST datasets
    rotated_train = RotatedMNIST(mnist_train)
    rotated_test = RotatedMNIST(mnist_test)
    
    # Create data loaders
    train_loader = DataLoader(rotated_train, batch_size=128, shuffle=True)
    test_loader = DataLoader(rotated_test, batch_size=128, shuffle=False)
    
    model = InterpretableAttentionLLMRotatedMNIST().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(1, 11):
        train(model, train_loader, optimizer, device, epoch)
        digit_accuracy, rotation_mse, interpretable_features, attention_weights = test(model, test_loader, device)
        
        # Analyze interpretable features and attention weights
        analyze_interpretable_features(interpretable_features, attention_weights, prompts)

if __name__ == "__main__":
    main()