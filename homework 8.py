
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model with corrected input size
class HairTypeClassifier(nn.Module):
    def __init__(self):
        super(HairTypeClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 100 * 100, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create model
model = HairTypeClassifier().to(device)

# Loss function
criterion = nn.BCEWithLogitsLoss()

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.8)

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

# Data transforms without augmentation
train_transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = ImageFolder('data/train', transform=train_transforms)
test_dataset = ImageFolder('data/test', transform=train_transforms)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

# Training function
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    history = {'acc': [], 'loss': [], 'test_acc': [], 'test_loss': []}
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct_train / total_train
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_acc)
        
        # Test evaluation
        model.eval()
        test_running_loss = 0.0
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels = labels.float().unsqueeze(1)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_running_loss += loss.item() * images.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        
        test_epoch_loss = test_running_loss / len(test_dataset)
        test_epoch_acc = correct_test / total_test
        history['test_loss'].append(test_epoch_loss)
        history['test_acc'].append(test_epoch_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, "
              f"Test Loss: {test_epoch_loss:.4f}, Test Acc: {test_epoch_acc:.4f}")
    
    return history

# First training without augmentation (10 epochs)
print("Training without data augmentation...")
history_no_aug = train_model(model, train_loader, test_loader, criterion, optimizer, 10)

# Calculate statistics
train_acc_median = np.median(history_no_aug['acc'])
train_loss_std = np.std(history_no_aug['loss'])

print(f"\nMedian of training accuracy: {train_acc_median:.4f}")
print(f"Standard deviation of training loss: {train_loss_std:.4f}")

# Data augmentation transforms
train_transforms_aug = transforms.Compose([
    transforms.RandomRotation(50),
    transforms.RandomResizedCrop(200, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create new training dataset with augmentation
train_dataset_aug = ImageFolder('data/train', transform=train_transforms_aug)
train_loader_aug = DataLoader(train_dataset_aug, batch_size=20, shuffle=True)

# Continue training with augmentation for 10 more epochs
print("\nContinuing training with data augmentation...")
history_with_aug = train_model(model, train_loader_aug, test_loader, criterion, optimizer, 10)

# Calculate statistics for augmented training
test_loss_mean = np.mean(history_with_aug['test_loss'])
test_acc_last5 = np.mean(history_with_aug['test_acc'][5:])  # Last 5 epochs

print(f"\nMean of test loss with augmentation: {test_loss_mean:.4f}")
print(f"Average test accuracy for last 5 epochs: {test_acc_last5:.4f}")

# Summary of results
print("\nSummary of results:")
print(f"Total model parameters: {total_params}")
print(f"Median training accuracy (no augmentation): {train_acc_median:.4f}")
print(f"Standard deviation of training loss (no augmentation): {train_loss_std:.4f}")
print(f"Mean test loss (with augmentation): {test_loss_mean:.4f}")
print(f"Average test accuracy for last 5 epochs: {test_acc_last5:.4f}")