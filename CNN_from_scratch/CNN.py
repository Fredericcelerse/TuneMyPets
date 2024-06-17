#!/usr/bin/env python

# Importing here the necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset, random_split
from torch.optim.lr_scheduler import StepLR
from PIL import Image

# Define transformations for the input data
data_transforms = transforms.Compose([
	  transforms.RandomResizedCrop(224),	# Randomly crop the images to 224x224
	  transforms.RandomHorizontalFlip(),	# Randomly flip images horizontally
          transforms.RandomRotation(15),	# Randomly rotate images by 15 degrees
          transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05),	# Randomly jitter color
	  transforms.ToTensor(),	# Convert images to tensors
	  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])	# Normalize images (parameters chosen from vgg and resnet)
])

# Path to the directory with images
data_dir = '/Users/celerse/Images/Ex2/kagglecatsanddogs_5340/PetImages'

# Function to check the validity of images
def check_image(path):
	try:
		img = Image.open(path)	# Open the image file
		img.verify()		# Verify that it is, in fact, an image
		img.close()		# Close the file
		return True
	except:
		return False

# Load dataset with custom transformations
dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
print(f"Total number of loaded Images : {len(dataset)}")

# Filter out corrupted images
valid_indices = [i for i in range(len(dataset)) if check_image(dataset.imgs[i][0])]
valid_images = Subset(dataset, valid_indices)
print(f"Number of valid Images: {len(valid_images)}")

# Split dataset into training and validation sets
train_size = int(0.8 * len(valid_images))
val_size = len(valid_images) - train_size
train_dataset, val_dataset = random_split(valid_images, [train_size, val_size])

# DataLoader for loading the dataset
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)	# First convolutional layer
        self.bn1 = nn.BatchNorm2d(32)	# Batch normalization
        self.pool = nn.MaxPool2d(2, 2)	# Max pooling layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)	# Second convolutional layer
        self.bn2 = nn.BatchNorm2d(64)	# Batch normalization
        self.fc1 = nn.Linear(64 * 56 * 56, 1024)	# First fully connected layer
        self.dropout = nn.Dropout(0.5)	# Dropout layer
        self.fc2 = nn.Linear(1024, 2)	# Second fully connected layer

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))	# Apply conv1 > bn1 > relu > pool layers
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))	# Apply conv2 > bn2 > relu > pool layers
        x = x.view(-1, 64 * 56 * 56)	# Flatten the output
        x = torch.relu(self.fc1(x))	# Apply relu activation function on the first fully connected layer
        x = self.dropout(x)	# Apply dropout
        x = self.fc2(x)	# Output layer
        return x

# Initialize model weights
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

# Initialize the model
model = CNN()
model.apply(init_weights)	# Apply weights initialization
criterion = nn.CrossEntropyLoss()	# Loss function
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)	# Optimizer
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)	# Learning rate scheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")	# Setup computation device
model = model.to(device)	# Move model to the designated device

# Function to train the model
def train_model(model, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()	# Set the model to training mode
        total_loss, total_corrects = 0, 0
        i = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)	# Move data to device
            optimizer.zero_grad()	# Clear existing gradients
            outputs = model(inputs)	# Forward pass
            loss = criterion(outputs, labels)	# Calculate loss
            loss.backward()	# Backward pass

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)	# Gradient clipping

            # Print gradient norms for debugging
            print("\nGradient norms:")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"{name}: {param.grad.norm().item()}")

            optimizer.step()	# Update weights
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            total_corrects += torch.sum(preds == labels.data)
            print(f'Epoch {epoch+1}/{num_epochs}, Minibatch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Accuracy: {torch.sum(preds == labels.data)/inputs.size(0):.4f}')
            i += 1
        epoch_loss = total_loss / len(valid_images)
        epoch_acc = total_corrects.double()/len(valid_images)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
        scheduler.step()	# Adjust the learning rate

# Train the model
train_model(model, criterion, optimizer)

# Save the trained model
torch.save(model, "my_CNN.pth")

