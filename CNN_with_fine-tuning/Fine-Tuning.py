#!/usr/bin/env python

# Import the necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset, random_split
from PIL import Image

# Define transformations for preprocessing the images
data_transforms = transforms.Compose([
	  transforms.RandomResizedCrop(224),	# Randomly crop the images to 224x224
	  transforms.RandomHorizontalFlip(),	# Randomly flip the images horizontally
	  transforms.ToTensor(),		# Convert images to tensor
	  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])	# Normalize the images
])

# Set the directory where the data is located
data_dir = '/Users/celerse/Images/Ex2/kagglecatsanddogs_5340/PetImages'
model_type = "Resnet"

# Function to check if an image file is corrupt
def check_image(path):
	try:
		img = Image.open(path)	# Attempt to open an image
		img.verify()		# Verify that it is a correct image
		img.close()		# Close the file
		return True
	except:
		return False	# Return False if the image is corrupt

# Load the dataset from the specified directory and apply transformations
dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
print(f"Nombre total d'images chargées : {len(dataset)}")

# Filter out corrupt images
valid_indices = [i for i in range(len(dataset)) if check_image(dataset.imgs[i][0])]
valid_images = Subset(dataset, valid_indices)
print(f"Nombre d'image valide: {len(valid_images)}")

# Split the dataset into training and validation sets
train_size = int(0.8 * len(valid_images))
val_size = len(valid_images) - train_size
train_dataset, val_dataset = random_split(valid_images, [train_size, val_size])

# DataLoader objects to load data in batches
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load a pretrained (VGG-16 or ResNet-18) model and freeze all layers before the final fully connected layer
if model_type == "Resnet":
        model = models.resnet18(pretrained=True)
else:
        model = models.vgg16(pretrained=True)
for param in model.parameters():
	param.requires_grad = False	# Freeze the parameters so as not to backprop through them

# Replace the final layer with a new layer that has only 2 output nodes, for our binary classification task
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)	# 2 because of we only have cat and dog here

# Set the computation device based on GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001) # Only optimize the parameters of the final layer

# Function to train the model
def train_model(model, criterion, optimizer, num_epochs=10):
	for epoch in range(num_epochs):
		model.train()	# Set the model to training mode
		total_loss, total_corrects = 0, 0
		for inputs, labels in train_loader:
			inputs, labels = inputs.to(device), labels.to(device)	# Transfer to device
			optimizer.zero_grad()	# Zero the gradients
			outputs = model(inputs)	# Forward pass
			loss = criterion(outputs, labels)	# Compute loss
			loss.backward()	# Backward pass
			optimizer.step()	# Update weights
			total_loss += loss.item() * inputs.size(0)
			_, preds = torch.max(outputs, 1)
			total_corrects += torch.sum(preds == labels.data)
		epoch_loss = total_loss / len(valid_images)
		epoch_acc = total_corrects.double()/len(valid_images)
		print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

# Train the model
train_model(model, criterion, optimizer)

# Save the trained model
if model_type == "Resnet":
        torch.save(model, "my_new_resnet18.pth")
else:
        torch.save(model, "my_new_vgg16.pth")
