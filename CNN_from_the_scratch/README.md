# CNN from the scratch

Here is explained how the code was built and works, but also how we could enhance it in future works.

## Code structuration

### 1. Data Preprocessing

**Transformations**: The code applies multiple transformations to prepare and augment the dataset:
- RandomResizedCrop to 224x224 pixels to standardize input size.
- RandomHorizontalFlip and RandomRotation (15 degrees) to augment the data and improve generalization.
- ColorJitter adjusts brightness, contrast, saturation, and hue to enhance image diversity.
- ToTensor converts images to PyTorch tensors.
- Normalize standardizes pixel values using predefined mean and standard deviation values for each channel.

This part is essential when we deal with dataset of Images, as we need to standardize all the Images together to allow the model to treat all the Images. Furthermore, it is important to normalize each data as it helps in the learning process. 

**Image Verification**: A custom function check_image is implemented to ensure that all loaded images are valid, preventing runtime errors due to corrupted files.

### 2. Dataset Loading and Splitting

- Utilizes PyTorch's ImageFolder to load images from a specified directory (data_dir).
- Filters out invalid images and splits the dataset into 80% training and 20% validation subsets to ensure the model is evaluated on unseen data.

### 3. Model Definition

**Architecture**: A custom convolutional neural network (CNN) class CNN:
- Two convolutional layers with batch normalization and max pooling to extract and downsample features.
- Two fully connected layers with dropout in between to prevent overfitting and make predictions.

**Weight Initialization**: Uses kaiming_normal_ initialization for convolutional and linear layers to prevent vanishing or exploding gradients during training.

### 4. Training Setup

**Optimization**: Employs SGD optimizer with a learning rate of 0.001 and momentum of 0.9, chosen to balance the speed and stability of convergence.

**Loss Function**: Uses CrossEntropyLoss for binary classification, suitable for categorical outcomes.

**Scheduler**: Implements a StepLR learning rate scheduler with a step size of 30 and gamma of 0.1 to reduce the learning rate by a factor of 10 every 30 epochs, helping to fine-tune the model as it converges.

### 5. Training Process

- Iterates through the training data in batches, computing losses and updating the model parameters.
- Gradient clipping is used to control the explosion of gradients, ensuring stable updates.
- Monitors gradient norms and prints training progress, including loss and accuracy for each minibatch, allowing for real-time monitoring of training performance.

### 6. Saving the Model

Saves the trained model to disk, allowing for later use in applications or further fine-tuning.

## Choosing hyperparameters

**Learning Rate and Momentum**: Selected to provide a good compromise between fast convergence and training stability.

**Batch Size**: Set to 64 to make effective use of memory resources while allowing for sufficient gradient estimation accuracy.

**Epochs and Scheduler**: Configured to adapt the learning rate during training, aiming for deep refinement in later stages of model fitting.

## How could we enhance the model in the future without the use of Fine-Tuning and Transfer Learning ?


