# CNN from the scratch

Here is explained how the code was built and works using fine-tuning methods with vgg-16 and ResNet-18.

## Code structuration

### 1. Data Preprocessing

**Transformations**: Uses a series of transformations to preprocess and augment the dataset, preparing it effectively for training a deep learning model:
- RandomResizedCrop to uniformly resize the images to 224x224 pixels.
- RandomHorizontalFlip to augment the data by introducing horizontal flips, increasing the dataset's variability.
- ToTensor and Normalize for converting images to PyTorch tensors and normalizing their pixel values, aligning with common preprocessing for pretrained models.

### 2. Data Loading and Validation

- Utilizes the ImageFolder loader to organize images from a structured directory, with an additional check to ensure all images are valid using a custom check_image function.
- Splits the dataset into training (80%) and validation (20%) sets, ensuring there is a set of unseen data to evaluate the model's performance.

### 3. Model Setup for Fine-Tuning

**Pretrained Model**: Loads a pretrained (ResNet-18 or vgg-16) model, widely used for image classification tasks due to its efficiency and effectiveness.

**Freezing Parameters**: Freezes all the pretrained layers' weights to retain the learned features and only tunes the new layers to the specific task, reducing training time and computational expense.

**Replacing the Fully Connected Layer**: Modifies the last fully connected layer to output two classes instead of the original 1000 classes used in ImageNet, tailoring the model to the binary classification task.

### 4. Training Environment Setup

**Device Configuration**: Configures the script to use GPU acceleration if available, enhancing training speed.

**Optimizer and Loss Function**: Employs the Adam optimizer for the newly added final layer and CrossEntropyLoss, common choices for classification tasks.

### 5. Training Process

- Executes the training over a specified number of epochs, printing out loss and accuracy metrics for each epoch to monitor progress and performance.
- Implements gradient clipping against the exploding gradient problem, a common issue when fine-tuning deep networks.

### 6. Model Saving

After training, the model is saved to the local file system, making it easy to deploy or further fine-tune later.

## Fine-Tuning Mechanism Explained

**Parameter Freezing**: The script freezes the parameters (weights) of all layers except the last fully connected layer. This approach leverages the learned features from the massive ImageNet dataset, on which the ResNet-18 was originally trained, and adapts it to a new, more specific task by only training the final classification layer.

**Focused Training**: By training only the last layer, the script effectively fine-tunes the pretrained network to classify new categories (cats vs. dogs). This method is computationally efficient and typically yields good results quickly when transitioning from general to specific tasks.

**Optimizer Isolation**: The Adam optimizer is specifically used to update the weights of the newly replaced fully connected layer, ensuring that only the relevant parts of the network are modified. This targeted optimization helps in quickly adapting the network’s output to the expected binary labels while preserving the integrity of the pre-learned features in earlier layers.
