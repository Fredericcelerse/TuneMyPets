# TuneMyPets
This introductory project demonstrates how to build and use fine-tuning/transfer learning approaches for image recognition.

In this example, we show how to construct an AI model based on the Convolutional Neural Network (CNN) architecture for image recognition and enhance its accuracy using fine-tuning and transfer learning approaches.

## Prerequisites

### Anaconda

To execute the code, we will set up a specific environment using Anaconda. To install it, visit [Anaconda Installation](https://docs.anaconda.com/free/anaconda/install/).

### Setup conda environment

First, create the conda environment:
```
conda create -n TuneMyPets python=3.8
```

Then, activate the conda environment:
```
conda activate TuneMyPets
```

Once the environment is properly created, install the necessary Python libraries to execute the code:
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
```
pip install pillow
```

> [!NOTE]
> If you have access to GPUs, then you can install the libraries adapted for this environment:
> ```
> conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
> ```

### Database

The database used in this project, named "Kaggle Cats and Dogs Dataset", is available from the following website: https://www.microsoft.com/en-us/download/details.aspx?id=54765   
Once you have downloaded the database, ensure that you have a directory that contains two main folders labeled "Cat" and "Dog", where images are located. This structure is essential for the CNN script.

## Project architecture

This example consists of two main parts:

***1. Building a CNN model from sratch***   
***2. Optimizing its efficiency***   

Let us see in more details these two aspects

### 1. Building a CNN model from sratch

Within the folder [CNN_from_scratch](CNN_from_scratch), there is a script named [CNN.py](CNN_from_scratch/CNN.py). After verifying the path to the database, you can launch the script by typing:
```
python CNN.py
```
The model will first load the entire database, then filter the images to keep only the valid ones, and then set up and train the CNN model. More details about the setup and how the model is built are explained in the [README.md](CNN_from_scratch/README.md) file, and comments about the code itself are available in the script.

### 2. Optimizing its efficiency

While running this code on a single CPU, the process is very lengthy and the accuracy is quite low, ranging from 50 to 60%, which seems almost random. While optimizing the code is a good idea, doing so without access to some GPUs is a challenging task... So how can we be more efficient in obtaining an AI model that performs the desired tasks without spending a lot of time on optimization and GPUs?

One strategy relies on the use of "Fine-Tuning" and "Transfer Learning", which are two approaches based on the goal of learning a new model by starting from a predefined model trained on previous data. While very similar, these two approaches must be distinguished:

- **Transfer Learning**: Transfer learning involves taking a pre-trained model from one task (usually with a large dataset) and applying it to a related but different task. This can be done by either using the model as a fixed feature extractor and training new top layers for the new task, or by adapting the entire model slightly to the new data, leveraging the pre-learned features to achieve better performance with less data for the new task.   

- **Fine-Tuning**: Fine-tuning is a specific type of transfer learning where the pre-trained model is further adjusted or "fine-tuned" for a new task. This typically involves unfreezing all or some of the layers of the model and continuing the training process on the new data, allowing the model to adjust the pre-learned weights more precisely to specifics of the new task. This is done under the assumption that the initial layers capture universal features that are useful across both tasks, while the later layers are adapted to the specifics of the new task.   

In the folder [CNN_with_fine-tuning](CNN_with_fine-tuning), we provide a documented script called [Fine-Tuning.py](CNN_with_fine-tuning/Fine-Tuning.py). More explanations are available in the corresponding [README.md](CNN_with_fine-tuning/README.md) and the script itself.   

After once again checking the database directory in the script and selecting the model you would like to use (VGG16 or ResNet18), you can launch the script by typing:
```
python Fine-Tuning.py
```

The most important takeaway here is that in only a few refinements, the model is able to learn considerably better compared to our CNN from scratch, showing the strong benefits that can come from using such approaches.

