# TuneMyPets
This is a small introductive projet that aims at showing how to build and using fine-tuning/transfer learning approaches for image recognition.

In this example, we show how to build a AI model based on the Convolution Neural Network (CNN) architecture for image recognition, and how we can enhance its accuracy by using fine-tuning and transfer learning approaches.

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
> If you GPUs access, then you can install the libraries adapated for this environment:
> ```
> conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
> ```

### Database

The database used in this project, named "Kaggle Cats and Dogs Dataset", comes from the following website: https://www.microsoft.com/en-us/download/details.aspx?id=54765   
Once the database, ensure that you have a directory that provides two main folders labeled as "Cat" and "Dog", and where images are localized. This structure of database will be essential for the CNN script. 

## Project architecture

This example is made of tow main parts:

***1. Building a CNN model from the sratch***   
***2. How can we optimize its efficiency***   

Let us see in more details these two aspects

### 1. Building a CNN model from the sratch

Within the folder [CNN_from_the_scratch](CNN_from_the_scratch), there is a script called [CNN.py](CNN_from_the_scratch/CNN.py). After carefully checked the path to the database, you can launch the script by typing:
```
python CNN.py
```
The model will first load the full database, then filter the images in order to keep only the valid ones, and then setup and train the CNN model. More details about the setup and how the model is built is explained in the [README.md](CNN_from_the_scratch/README.md) file and commentaries about the code itself is available in the script. 

### 2. How can we optimize its efficiency

Whereas we try to run this code on a single CPU, the process is very long and the accuracy is very low, ranging from 50 to 60% and seems to correspond to randomicity. Whereas optimizing the code seems to be a good idea, doing it without having access to some GPU is a very hard task ... So how can we be more efficient to obtain an AI model that perform the desired tasks without having to pay lot of times in optimization and GPUs ?   

One strategy relies on the use of "Fine-Tuning" and "Transfer Learning", which are two approaches relying on the aim to learn a new model by starting from a predined model trained before on previous data. While very similar, these two approaches have to be distinguished:   

- $${\color{red}{Transfer Learning}}: Transfer learning involves taking a pre-trained model from one task (usually with a large dataset) and applying it to a related but different task. This can be done by either using the model as a fixed feature extractor and training new top layers for the new task, or by adapting the entire model slightly to the new data, leveraging the pre-learned features to achieve better performance with less data for the new task.   

- **Fine-Tuning**: Fine-tuning is a specific type of transfer learning where the pre-trained model is further adjusted or "fine-tuned" for a new task. This typically involves unfreezing all or some of the layers of the model and continuing the training process on the new data, allowing the model to adjust the pre-learned weights more precisely to specifics of the new task. This is done under the assumption that the initial layers capture universal features that are useful across both tasks, while the later layers are adapted to the specifics of the new task.   

In the folder [Fine-Tuning](Fine-Tuning), we provide a documented script called [CNN_with_fine-tuning.py](Fine-Tuning/CNN_with_fine-tuning.py). More explanations are available in the corresponding [README.md](CNN_with_fine-tuning/README.md) and the script itself.   

After once again check the database directory in the script and select the model ou would like to use (vgg16 or ResNet18), you can launch the script by typing:
```
python CNN_with_fine-tuning.py
```

The most important to retain here is that in only few refinements, the model is able to learn considerably better compared to our CNN from the sractch, showing the strong benefits that could come from using such approaches. 

