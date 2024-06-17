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

### Database

The database used in this project, named "Kaggle Cats and Dogs Dataset", comes from the following website: https://www.microsoft.com/en-us/download/details.aspx?id=54765

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
The model will first load the full database, then filter the images in order to keep only the valid ones, and then setup and train the CNN model. More details about the setup and how the model is built is explained in the [README.md](CNN_from_the_scratch/README.md) file. 

### 2. How can we optimize its efficiency



