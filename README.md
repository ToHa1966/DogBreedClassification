# Creating a Dog Breed Classifier
This project is part of the Udacity Deep Learning Nanodegree program.

### Project Status
Completed

## Project Intro
The purpose of this project is to acquire skills in developing an AI application

## Project Description
In this project an image classifier is built to recognize dog breeds. The image dataset is loaded and preprocessed, 
the classifier is built from scratch or using an existing model and tansfer learning, the classifier is trained 
to predict image content and the accuracy is measured. This application can be trained on any set of labeled images.
Furthermore a human and dog detector was created using OpenCV and the VGG16 model.


## Methods Used
- Loading and transforming data
- Loading a pretrained model
- Building a classifier for the model
- Train the classifier
- Test the model
- Loading/saving checkpoints
- Preprocessing images
- Making predictions
- Build detectors for humans and dogs in images 

## Technologies Used
- Jupyter Notebook
- Python
- Numpy
- Pandas
- Matplotlib
- Seaborn
- Torch
- Torchvision
- PIL
- Cuda
- Glob
- OpenCV
- os
- tqdm
- requests
- skimage
- Flask

## Flask App

A Flask App for the dog breed classifier can be found in the DogBreed directory.
You can use it for pictures that have a http file path but you can't use both internet sources and local sources.
Furthermore the file size has to be above 300px. Optimizing these issues was not in scope of this project.
