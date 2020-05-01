import numpy as np
from glob import glob
import torch
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.models as models
import torchvision.transforms as transforms
import os
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
import requests
from io import BytesIO
from urllib.request import urlopen
import io
import skimage


def face_detector(img_path1):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    img = skimage.io.imread(img_path1)
    #img = cv2.cvtColor(img_path2, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0




def VGG16_predict(img_path):

    img = Image.open(img_path).convert('RGB')
    im_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    image = im_transform(img)[:3, :, :].unsqueeze(0)
    output = VGG16(image)
    _, pred = torch.max(output, 1)
    pred1 = np.squeeze(pred.numpy())
    pred1 = int(pred1)
    return pred1  # predicted class index

def dog_detector(img_path):
    VGG16 = models.vgg16(pretrained=True)
    dogs = list(range(151,269))
    y = VGG16_predict(img_path)
    return True if y in dogs else False


VGG16 = models.vgg16(pretrained=True)

n_inputs = 4096
n_outputs = 133

out_layer = nn.Linear(n_inputs, n_outputs)
VGG16.classifier[6] = out_layer
model_transfer = VGG16

VGG16 = models.vgg16(pretrained=True)

def load_checkpoint(filepath, model):
    checkpoint = torch.load(filepath)
    model1 = model
    for param in model1.parameters():
        param.requires_grad = False
    model1.classes = checkpoint['classes']
    model1.load_state_dict(checkpoint['state_dict'])
    model1.class_names = checkpoint['class_names']
    return model1


def process_image(img_path):

    image = Image.open(img_path)
    size = 256
    shortest_side = min(image.width, image.height)
    image = image.resize((int((image.width / shortest_side) * size), int((image.height / shortest_side) * size)))
    left_corner = (image.width - 224) / 2
    bottom_corner = (image.height - 224) / 2
    right_corner = (image.width + 224) / 2
    top_corner = (image.height + 224) / 2
    image = image.crop((left_corner, bottom_corner, right_corner, top_corner))
    np_image = np.array(image) / 255
    np_image = (np_image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    np_image1 = np_image.transpose((2, 0, 1))
    return np_image1


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    image = image.transpose((1, 2, 0))
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    std = np.reshape(std, (1, 3))
    image = std * image + mean
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax


def predict1(image_path, model, topk=2):

    model.to('cpu')
    image = process_image(image_path)
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    image_tensor = image_tensor.unsqueeze(0)
    ps = torch.exp(model.forward(image_tensor))
    top_ps, top_labels = ps.topk(topk)
    top_ps = top_ps.detach().numpy().tolist()[0]
    sum_ps = sum(top_ps)
    top_ps1 = list((np.array(top_ps) / sum_ps) * 100)
    top_labels = top_labels.detach().numpy().tolist()[0]
    top_breeds = [model.class_names[top_labels[0]], model.class_names[top_labels[1]]]
    return top_ps1, top_breeds


def plot_image(image_path, model):
    image = process_image(image_path)
    imshow(image)
    ps, breeds = predict(image_path, model)

    plt.figure(figsize=(6, 10))
    plt.subplot(2, 1, 2)
    sns.barplot(x=ps, y=breeds);
    plt.show()

    if ps[0] >= 60.0:
        print('Anyway ! You are a beautiful {} !'.format(breeds[0]))
    elif ps[1] > 40.0:
        print('Anyway ! You are a genetically very sound mixture of a {} and a {}. Very beautiful!'.format(breeds[0],
                                                                                                           breeds[1]))


def run_app(img_path, img_path1):
    ## handle cases for a human face, dog, and neither
    if dog_detector(img_path):
        x = 'dog'

    elif face_detector(img_path1):
        x = 'human'

    else:
        x = 'alien'
    return x