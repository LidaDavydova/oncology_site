from django.shortcuts import render
import os
import PIL
from tensorflow import keras
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np

# Create your views here.

def model_output(img):
    # upload img
    # img_path = 'Melanoma.jpg' # измени на путь к своему изображению (вроде работает для png и jpg формата)
    image = keras.utils.load_img(img, target_size=(224,224))
    input_arr = keras.utils.img_to_array(image)
    input_arr = np.array([input_arr])

    # upload model
    model = keras.models.load_model('../../models/kerasSimple.h5') # измени на свой путь к модели .h5
    pred = model.predict(input_arr)

    # вывод класса изображения
    classes = {0: 'melanoma', 1: 'nevus'}
    clss = tf.argmax(pred, axis=1)
    return classes[int(clss)]

def home(request):
    if request.method == 'POST' and request.FILES['image']:
        img = request.FILES['image']
        right_formats = ['.png', '.jpg']
        if any(i in img.name for i in right_formats):
            prediction = model_output(img.file)
            return render(request, 'home.html', {'prediction': prediction, 'error': ''})
        return render(request, 'home.html', {'error': "Формат изображения: '.png', '.jpg'"}) 
    return render(request, 'home.html', {'error': ''})