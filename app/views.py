from django.shortcuts import render
from tensorflow import keras
import tensorflow as tf
import numpy as np
import pathlib
from sklearn.utils import Bunch
import pickle
import pandas as pd

# Create your views here.

def model_output(img) -> tuple:
    # upload img
    # img_path = 'Melanoma.jpg' # измени на путь к своему изображению (вроде работает для png и jpg формата)
    image = keras.utils.load_img(img, target_size=(224,224))
    input_arr = keras.utils.img_to_array(image)
    input_arr = np.array([input_arr])

    # upload model
    model = keras.models.load_model(f'{pathlib.Path(__file__).parent}/model/model_keras_2_1.keras') 
    pred = model.predict(input_arr)
    pred2 = pred.tolist()[0]
    diff = abs(pred2[0]-pred2[1])
    
    range_len = 5.0
    min_el = 8.369982242584229e-05
    
    precis = (diff-min_el)/range_len

    # вывод класса изображения
    classes = {0: 'melanoma', 1: 'nevus'}
    clss = tf.argmax(pred, axis=1)
    return (classes[int(clss)], precis, diff, clss)
    
def load_my_fancy_dataset(df):
    localization_dict_val = {
        'lower extremity': 0, 
        'torso': 1, 
        'upper extremity': 2, 
        'head/neck': 3,
        'unknown': 4, 
        'palms/soles': 5, 
        'oral/genital': 6, 
        'scalp': 7, 
        'ear': 8, 
        'face': 9,
        'back': 10, 
        'trunk': 11, 
        'chest': 12, 
        'abdomen': 13, 
        'genital': 6, 
        'neck': 14, 
        'hand': 15,
        'foot': 16, 
        'acral': 17
    }
    df.localization = df.localization.map(localization_dict_val)
    df.sex = df.sex.map({'male': 0, 'female': 1})
    print(df)
    data = []
    target = []
    for i, row in df.iterrows():
        lst = row.tolist()
        features = lst[:-1]
        label = lst[-1]
        data.append([float(num) for num in features])
        target.append(int(label))
    
    data = np.array(data)
    target = np.array(target)
    return Bunch(data=data, target=target, feature_names=df.columns[:-1])

def model2_output(features):
    with open(f'{pathlib.Path(__file__).parent}/model/model_sklearn_1_0.pkl', 'rb') as f:
        gb = pickle.load(f)
    y_pred_gb = gb.predict(features.data)
    classes_for_2_model = {1: 'melanoma', 0: 'nevus'}
    return classes_for_2_model[int(y_pred_gb[0])]
    
def home(request):
    if 'send_img' in request.POST and request.method == 'POST':
        if request.FILES['image']:
            img = request.FILES['image']
            right_formats = ['.png', '.jpg', '.jpeg']
            if any(i in img.name for i in right_formats):
                out1, precis, diff, target = model_output(img.file)
                text1 = f"Только по изображению модель уверенна на {round(precis*100, 2)}% , что на изображении {out1}"

                age = request.POST.get('age')
                sex = request.POST.get('sex')
                localization = request.POST.get('area')
                print(age, sex, localization)

                if age=='' or sex=='' or localization=='':
                    return render(request, 'home.html', {'error': "Не все поля введены"}) 

                target = 0 if target == 1 else 1 # because of error while model fitting
                
                features = pd.DataFrame([{
                    'advantage': diff,
                    'sex': sex,
                    'age': age,
                    'localization': localization,
                    'target': target
                }])
                print(features)
            
                features = load_my_fancy_dataset(features)
                
                out2 = model2_output(features)
                
                if features.target == out2:
                    if features.target == 1:
                        text2 = "melanoma"
                    else:
                        text2 = "nevus"
                else:
                    text2 = f"1 модель распознала {out1}, 2 модель (на основе данных из первой) определила {out2}"

                text3 = f"1 модель уверенна на {round(precis*100, 2)}%, что на изображении {out2}"
                return render(request, 'home.html', {
                            'prediction': out1, 
                            'text1': text1,
                            'text2': text2,
                            'text3': text3,
                            'error': ''
                            })
            return render(request, 'home.html', {'error': "Формат изображения: '.png', '.jpg'"}) 
    
    if 'send_data' in request.POST and request.method == 'POST':
        age = request.POST.get('age')
        sex = request.POST.get('age')
        localization = request.POST.get('area')

        target = 0 if out1 == 1 else 1 # because of error while model fitting
        features = pd.DataFrame([{
            'advantage': diff,
            'sex': sex,
            'age': age,
            'localization': localization,
            'target': target
        }])
        features.localization = features.localization.map(localization_dict_val)
        features.sex = features.sex.map({'male': 0, 'female': 1})
        features = load_my_fancy_dataset(features)
        
        out2 = model2_output(features)
        
        if features.target == out2:
            if features.target == 1:
                text2 = "melanoma"
            else:
                text2 = "nevus"
        else:
            text2 = f"1 модель распознала {out1}, 2 модель (на основе данных из первой) определила {out2}"

        text3 = f"1 модель уверенна на {round(precis*100, 2)}%, что на изображении {out2}"
        return render(request, 'home.html', {
                    'prediction': out1, 
                    'text1': '',
                    'text2': text2,
                    'text3': text3,
                    'error': ''
                    })

    return render(request, 'home.html', {'error': ''})
