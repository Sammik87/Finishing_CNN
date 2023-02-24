import io
import streamlit as st
from PIL import Image
import numpy as np
import keras.utils as image
#from keras.applications.efficientnet import preprocess_input, decode_predictions
from keras.models import load_model

#@st.cache(allow_output_mutation=True)
#def load_model():
#    return EfficientNetB0(weights='imagenet')

def preprocess_image(img):
    img = img.resize((75, 55))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)
    x /= 255.
    return x

def load_image():
    uploaded_file = st.file_uploader(label = 'Выберите изображение для распознавания')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

def print_predictions(preds):
    pred_1 = np.round(preds[0][0] * 100, 2)
    pred_2 = np.round((1 - preds[0][0]) * 100, 2)
    st.write('Вероятность, что квартира с отделкой:', pred_1)
    st.write('Вероятность, что квартира без отделки:', pred_2)

# Загрузка модели с диска
model = load_model('model_cnn_fc_1.h5')

st.markdown('<p style="font-size:20px; color:green;">Распознование состояния внутренней отделки квартиры</p', unsafe_allow_html=True)
img = load_image()
result = st.button('Распознать изображение')
if result:
    x = preprocess_image(img)
    preds = model.predict(x)
    st.write('**Результаты распознавания:**')
    print_predictions(preds)