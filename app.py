import gdown
import numpy as np
import streamlit as st
from keras.models import load_model
from keras.utils import load_img, img_to_array


# Loading the Model
gdown.download(id='1zo1ZTyuKg-NvB7J3aDN6uEjQhm3sP1Cx', output='model.h5')
model = load_model('model.h5')

# Name of Classes
CLASS_NAMES = []
enc_file = open('encoder_data.txt', 'r')
for cls in enc_file:
    CLASS_NAMES.append(cls)
enc_file.close()

# Setting the App
st.title('Plant Disease Prediction')
st.markdown('Upload an image of the plant')
test_img = st.file_uploader('Choose an image...', type='jpg')
submit = st.button('Predict')

if submit:
    if test_img is not None:
        # Convert image into batch
        test_img = load_img(test_img, target_size=(256, 256))
        test_img = img_to_array(test_img)
        test_img = np.expand_dims(test_img, axis=0)

        prediction = model.predict(test_img)

        st.title(f'Plant Disease is {CLASS_NAMES[np.argmax(prediction)]}')
