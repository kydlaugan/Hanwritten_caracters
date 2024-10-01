import streamlit as st
import pandas as pd
import numpy as np
import cv2
from utils import *
from tensorflow.keras.models import load_model
from io import StringIO
import tensorflow as tf



cnn_model = load_model('4_best_model.h5.keras')
model = load_model('modele.h5.keras')

        
st.title('Handwriting recognition')
container = st.container(border=True)
tab1, tab2 , tab3 = container.tabs(["features extraction", "SVM" , "CNN"])
tab1.subheader('1- Detection des régions stables avec MOPS')
tab1.write('L`extraction des régions stables avec MOPS (Maximally Stable Extremal Regions) est une technique de détection de régions dans une image.')
tab1.write(" Il detecte  les MSERs qui sont des régions dans une image qui restent stables sur une large gamme de seuils de segmentation. En d'autres termes, ces régions ne changent pas beaucoup même si on modifie le seuil utilisé pour les segmenter.")

uploaded_files = tab1.file_uploader("importer une image", accept_multiple_files=True)
row1 = tab1.columns(3)


for uploaded_file in uploaded_files:
    # Lire le fichier téléchargé en tant que tableau numpy
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_1 = cv2.imdecode(file_bytes, 1)
    image_2 = cv2.imdecode(file_bytes, 1)
    image_region = affiche_regions(image_1)  
    image_region2 = affiche_regions2(image_2)  
    tile = row1[0].container(height=140)
    tile1 = row1[1].container(height=140)
    tile2 = row1[2].container(height=140)
    tile.image(image, channels="BGR", caption="Image originale")
    tile1.image(image_region, channels="BGR", caption="Régions stables détectées")
    tile2.image(image_region2, channels="BGR", caption="Régions stables détectées")

tab1.subheader('2- Extraction des caractéristiques avec Harris')

uploaded_files_1 = tab1.file_uploader("importer une image")
if uploaded_files_1:
    # Lire le fichier téléchargé en tant que tableau numpy
    file_bytes = np.asarray(bytearray(uploaded_files_1.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    #image = cv2.resize(image, (177, 177))
    image_harris = harris_corners(image)  
    tab1.image(image_harris, channels="BGR", caption="Image  avec le coins de harris" )


tab1.subheader('3- Extraction des caractéristiques avec SIFT')

uploaded_files_2 = tab1.file_uploader("une image")
if uploaded_files_2:
    # Lire le fichier téléchargé en tant que tableau numpy
    file_bytes = np.asarray(bytearray(uploaded_files_2.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    #image = cv2.resize(image, (177, 177))
    image_harris = sift_corners(image)  
    tab1.image(image_harris, channels="BGR", caption="Image  avec le coins de SIFT" )



tab2.subheader('1- Extraction des caracteristiques sur le jeu de données MNIST avant l`entrainement')
tab2.image("img1.png")
tab2.image("img2.png")


uploaded_file_3 = tab2.file_uploader("Choose a file")
if uploaded_file_3 is not None:

    # Charger et prétraiter l'image
    preprocessed_image = load_and_preprocess_image(uploaded_file_3)

    # Faire une prédiction
    prediction = cnn_model.predict(preprocessed_image)

    # Convertir la prédiction en classe
    predicted_class = np.argmax(prediction, axis=1)
    row2 = tab2.columns(2)
    tile3 = row2[0].container(height=300)
    tile4 = row2[1].container(height=300)
    tile3.image(uploaded_file_3 ,  channels="BGR")
    tile4.header(predicted_class[0])


tab3.subheader('Mise sur pied d`un réseau  CNN  sur le jeu de données MNIST pour la prediction des caractères manuscrits')

uploaded_file_4 = tab3.file_uploader("ouvrir un fichier")
if uploaded_file_4 is not None:
        # Lire le fichier téléchargé en tant que tableau numpy
    file_bytes = np.asarray(bytearray(uploaded_file_4.read()), dtype=np.uint8)
    image_cnn = cv2.imdecode(file_bytes, 1)

    img = cv2.cvtColor(image_cnn, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (400,440))

    img_copy = cv2.GaussianBlur(image_cnn, (7,7), 0)
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

    img_final = cv2.resize(img_thresh, (28,28))
    img_final =np.reshape(img_final, (1,28,28,1))

    word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}
    img_pred = word_dict[np.argmax(model.predict(img_final))]

    row3 = tab3.columns(2)
    tile5 = row3[0].container(height=300)
    tile6 = row3[1].container(height=300)
    tile5.image(uploaded_file_4 ,  channels="BGR")
    tile6.header(img_pred)



