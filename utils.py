import numpy as np
import cv2
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

def affiche_regions(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Utiliser MOPS pour détecter les régions stables maximales
    mops = cv2.MSER_create()
    regions, _ = mops.detectRegions(gray)

    # Dessiner les régions stables maximales en jaune
    for region in regions:
        bbox = cv2.boundingRect(region)
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 255), 2)  # Rectangle jaune
    return image

def affiche_regions2(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Utiliser MOPS pour détecter les régions stables maximales
    mops = cv2.MSER_create()
    regions, _ = mops.detectRegions(gray)

    # Dessiner les régions stables maximales en jaune
    for region in regions:
        bbox = cv2.boundingRect(region)
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 255), 2)  # Rectangle jaune
        break
    return image

def harris_corners(image) :
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Utiliser Harris pour détecter les coins
    harris_corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    harris_corners = cv2.dilate(harris_corners, None)

    # Définir un seuil pour les coins Harris
    image[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]
    return image

def sift_corners(image) :
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Utiliser SIFT pour extraire les keypoints et descripteurs
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Dessiner les keypoints SIFT sur l'image
    cv2.drawKeypoints(image, keypoints, image, color=(0, 255, 0) ,  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return image


# Fonction pour charger et prétraiter une image
def load_and_preprocess_image(img_path):
    # Charger l'image
    img = image.load_img(img_path, color_mode='grayscale', target_size=(28, 28))
    
    # Convertir l'image en tableau numpy
    img_array = image.img_to_array(img)
    
    # Normaliser les valeurs de l'image
    img_array = img_array / 255.0
    
    # Ajouter une dimension pour correspondre au batch_size attendu par le modèle
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array