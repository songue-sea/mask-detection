import cv2
import keras.utils as image
import matplotlib.pyplot as plt
# utils.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from myApplication import app  #  cdb est bien ton module principal

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_image(img_path, show=False):
    """Charge et prépare une image."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)  # (hauteur, largeur, canaux)
    img_tensor = np.expand_dims(img_tensor, axis=0)  # (1, hauteur, largeur, canaux)
    if show:
        plt.imshow(img_tensor[0].astype('uint8'))
        plt.axis('off')
        plt.show()
    return img_tensor

def generate_adversarial_image(model, img_path, target_label, epsilon=0.5, save_path="upload/adverses/adverse.jpg"):
    """
    Génère une image adversariale à partir de l'image d'entrée.
    Paramètres :
      - model : le modèle à attaquer.
      - img_path : chemin vers l'image.
      - target_label : l'indice de la classe cible (ex. 0 pour bird, 1 pour cat, 2 pour dog).
      - epsilon : facteur de perturbation.
      - save_path : chemin pour sauvegarder l'image générée.
    """
    # Charger l'image
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    # On ne normalise pas pour rester dans l'échelle [0,255]
    img_tensor = tf.convert_to_tensor(img_tensor, dtype=tf.float32)

    # Convertir le label cible en one-hot encoding (assumant que le modèle a été entraîné avec categorical_crossentropy)
    target = to_categorical(target_label, num_classes=model.output.shape[-1])
    target = tf.convert_to_tensor([target], dtype=tf.float32)

    # Calculer les gradients par rapport à l'image d'entrée
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        prediction = model(img_tensor)
        loss = tf.keras.losses.categorical_crossentropy(target, prediction)

    gradient = tape.gradient(loss, img_tensor)
    signed_grad = tf.sign(gradient)

    # Générer l'image adversariale
    adversarial_img = img_tensor + epsilon * signed_grad
    adversarial_img = tf.clip_by_value(adversarial_img, 0, 255)

    # Sauvegarder l'image générée
    adversarial_img_np = adversarial_img.numpy()[0].astype('uint8')
    plt.imsave(save_path, adversarial_img_np)

    return adversarial_img_np, img_tensor.numpy()[0]


def generate_adversarial_image2(model, img_path, target_label, epsilon=0.5, save_path="upload/adverses/adverse.jpg"):
    """
    Génère une image adversariale à partir de l'image d'entrée.
    Paramètres :
      - model : le modèle à attaquer.
      - img_path : chemin vers l'image.
      - target_label : l'indice de la classe cible (ex. 0 pour mask, 1 pour no_mask).
      - epsilon : facteur de perturbation.
      - save_path : chemin pour sauvegarder l'image générée.
    """
    # Charger l'image
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = tf.convert_to_tensor(img_tensor, dtype=tf.float32)

    # Convertir le label cible en one-hot encoding et ajouter une dimension
    target = to_categorical(target_label, num_classes=model.output.shape[-1])
    target = np.expand_dims(target, axis=0)  # S'assurer que la forme est (1, num_classes)
    target = tf.convert_to_tensor(target, dtype=tf.float32)

    # Calculer les gradients par rapport à l'image d'entrée
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        prediction = model(img_tensor)
        loss = tf.keras.losses.categorical_crossentropy(target, prediction, from_logits=False)

    gradient = tape.gradient(loss, img_tensor)
    signed_grad = tf.sign(gradient)

    # Générer l'image adversariale
    adversarial_img = img_tensor + epsilon * signed_grad
    adversarial_img = tf.clip_by_value(adversarial_img, 0, 255)

    # Sauvegarder l'image générée
    adversarial_img_np = adversarial_img.numpy()[0].astype('uint8')
    plt.imsave(save_path, adversarial_img_np)

    return adversarial_img_np, img_tensor.numpy()[0]

def apply_advanced_defenses(image_tensor, noise_level=0.05, kernel_size=3, bit_depth=3):
    """
    Applique des défenses avancées : bruit, lissage gaussien, filtrage médian, feature squeezing et débruitage non-local.
    """
    # Bruit aléatoire
    noise = np.random.normal(loc=0.0, scale=noise_level * 255, size=image_tensor.shape)
    noisy_image = image_tensor + noise
    noisy_image = np.clip(noisy_image, 0, 255)

    # Lissage gaussien
    smoothed_image = cv2.GaussianBlur(noisy_image[0], (kernel_size, kernel_size), 0)

    # Filtrage médian
    median_filtered_image = cv2.medianBlur(smoothed_image.astype('uint8'), kernel_size)

    # Feature squeezing
    max_val = 2 ** bit_depth - 1
    squeezed_image = np.round(median_filtered_image / 255.0 * max_val) / max_val * 255
    squeezed_image = np.clip(squeezed_image, 0, 255)

    # Débruitage non-local
    denoised_image = cv2.fastNlMeansDenoisingColored(squeezed_image.astype('uint8'), None, 10, 10, 7, 21)

    return np.expand_dims(denoised_image, axis=0)
