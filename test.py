from matplotlib import pyplot as plt
from tensorflow.python.keras.models import load_model

from utils import generate_adversarial_image2
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Rescaling

# Charger le modèle avec la couche Rescaling
model = load_model("./mask_no_mask_version1.h5", custom_objects={"Rescaling": Rescaling})

adv_img, original_img = generate_adversarial_image2(model, "upload/mask.jpg", target_label=0,epsilon=50)

# Afficher les images
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(original_img.astype('uint8'))
plt.title("Image Originale")

plt.subplot(1,2,2)
plt.imshow(adv_img)
plt.title("Image Adversariale")

plt.show()
