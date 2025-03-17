"""Contains configurations file"""
from keras.models import load_model
import os
SECRET_KEY = os.urandom(32)
SQLALCHEMY_DATABASE_URI = "sqlite:///database.db"
SQLALCHEMY_TRACK_MODIFICATIONS = False
JWT_SECRET_KEY = os.urandom(32)
UPLOAD_FOLDER = './uploads'
AVERSE_FOLDER = "./averse"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MODEL_PATH = "./mask_no_mask_version1.h5"
MODEL = load_model(MODEL_PATH)