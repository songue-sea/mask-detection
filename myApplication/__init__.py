from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_sqlalchemy import SQLAlchemy


jwt = JWTManager()

app = Flask(__name__)
CORS(app)
app.config.from_object('config')
db = SQLAlchemy()
db.init_app(app)
jwt.init_app(app)
app.config['JWT_SECRET_KEY'] = 'Super_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

from . import models

