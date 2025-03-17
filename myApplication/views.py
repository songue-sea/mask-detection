"""The views content routes or API are defined here"""
import os
from datetime import timedelta

import numpy as np
from flask import request, jsonify, send_file
from flask_jwt_extended import create_access_token, create_refresh_token, jwt_required, get_jwt_identity
from werkzeug.security import check_password_hash

from myApplication import app, db
from myApplication.models import User
from utils import allowed_file, load_image, generate_adversarial_image2, apply_advanced_defenses


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'

# Registration
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data.get('username') or not data.get('email') or not data.get('password'):
        return jsonify({'error': 'Tous les champs sont obligatoire'}), 400

    if User.query.filter_by(username=data['username']).first():
        return jsonify({"error": "Nom d'utilisateur déja pris"}), 400

    if User.query.filter_by(email=data['email']).first():
        return jsonify({"error": "email  déja utilisé"}), 400

    new_user = User(username=data['username'], email=data['email'])
    new_user.set_password(data['password'])
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "Inscription réussie !"}), 201

# Login API
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    user = User.query.filter_by(username=data['username']).first()
    if user and check_password_hash(user.password, data['password']):
        access_token = create_access_token(identity=user.username, expires_delta=timedelta(hours=1))
        refresh_token = create_refresh_token(identity=user.username)
        return jsonify(
            {"access_token": access_token,
             "refresh_token": refresh_token,
             "user": {"email": user.email,
                      "username": user.username
                      }
             }
        ), 200
    return jsonify({"error": "Identifiants invalides"}), 401

# protected API for testing
@app.route('/protected', methods=['GET'])
@jwt_required()
def protected():
    current_user = get_jwt_identity()
    return jsonify({"message": f"Bienvenue {current_user}, vous êtes authentifié !"}), 200

@app.route('/refresh', methods=['POST'])
@jwt_required()
def refresh():
    identiy = get_jwt_identity()
    new_access_token = create_access_token(identity=identiy, expires_delta=timedelta(hours=1))
    return jsonify({"access_token": new_access_token})


@app.route('/predict', methods=['POST'])
@jwt_required()
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No image sent "}), 400

    file = request.files["file"]
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    try:
        file_path = os.path.join('upload', file.filename)
        print(file_path)
        file.save(file_path)
        image = load_image(file_path)
        models = app.config['MODEL']
        prediction = models.predict(image)

        classes = ["with-mask", "without-mask"]
        predicted_classes = classes[prediction.argmax()]
        print(predicted_classes)
        return jsonify({"prediction": predicted_classes, "confidence": float(np.max(prediction))}), 200
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

@app.route('/generate-adversarial',methods=['POST'])
@jwt_required()
def generate_adversarial():
    if "file" not in request.files or "epsilon" not in request.form or "label" not in request.form:
        return jsonify({"error": "Image , epsilon and label required"}), 400

    file = request.files["file"]
    try:
        epsilon = float(request.form["epsilon"])
        target_label = int(request.form["label"])
        print(request.form["label"])
    except ValueError:
        return jsonify({"error": "Erreur de valeur"}), 400

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400
    if target_label < 0 or target_label >= app.config["MODEL"].output.shape[1]:
        return jsonify(
            {"error": f"Le label doit être compris entre 0 et {app.config['MODEL'].output.shape[-1] - 1}."}), 400
    try:
        file_path = os.path.join('./upload', file.filename)
        file.save(file_path)
        model = app.config['MODEL']
        adv_img, original_img = generate_adversarial_image2(model, file_path, target_label=target_label, epsilon=epsilon)
        return send_file("../upload/adverses/adverse.jpg", as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict-robust', methods=['POST'])
@jwt_required()
def predict_robust():
    if "file" not in request.files:
        return jsonify({"error": "No image sent "}), 400
    file = request.files["file"]
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400
    try:
        file_path = os.path.join('averse', file.filename)
        file.save(file_path)
        image = load_image(file_path)

        defended_image = apply_advanced_defenses(image)
        model = app.config['MODEL']
        prediction = model.predict(defended_image)
        classes = ["with-mask", "without-mask"]
        predicted_classes = classes[prediction.argmax()]
        print(predicted_classes)
        return jsonify({"prediction": predicted_classes, "confidence": float(np.max(prediction))}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run()

