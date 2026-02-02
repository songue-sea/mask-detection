Voici la partie back-end de l'application. Cette partie recouvre les API de l'application. 
L'application a besoin de la partie front-end pour bien fonctionner (voir le dépôt https://github.com/songue-sea/mask-detection-front-end) 
Pour bien installer l'application flask voici les étapes à suivre (sous pycharm est l'ideal ) :
1. Créer un répertoire vide qui doit accueillir les fichiers du projet
2. cloner le projet (avec git un nouveau dossier "mask-detection" est créé conteant tous les fichiers)
3. Se déplacer dans le répertoire "mask-detection" et créer un nouveau environnement python pour faciliter l'installation des dépendances du projet
4. Activer l'environnement , et installer les dépendances spécifiées dans le fichier "requirements.txt"
5. Si vous êtes dans pycharm , s'assurer que toutes les dépendances sont installées
6. NB: Durant l'installation il se peut que tensorflow ne soit pas prise en compte  ou le package "cv2" , veuillez l'installer manuellement avec pip ou dans l'interpréteur de python sous pycharm
7. (La version 2.15.1 est requise)
8. Pour démmarer l'application flask sous windows voici les étapes :
9. $env:FLASK_APP="run"
10. flask run


Détails des fonctionnalités :

-Toutes les endpoints de l'application sont regroupés dans views.py (l'authentification , prédiction ...) et peuvent être testés manuellement (par ex postman ou curl ou autre outil client)
-Le modèle est sauvegardé dans le répertoire et peut aussi être testé 

NB: En cas d'erreur ou de soucis , veuillez me contacter via l'adresse email "songue.sea@gmail.com"

