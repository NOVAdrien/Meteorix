# Commandes utiles pour lancer un jupyter notebook à distance à partir de Dalek

## Créer un environnement virtuel sur Dalek pour installer des librairies distantes

### Créer l'environnement
python3 -m venv yolo_env

### Définir l'environnement
source yolo_env/bin/activate

### Installer pip pour ensuite y ajouter les librairies
python -m pip install --upgrade pip

### Installer les librairies nécessaires
pip install ultralytics

-----------------------------------------------------------------------------------

## Aller sur Dalek

### Aller dans le bon dossier
cd ./../../meteorix-main4-25/Adrien/Test/

### Lancer l'environnement virtuel
source ./yolo_env/bin/activate

### Réserver un noeud avec GPU
srun -p az4-n4090 --gres=gpu:dgpu --time=01:00:00 --pty /bin/bash
srun -p az4-n4090 -w az4-n4090-0 --gres=gpu:dgpu:1 --time=01:00:00 --pty /bin/bash

### Installer "notebook" (si pas déjà fait)
./yolo_env/bin/python -m pip install -q notebook

### Lancer le port de connexion
./yolo_env/bin/python -m notebook --no-browser --port=8888
./yolo_env/bin/python -m notebook --no-browser --ip=0.0.0.0 --port=8889

-----------------------------------------------------------------------------------

## Sur un terminal local

### Lancer le port
ssh -N -L 8889:localhost:8889 front.dalek.lip6
ssh -N -L 8889:az4-n4090-0:8889 panguela@front.dalek.lip6
-----------------------------------------------------------------------------------

## Sur mon navigateur

### Se connecter au port de connexion
http://localhost:8889