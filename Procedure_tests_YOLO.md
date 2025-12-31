# Procédure tests YOLO

1. Se placer dans le repo darknet/.

2. Nettoyer tous les fichiers objets et exécutable avec:
- make clean

3. Lancer la compilation avec:
- make all

4. Lancer l'exécutable avec:
- ./darknet yolo test cfg/yolov1.cfg yolov1.weights data/dog.jpg
-> Remplacer cfg/yolov1.cfg et yolov1.weights par le bon modèle YOLO
-> Remplacer data/dog.jpg par l'image souhaitée

5. Récupérer la sortie dans le fichier 'predictions.jpg'.