# YOLOv1 — tests préliminaires

Ce dossier regroupe les premiers essais réalisés avec Yolov1 via Darknet au début du projet.

## Rôle de cette partie
Cette section conserve une trace des expérimentations exploratoires menées avant la mise en place du pipeline principal basé sur Yolo26 et DETR.

Elle permet de :
- documenter les premières tentatives de détection sur images de météores
- garder les notebooks, données et résultats associés à Yolov1
- distinguer clairement les travaux historiques du pipeline actif du projet

## Contenu attendu
Le dossier peut contenir :
- `notebooks/` : notebooks de test, d'inférence ou de validation rapide
- `data/` : images utilisées pour les essais préliminaires
- `results/` : sorties générées par Darknet, visualisations et essais de détection

## Statut
Cette partie doit être considérée comme une archive de travail exploratoire.

Elle n'est pas le coeur du projet actuel, mais elle reste utile pour :
- comprendre l'évolution méthodologique du projet
- retrouver une procédure ancienne
- comparer les premières approches avec les modèles plus récents

## Notebook principal
Le notebook `test_yolov1_darknet.ipynb` propose une version de la procédure de test sur une image :
- paramètres regroupés au même endroit
- vérifications de fichiers plus explicites
- patch optionnel pour afficher la probabilité
- séparation claire entre préparation, inférence et affichage

## Remarque
Les expériences principales du projet sont désormais organisées séparément autour de Yolo26 et DETR.  
La partie Yolov1 est conservée comme étape préliminaire de l'historique du projet.
