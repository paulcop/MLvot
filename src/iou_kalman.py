import torchvision.models as models
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as transforms
import torch
import cv2
import os
import numpy as np
import csv
from scipy.optimize import linear_sum_assignment
from KalmanFilter import KalmanFilter


def lire_detections(chemin_fichier):
    resultat_detections = []
    with open(chemin_fichier, 'r') as fichier:
        for ligne in fichier:
            elements = ligne.strip().split(',')
            num_image = int(elements[0])
            id_obj = int(elements[1])
            x_gauche = float(elements[2])
            y_haut = float(elements[3])
            largeur = float(elements[4])
            hauteur = float(elements[5])
            confiance = float(elements[6])
            position_x = float(elements[7])
            position_y = float(elements[8])
            position_z = float(elements[9])
            resultat_detections.append({
                'frame': num_image,
                'id': id_obj,
                'bb_left': x_gauche,
                'bb_top': y_haut,
                'bb_width': largeur,
                'bb_height': hauteur,
                'conf': confiance,
                'pos_x': position_x,
                'pos_y': position_y,
                'pos_z': position_z
            })
    return resultat_detections

def calculer_intersection_sur_union(boite1, boite2):
    """
    Calcule l'Intersection sur Union (IoU) entre deux boîtes englobantes.
    :param boite1: [x gauche, y haut, largeur, hauteur] de la boîte 1
    :param boite2: [x gauche, y haut, largeur, hauteur] de la boîte 2
    :return: Score IoU
    """
    # Conversion des boîtes au format [x1, y1, x2, y2]
    boite1 = [boite1[0], boite1[1], boite1[0] +
              boite1[2], boite1[1] + boite1[3]]
    boite2 = [boite2[0], boite2[1], boite2[0] +
              boite2[2], boite2[1] + boite2[3]]

    # Calcul de l'intersection
    x1_inter = max(boite1[0], boite2[0])
    y1_inter = max(boite1[1], boite2[1])
    x2_inter = min(boite1[2], boite2[2])
    y2_inter = min(boite1[3], boite2[3])

    aire_inter = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Calcul de l'union
    aire_boite1 = (boite1[2] - boite1[0]) * (boite1[3] - boite1[1])
    aire_boite2 = (boite2[2] - boite2[0]) * (boite2[3] - boite2[1])
    aire_union = aire_boite1 + aire_boite2 - aire_inter

    # Calcul de IoU
    score_iou = aire_inter / aire_union

    return score_iou


def creer_matrice_similarite(detections):
    """
    Crée une matrice de similarité basée sur l'IoU pour toutes les détections.
    :param detections: Liste de détections
    :return: Matrice de similarité
    """
    nb_detections = len(detections)
    matrice_similarite = [[0.0] * nb_detections for _ in range(nb_detections)]

    for i in range(nb_detections):
        for j in range(nb_detections):
            if i != j:
                boite1 = [detections[i]['bb_left'], detections[i]['bb_top'],
                          detections[i]['bb_width'], detections[i]['bb_height']]
                boite2 = [detections[j]['bb_left'], detections[j]['bb_top'],
                          detections[j]['bb_width'], detections[j]['bb_height']]
                score_iou = calculer_intersection_sur_union(boite1, boite2)
                matrice_similarite[i][j] = score_iou

    return matrice_similarite


def associer_detections_et_pistes(detections, pistes, seuil_iou):
    """
    Associe les détections aux pistes existantes en utilisant un seuil IoU.
    :param detections: Liste des détections
    :param pistes: Liste des pistes
    :param seuil_iou: Seuil pour considérer une association
    :return: Pistes mises à jour
    """
    for piste in pistes:
        derniere_detection = piste['detections'][-1]
        meilleur_iou = -1
        meilleure_detection = None

        for detection in detections:
            iou = calculer_intersection_sur_union([derniere_detection['bb_left'], derniere_detection['bb_top'], derniere_detection['bb_width'], derniere_detection['bb_height']],
                                                  [detection['bb_left'], detection['bb_top'], detection['bb_width'], detection['bb_height']])
            if iou > meilleur_iou and iou > seuil_iou:
                meilleur_iou = iou
                meilleure_detection = detection

        if meilleure_detection:
            piste['detections'].append(meilleure_detection)
            detections.remove(meilleure_detection)

    return pistes


def gerer_pistes(pistes, detections, seuil_iou):
    """
    Gestion des pistes en utilisant le seuil IoU pour l'association.
    :param pistes: Liste des pistes
    :param detections: Liste des détections
    :param seuil_iou: Seuil pour l'association
    :return: Liste des pistes mises à jour
    """
    pistes_mises_a_jour = []

    # Associer les détections aux pistes existantes
    for piste in pistes:
        derniere_detection = piste['detections'][-1]
        meilleur_iou = -1
        meilleure_detection = None

        for detection in detections:
            iou = calculer_intersection_sur_union([derniere_detection['bb_left'], derniere_detection['bb_top'], derniere_detection['bb_width'], derniere_detection['bb_height']],
                                                  [detection['bb_left'], detection['bb_top'], detection['bb_width'], detection['bb_height']])
            if iou > meilleur_iou and iou > seuil_iou:
                meilleur_iou = iou
                meilleure_detection = detection

        if meilleure_detection:
            piste['detections'].append(meilleure_detection)
            pistes_mises_a_jour.append(piste)
            detections.remove(meilleure_detection)

    # Créer de nouvelles pistes pour les détections non associées
    for detection in detections:
        nouvelle_piste = {'id': len(
            pistes) + len(pistes_mises_a_jour), 'detections': [detection]}
        pistes_mises_a_jour.append(nouvelle_piste)

    return pistes_mises_a_jour


def dessiner_resultats_suivi(repertoire_images, pistes, seuil_iou=0.5):
    """
    Dessine les boîtes englobantes, identifiants et trajectoires pour visualiser les résultats du suivi.
    :param repertoire_images: Chemin vers les images
    :param pistes: Liste des pistes
    :param seuil_iou: Seuil IoU pour considérer les détections comme le même objet
    """
    fichiers_images = sorted(os.listdir(repertoire_images))
    ids_uniques = {}
    compteur_ids = 0

    # Création d'une seule fenêtre pour l'affichage
    cv2.namedWindow('Résultats de Suivi', cv2.WINDOW_NORMAL)

    for idx, nom_fichier in enumerate(fichiers_images):
        chemin_image = os.path.join(repertoire_images, nom_fichier)
        image = cv2.imread(chemin_image)

        correspondance_ids = {}
        for piste in pistes:
            for detection in piste['detections']:
                if detection['frame'] == idx:
                    boite_detection = [
                        detection['bb_left'], detection['bb_top'], detection['bb_width'], detection['bb_height']]
                    objet_trouve = False
                    for id_obj, boite in ids_uniques.items():
                        iou = calculer_intersection_sur_union(
                            boite, boite_detection)
                        if iou >= seuil_iou:
                            correspondance_ids[detection['id']] = id_obj
                            objet_trouve = True
                            break

                    if objet_trouve:
                        id_obj = correspondance_ids[detection['id']]
                    else:
                        id_obj = compteur_ids
                        ids_uniques[id_obj] = boite_detection
                        compteur_ids += 1

                    # Dessiner la boîte englobante et l'identifiant
                    x_gauche, y_haut, largeur, hauteur = int(detection['bb_left']), int(
                        detection['bb_top']), int(detection['bb_width']), int(detection['bb_height'])
                    cv2.rectangle(image, (x_gauche, y_haut), (x_gauche +
                                  largeur, y_haut + hauteur), (0, 255, 0), 2)
                    cv2.putText(image, str(id_obj), (x_gauche, y_haut - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('Résultats de Suivi', image)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()



def gestion_pistes_avec_algorithme_hongrois(pistes_actuelles, nouvelles_detections, matrice_de_similarite):
    """
    Gère les pistes en utilisant l'algorithme hongrois pour une assignation optimale.
    :param pistes_actuelles: Liste des pistes actuelles
    :param nouvelles_detections: Liste des nouvelles détections
    :param matrice_de_similarite: Matrice de similarité (IoU) entre les détections et les pistes
    :return: Liste des pistes mises à jour
    """
    nb_pistes = len(pistes_actuelles)
    nb_detections = len(nouvelles_detections)

    # Si aucune piste ou détection, retourner les pistes existantes
    if nb_pistes == 0 or nb_detections == 0:
        return pistes_actuelles

    # Conversion de la matrice de similarité en matrice de coût
    matrice_de_cout = -np.array(matrice_de_similarite)

    # Trouver l'assignation optimale avec l'algorithme hongrois
    indices_pistes, indices_detections = linear_sum_assignment(matrice_de_cout)

    pistes_mises_a_jour = pistes_actuelles.copy()

    # Mettre à jour les pistes avec les détections assignées
    for indice_piste, indice_detection in zip(indices_pistes, indices_detections):
        pistes_mises_a_jour[indice_piste]['detections'].append(
            nouvelles_detections[indice_detection])

    # Ajouter de nouvelles pistes pour les détections non assignées
    for i in range(nb_detections):
        if i not in indices_detections:
            nouvelle_piste = {
                'id': len(pistes_mises_a_jour),
                'detections': [nouvelles_detections[i]]
            }
            pistes_mises_a_jour.append(nouvelle_piste)

    return pistes_mises_a_jour


def sauvegarder_resultats_suivi(pistes, nom_sequence):
    """
    Sauvegarde les résultats de suivi dans un fichier texte.
    :param pistes: Liste des pistes
    :param nom_sequence: Nom de la séquence traitée
    """
    nom_fichier_sortie = f"{nom_sequence}_resultats_suivi.txt"

    with open(nom_fichier_sortie, 'w', newline='') as fichier:
        ecrivain = csv.writer(fichier, delimiter=',')
        for piste in pistes:
            for detection in piste['detections']:
                ecrivain.writerow([
                    detection['frame'],
                    piste['id'],
                    detection['bb_left'],
                    detection['bb_top'],
                    detection['bb_width'],
                    detection['bb_height'],
                    detection['conf'],
                    detection['pos_x'],
                    detection['pos_y'],
                    detection['pos_z']
                ])


def gestion_pistes_avec_hongrois_et_kalman(pistes_actives, nouvelles_detections, matrice_iou):
    """
    Gère les pistes en utilisant l'algorithme hongrois pour l'assignation et le filtre de Kalman pour la prédiction et la mise à jour des états.
    :param pistes_actives: Liste des pistes actuellement actives
    :param nouvelles_detections: Liste des nouvelles détections à considérer
    :param matrice_iou: Matrice de l'intersection sur l'union pour toutes les détections
    :return: Liste des pistes mises à jour
    """
    nombre_pistes = len(pistes_actives)
    nombre_detections = len(nouvelles_detections)

    if nombre_pistes == 0 or nombre_detections == 0:
        return pistes_actives

    matrice_couts = -np.array(matrice_iou)
    lignes_assignees, colonnes_assignees = linear_sum_assignment(matrice_couts)

    pistes_maj = []

    # Mise à jour des pistes avec les détections assignées en utilisant le filtre de Kalman
    for indice_piste, indice_detection in zip(lignes_assignees, colonnes_assignees):
        piste = pistes_actives[indice_piste]
        detection = nouvelles_detections[indice_detection]

        # Mise à jour du filtre de Kalman avec la nouvelle détection
        filtre_kalman = KalmanFilter(
            dt=1, u_x=0, u_y=0, std_acc=1, x_std_mesure=1, y_std_mesure=1)
        filtre_kalman.etat = np.array([[detection['pos_x']], [detection['pos_y']], [
                                      0], [0]])  # Initialisation de l'état du filtre
        # Mise à jour avec la détection
        filtre_kalman.mise_a_jour(
            np.array([[detection['pos_x']], [detection['pos_y']]]))

        # Ajout de l'état du filtre de Kalman à la piste
        piste['etat_kalman'] = filtre_kalman.obtenir_etat()
        pistes_maj.append(piste)

    # Gestion des pistes non assignées en prédisant leur état
    for indice_detection in range(nombre_detections):
        if indice_detection not in colonnes_assignees:
            detection = nouvelles_detections[indice_detection]

            # Prédiction du filtre de Kalman sans nouvelle mesure
            filtre_kalman = KalmanFilter(
                dt=1, u_x=0, u_y=0, std_acc=1, x_std_mesure=1, y_std_mesure=1)
            etat_predi, _ = filtre_kalman.predire()

            # Création d'une nouvelle piste avec l'état prédit
            nouvelle_piste = {'id': nombre_pistes + indice_detection,
                              'detections': [detection], 'etat_kalman': etat_predi}
            pistes_maj.append(nouvelle_piste)

    return pistes_maj


######

# Transformer pour préparer les images pour le modèle CNN
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Fonction pour extraire les embeddings visuels
def extract_visual_embedding(image, cnn_model):
    image_tensor = preprocess(image).unsqueeze(
        0) 


    with torch.no_grad():
        embedding = cnn_model(image_tensor)

    return embedding.squeeze().numpy()

# Mise à jour des pistes avec les embeddings visuels
def update_tracks_with_visual_info(tracks, detections, iou_matrix, cnn_model):
    cost_matrix = np.zeros_like(iou_matrix)
    for i, track in enumerate(tracks):
        for j, detection in enumerate(detections):

            track_embedding = extract_visual_embedding(
                track['last_frame'], cnn_model)
            detection_embedding = extract_visual_embedding(
                detection['frame'], cnn_model)

            visual_similarity = cosine_similarity(
                [track_embedding], [detection_embedding])[0][0]

            cost_matrix[i, j] = -iou_matrix[i, j] + visual_similarity

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    for row, col in zip(row_indices, col_indices):
        tracks[row]['detections'].append(detections[col])

    return tracks



def track_objects(frames, cnn_model):
    tracks = []
    detections = lire_detections(frames)
    iou_matrix = creer_matrice_similarite(detections)

    tracks = [{'id': i, 'detections': [detections[i]]}
              for i in range(len(detections))]
    images_folder_path = 'ADL-Rundle-6/img1'
    image_files = sorted([os.path.join(images_folder_path, file)
                         for file in os.listdir(images_folder_path)])
    sigma_iou = 0.5
    
    tracks = update_tracks_with_visual_info(
        tracks, detections, iou_matrix, cnn_model)

    return tracks


"""video_frames = 'ADL-Rundle-6/det/det.txt'
cnn_model = models.resnet50(pretrained=True)
tracked_objects = track_objects(video_frames, cnn_model)"""
