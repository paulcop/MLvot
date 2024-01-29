import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def lire_detections(chemin_fichier):
    data_detections = {}
    with open(chemin_fichier, 'r') as fichier:
        for ligne in fichier:
            elements = ligne.strip().split(',')
            num_frame = int(elements[0])
            detection = np.array([float(elem) for elem in elements[1:]])

            data_detections.setdefault(num_frame, []).append(detection)

    for frame in data_detections:
        data_detections[frame] = np.vstack(data_detections[frame])

    return data_detections


def calcul_iou(boite_a, boite_b):
    xa, ya, la, ha = boite_a
    xb, yb, lb, hb = boite_b

    inter_x = max(xa, xb)
    inter_y = max(ya, yb)
    inter_droit = min(xa + la, xb + lb)
    inter_bas = min(ya + ha, yb + hb)

    if inter_droit < inter_x or inter_bas < inter_y:
        return 0

    aire_inter = (inter_droit - inter_x) * (inter_bas - inter_y)
    aire_union = la * ha + lb * hb - aire_inter

    return aire_inter / aire_union if aire_union != 0 else 0


def associer_detections(detections, suivi_objets, num_frame, id_suivi, seuil):
    if num_frame not in detections:
        return suivi_objets, id_suivi

    # Préparation de la matrice de coût basée sur l'IoU
    nb_detections = len(detections[num_frame])
    nb_trajectoires = len(suivi_objets)
    coût = np.ones((nb_trajectoires, nb_detections))

    for t, trajet in enumerate(suivi_objets.values()):
        _, xt, yt, lt, ht, _, _, _, _ = trajet[-1]
        boite_trajet = [xt, yt, lt, ht]
        for d, (x, y, l, h, _, _, _, _, _) in enumerate(detections[num_frame]):
            coût[t, d] = 1 - calcul_iou([x, y, l, h], boite_trajet)

    # Appliquer l'algorithme hongrois
    lignes, colonnes = linear_sum_assignment(coût)

    # Associer les détections aux trajectoires
    detection_associee = np.zeros(nb_detections, dtype=bool)
    for ligne, colonne in zip(lignes, colonnes):
        if coût[ligne, colonne] < 1 - seuil:
            id_trajectoire = list(suivi_objets.keys())[ligne]
            suivi_objets[id_trajectoire].append(detections[num_frame][colonne])
            detections[num_frame][colonne][0] = id_trajectoire
            detection_associee[colonne] = True

    # Gérer les trajectoires non associées et les nouvelles détections
    for d, assoc in enumerate(detection_associee):
        if not assoc:
            detections[num_frame][d][0] = id_suivi
            suivi_objets[id_suivi] = [detections[num_frame][d]]
            id_suivi += 1

    return suivi_objets, id_suivi


def tracker_objets(detections, seuil_iou):
    suivi_actuel = {}
    id_actuel = 0

    for frame in range(1, max(detections.keys()) + 1):
        if frame in detections:
            suivi_actuel, id_actuel = associer_detections(
                detections, suivi_actuel, frame, id_actuel, seuil_iou)

    return suivi_actuel, id_actuel


def afficher_detections(detections, chemin):
    for num_frame, frame_detections in detections.items():
        chemin_image = f"ADL-Rundle-6/img1/{num_frame:06d}.jpg"
        img = cv2.imread(chemin_image)

        for detect in frame_detections:
            id_objet, x, y, l, h, _, _, _, _ = detect
            cv2.rectangle(img, (int(x), int(y)),
                          (int(x + l), int(y + h)), (255, 0, 0), 2)
            cv2.putText(img, f'ID:{int(id_objet)}', (int(x), int(
                y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imwrite(f"{chemin}/{num_frame:06d}_resultat.jpg", img)


def sauvegarder_resultats(suivi_objets, chemin_fichier_sortie):
    with open(chemin_fichier_sortie, 'w') as fichier:
        for id_trajectoire, trajet in suivi_objets.items():
            for detection in trajet:
                # Mettre à jour le champ 'conf' en 1
                detection[6] = 1
                ligne = ','.join(map(str, detection))
                fichier.write(ligne + '\n')


def executer():
    detections = lire_detections("ADL-Rundle-6/det/det.txt")
    suivi_objets, _ = tracker_objets(detections, 0.5)
    chemin_sortie = "output/"
    afficher_detections(detections, chemin_sortie)
    sauvegarder_resultats(suivi_objets, f"resultats_suivi.txt")


if __name__ == "__main__":
    executer()
