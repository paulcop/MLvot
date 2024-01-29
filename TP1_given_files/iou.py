import cv2
import numpy as np


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
    a_supprimer = []
    detection_associee = np.zeros(len(detections[num_frame]), dtype=bool)

    for id_objet, trajet in suivi_objets.items():
        _, xt, yt, lt, ht, _, _, _, _ = trajet[-1]
        boite_trajet = [xt, yt, lt, ht]

        iou_max = -1
        detection_id = -1

        for i, (x, y, l, h, _, _, _, _, _) in enumerate(detections[num_frame]):
            if not detection_associee[i]:
                iou_actuel = calcul_iou([x, y, l, h], boite_trajet)
                if iou_actuel > iou_max:
                    iou_max = iou_actuel
                    detection_id = i

        if iou_max < seuil:
            a_supprimer.append(id_objet)
        elif detection_id != -1:
            trajet.append(detections[num_frame][detection_id])
            detections[num_frame][detection_id][0] = id_objet
            detection_associee[detection_id] = True

    for id_del in reversed(a_supprimer):
        del suivi_objets[id_del]

    for i, assoc in enumerate(detection_associee):
        if not assoc:
            detections[num_frame][i][0] = id_suivi
            suivi_objets[id_suivi] = [detections[num_frame][i]]
            id_suivi += 1

    return suivi_objets, id_suivi


def tracker_objets(detections, seuil_iou):
    suivi_actuel = {}
    id_actuel = 0

    for frame in range(1, max(detections.keys()) + 1):
        if frame in detections:
            suivi_actuel, id_actuel = associer_detections(
                detections, suivi_actuel, frame, id_actuel, seuil_iou)

    return detections


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


def executer():
    detections = lire_detections("ADL-Rundle-6/det/det.txt")
    detections = tracker_objets(detections, 0.5)
    chemin_sortie = "output/"
    afficher_detections(detections, chemin_sortie)


if __name__ == "__main__":
    executer()
