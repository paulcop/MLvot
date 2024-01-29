import iou_kalman as iou
import os


print("Start test")
file_path = 'ADL-Rundle-6/det/det.txt'
print(f"Load the file {file_path}")
loaded_detections = iou.lire_detections(file_path)
print("load finish")
# print(f"{loaded_detections[0]}")

print("Start similarity creation matrix")
# Créer la matrice de similarité
similarity_matrix = iou.creer_matrice_similarite(loaded_detections)
print("creation similarity matrix finish")
# print(similarity_matrix[0])

# Initialiser les pistes (tracks)
print("Start creat tracks")
tracks = [{'id': i, 'detections': [loaded_detections[i]]} for i in range(len(loaded_detections))]
print(tracks[0])
print("finish create tracks")

# Définir le chemin vers le dossier contenant les images
print("Start load all pictures")
images_folder_path = 'ADL-Rundle-6/img1'  
# Lire les images du dossier
image_files = sorted([os.path.join(images_folder_path, file) for file in os.listdir(images_folder_path)])
print("finish load all picture")

# Définir le seuil sigma_iou
print("Start associate detection to track")
sigma_iou = 0.5  # Vous pouvez ajuster cette valeur selon vos besoins
# Associer les détections aux pistes
# updated_tracks = iou.associate_detections_to_tracks(loaded_detections, tracks, sigma_iou)
# print(updated_tracks[0])
print("Finish associate detection to track")

print("Start track management")
### LIGNE TP02 ###
#updated_tracks = iou.gerer_pistes(tracks, loaded_detections, sigma_iou)
### LIGNE TP03 ###
#updated_tracks_hungarian = iou.gestion_pistes_avec_algorithme_hongrois(
#    tracks, loaded_detections, similarity_matrix)
### LIGNE TP04 ###
#updated_tracks_hungarian = iou.gestion_pistes_avec_hongrois_et_kalman(
#    tracks, loaded_detections, similarity_matrix)
### LIGNE TP05 ###
updated_trads_IA = iou.track_management_with_hungarian_and_kalman_filter_and_ia(
    tracks, loaded_detections, similarity_matrix, "ResNet50")


print(updated_trads_IA[0])
print("Finish track_management")

print("Start writing")
sequence_name = "TP04"
iou.sauvegarder_resultats_suivi(updated_trads_IA, sequence_name)
print("Finish writing")

print("Start Drawing")
# Afficher les résultats du suivi
iou.dessiner_resultats_suivi(images_folder_path, updated_trads_IA)
print("Finish Drawing")
