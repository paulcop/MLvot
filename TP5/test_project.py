import iou_kalman as iou
import os

print("Démarrage du test.")
file_path = 'ADL-Rundle-6/det/det.txt'
print(f"Chargement des détections depuis {file_path}...")
loaded_detections = iou.load_detections(file_path)

print("Création de la matrice de similarité...")
similarity_matrix = iou.create_similarity_matrix(loaded_detections)

print("Initialisation des pistes...")
tracks = [{'id': i, 'detections': [loaded_detections[i]]}
          for i in range(len(loaded_detections))]

print("Chargement des images...")
images_folder_path = 'ADL-Rundle-6/img1'
image_files = sorted([os.path.join(images_folder_path, file)
                     for file in os.listdir(images_folder_path)])


sigma_iou = 0.5  # Seuil pour l'IoU
print("Association des détections aux pistes...")
### LIGNE TP02 ###
### updated_tracks = iou.track_management(tracks, loaded_detections, sigma_iou)
### LIGNE TP03 ###
### updated_tracks_hungarian = iou.track_management_with_hungarian(tracks, loaded_detections, similarity_matrix)
### LIGNE TP04 ###
updated_tracks_hungarian = iou.track_management_with_hungarian_and_kalman_filter(tracks, loaded_detections, similarity_matrix)
### LIGNE TP05 ###
#updated_trads_IA = iou.track_management_with_hungarian_and_kalman_filter_and_ia(
#    tracks, loaded_detections, similarity_matrix, "ResNet50")

print(updated_tracks_hungarian[0])
print("Finish track_management")

print("Enregistrement des résultats...")
sequence_name = "TP04"
iou.save_tracking_results(updated_tracks_hungarian, sequence_name)

print("Début de la visualisation du suivi...")
iou.draw_tracking_results(images_folder_path, updated_tracks_hungarian)

print("Fin du test.")
