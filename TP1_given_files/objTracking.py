import cv2
import numpy as np
from KalmanFilter import KalmanFilter
from Detector import detect
import matplotlib.pyplot as plt
from PIL import Image

kf = KalmanFilter(dt=0.1, u_x=1, u_y=1, std_acc=1, x_std_meas=0.1, y_std_meas=0.1)
cap = cv2.VideoCapture("TP1_given_files/randomball.avi")
fig = plt.figure()
frames = []
positions = []
for i in range(125):
    ret, frame = cap.read()
    if not ret:
        break

    centers = detect(frame)
    detected_point = detect(frame)[0]
    detected_point = detected_point.reshape(-1)

    kf.update(detected_point)
    kf.predict()

    # Imprimer l'état prédit pour le diagnostic
    print("Predicted State:", kf.x.T)

    predicted_position = (int(kf.x[0, 0]), int(kf.x[1, 0]))
    positions.append(predicted_position)

    #for j in range(1, len(positions)):
    #    cv2.line(frame, positions[j - 1], positions[j], (0, 255, 255), 2)

    #cv2.rectangle(frame, (predicted_position[0] - 10, predicted_position[1] - 10),
    #              (predicted_position[0] + 10, predicted_position[1] + 10), (0, 255, 0), 2)

    cv2.rectangle(frame, (int(detected_point[0] - 10), int(detected_point[1] - 10)),
                  (int(detected_point[0] + 10), int(detected_point[1] + 10)), (0, 0, 255), 2)

    frames.append(frame)

images = [Image.fromarray(f, mode="RGB") for f in frames]
output_file = "randomball.gif"
images[0].save(
    output_file,
    format="GIF",
    append_images=images[1:],
    save_all=True,
    duration=10,
    loop=0,
)
