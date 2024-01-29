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
trajectory = []

for i in range(125):
    ret, frame = cap.read()
    if not ret:
        break

    centers = detect(frame)
    if centers:  # Check if any center is detected
        detected_point = centers[0].reshape(-1)

        kf.update(detected_point)
    
    kf.predict()
    predicted_state = kf.x

    # Store the predicted point in the trajectory
    trajectory.append((int(predicted_state[0]), int(predicted_state[1])))

    # Draw trajectory
    for j in range(1, len(trajectory)):
        cv2.line(frame, trajectory[j - 1], trajectory[j], (255, 255, 0), 2)


    if centers:
        # Draw rectangle around detected point
        cv2.rectangle(frame, (int(detected_point[0] - 10), int(detected_point[1] - 10)),
                      (int(detected_point[0] + 10), int(detected_point[1] + 10)), (0, 0, 255), 2)
        
        # Draw a green circle at the center of the detected point
        cv2.circle(frame, (int(detected_point[0]), int(
            detected_point[1])), 3, (0, 255, 0), -1)

    # Draw red rectangle for prediction
    cv2.rectangle(frame, (int(predicted_state[0] - 10), int(predicted_state[1] - 10)),
                  (int(predicted_state[0] + 10), int(predicted_state[1] + 10)), (255, 0, 0), 2)

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
