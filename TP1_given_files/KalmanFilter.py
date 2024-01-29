import numpy as np


class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        # Temps d'échantillonnage
        self.dt = dt
        self.u_x = u_x
        self.u_y = u_y
        self.std_acc_x = y_std_meas
        self.std_acc_y = x_std_meas
        self.time_state = 0

        # Vecteur de contrôle (accélération en x et y)
        self.u = np.array([[u_x], [u_y]])

        # État initial
        self.x = np.array([[0], [0], [0], [0]])  # x0, y0, vx0, vy0

        # Matrice du modèle du système
        self.A = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # Matrice de contrôle
        self.B = np.array([[(self.dt**2) / 2, 0],
                           [0, (self.dt**2) / 2],
                           [self.dt, 0],
                           [0, self.dt]])

        # Matrice de mappage de mesure
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # Covariance du bruit du processus
        self.Q = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]]) * std_acc**2

        # Covariance du bruit de mesure
        self.R = np.array([[x_std_meas**2, 0],
                           [0, y_std_meas**2]])

        # Covariance initiale de l'erreur de prédiction
        self.P = np.eye(self.A.shape[1])

    def predict(self):
        # Mise à jour de l'état
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)

        # Mise à jour de la covariance de l'erreur
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        z = np.array(z).reshape(-1, 1)
        # Calcul du gain de Kalman
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Mise à jour de l'état
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)

        # Mise à jour de la covariance de l'erreur
        I = np.eye(self.H.shape[1])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

    def get_state(self):
        return self.state
