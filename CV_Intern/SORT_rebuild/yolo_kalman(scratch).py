import cv2
import numpy as np
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression  # Import NMS utility

class KalmanFilter:
    def __init__(self):
        # Define state dimensions
        self.state = np.zeros((4, 1), dtype=np.float32)  # [x, y, vx, vy]
        self.state_cov = np.eye(4, dtype=np.float32)     # State covariance matrix

        # Define matrices
        self.transition_matrix = np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], dtype=np.float32)  # F
        self.measurement_matrix = np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0]], dtype=np.float32)  # H
        self.process_noise_cov = np.eye(4, dtype=np.float32) * 1e-2  # Q
        self.measurement_noise_cov = np.eye(2, dtype=np.float32) * 1e-1  # R
        self.identity = np.eye(4, dtype=np.float32)  # Identity matrix (I)

    def predict(self):
        # State prediction: x' = Fx
        self.state = np.dot(self.transition_matrix, self.state)

        # Covariance prediction: P' = FPF' + Q
        self.state_cov = np.dot(np.dot(self.transition_matrix, self.state_cov), 
                                self.transition_matrix.T) + self.process_noise_cov

        return self.state[:2]  # Return predicted (x, y)

    def correct(self, measurement):
        # Compute Kalman gain: K = PH'(HPH' + R)^-1
        S = np.dot(self.measurement_matrix, 
                   np.dot(self.state_cov, self.measurement_matrix.T)) + self.measurement_noise_cov
        K = np.dot(np.dot(self.state_cov, self.measurement_matrix.T), np.linalg.inv(S))

        # Update state with measurement: x = x' + K(z - Hx')
        y = measurement - np.dot(self.measurement_matrix, self.state)
        self.state = self.state + np.dot(K, y)

        # Update covariance: P = (I - KH)P'
        self.state_cov = np.dot(self.identity - np.dot(K, self.measurement_matrix), self.state_cov)

        return self.state

# Instantiate the custom Kalman filter
kalman_filter = KalmanFilter()

# Load the YOLOv5 model
MODEL_PATH = "/home/vignesh/Downloads/best_weights.pt"
device = 'cpu'
model = attempt_load(MODEL_PATH, device)

# Initialize webcam capture
cap = cv2.VideoCapture(0)

IMG_SIZE = 480
CONF_THRESHOLD = 0.25

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]  # Original frame size
    img_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img_tensor = torch.from_numpy(img_resized).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor)[0]

    pred = non_max_suppression(pred, conf_thres=CONF_THRESHOLD, iou_thres=0.45)[0]

    if pred is not None and len(pred) > 0:
        # Scale coordinates manually without scale_coords
        gain_w, gain_h = w / IMG_SIZE, h / IMG_SIZE  # Scaling factors
        pred[:, :4] *= torch.tensor([gain_w, gain_h, gain_w, gain_h])  # Scale bounding box

        # Select the bounding box with the maximum area
        selected_box = max(pred, key=lambda det: (det[2] - det[0]) * (det[3] - det[1]))

        # Extract the selected bounding box details
        x1, y1, x2, y2, conf, cls = selected_box.cpu().numpy()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        conf = float(conf)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Kalman filter
        measurement = np.array([[center_x], [center_y]], dtype=np.float32)
        kalman_filter.correct(measurement)
        prediction = kalman_filter.predict()
        pred_x, pred_y = int(prediction[0]), int(prediction[1])

        # Draw bounding box and centers
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        cv2.circle(frame, (pred_x, pred_y), 5, (255, 0, 0), -1)

        # Display confidence score
        cv2.putText(frame, f'Conf: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        # No detections
        cv2.putText(frame, 'No detections', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('YOLOv5 + Custom Kalman Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

