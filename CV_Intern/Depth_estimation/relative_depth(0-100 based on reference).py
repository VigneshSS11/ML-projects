import cv2
import mediapipe as mp
import numpy as np
import time

class FaceDepthEstimator:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        # Initialize with default values
        self.very_close = 50000  # Face area when very close
        self.very_far = 20000    # Face area when very far
        self.calibrated = False
        self.calibration_samples = {'close': [], 'far': []}
        
    def get_face_area(self, face_landmarks, image_shape):
        """Calculate face area in pixels"""
        h, w = image_shape[:2]
        
        # Get face width (distance between ears)
        left_ear = face_landmarks.landmark[234]
        right_ear = face_landmarks.landmark[454]
        face_width = abs(right_ear.x - left_ear.x) * w
        
        # Get face height (forehead to chin)
        forehead = face_landmarks.landmark[10]
        chin = face_landmarks.landmark[152]
        face_height = abs(chin.y - forehead.y) * h
        
        return face_width * face_height, (face_width, face_height)
    
    def calibrate(self, frame, position='close'):
        """Collect calibration samples for close or far positions"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return frame, "No face detected"
            
        face_landmarks = results.multi_face_landmarks[0]
        face_area, _ = self.get_face_area(face_landmarks, frame.shape)
        
        # Add to calibration samples
        self.calibration_samples[position].append(face_area)
        
        # If we have enough samples for both positions
        if (len(self.calibration_samples['close']) >= 30 and 
            len(self.calibration_samples['far']) >= 30):
            # Calculate average values
            self.very_close = np.median(self.calibration_samples['close'])
            self.very_far = np.median(self.calibration_samples['far'])
            self.calibrated = True
            
        # Draw calibration info
        cv2.putText(frame, f"Calibrating {position} position", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Current area: {face_area:.0f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Samples: {len(self.calibration_samples[position])}/30", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame, "Calibrating..."
    
    def estimate_depth(self, frame):
        """Estimate relative depth based on face size"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return frame, None, "No face detected"
        
        face_landmarks = results.multi_face_landmarks[0]
        face_area, (face_width, face_height) = self.get_face_area(face_landmarks, frame.shape)
        
        # Calculate relative depth (0-100 scale)
        relative_depth = 100 - min(100, max(0, 
            ((face_area - self.very_far) / (self.very_close - self.very_far)) * 100
        ))
        
        # Draw debug information
        cv2.putText(frame, f"Face Area: {face_area:.0f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Width: {face_width:.0f}px", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Height: {face_height:.0f}px", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Depth: {relative_depth:.1f}%", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw distance ruler
        self.draw_ruler(frame, relative_depth)
        
        return frame, relative_depth, "OK"
    
    def draw_ruler(self, frame, relative_depth):
        """Draw a visual ruler showing depth"""
        height = frame.shape[0]
        width = frame.shape[1]
        
        # Draw ruler on right side
        ruler_x = width - 50
        for i in range(10):
            y = int(height * i / 10)
            cv2.line(frame, (ruler_x, y), (ruler_x + 20, y), (0, 255, 0), 1)
            cv2.putText(frame, f"{100-i*10}", (ruler_x + 25, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw current depth indicator
        indicator_y = int(height * relative_depth / 100)
        cv2.circle(frame, (ruler_x + 10, indicator_y), 5, (0, 0, 255), -1)

def main():
    cap = cv2.VideoCapture(0)
    estimator = FaceDepthEstimator()
    
    # Calibration mode
    print("Starting calibration...")
    print("Position your face CLOSE to the camera and press 'c'")
    print("Position your face FAR from the camera and press 'f'")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if not estimator.calibrated:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                frame, status = estimator.calibrate(frame, 'close')
            elif key == ord('f'):
                frame, status = estimator.calibrate(frame, 'far')
            elif key == ord('q'):
                break
        else:
            frame, depth, status = estimator.estimate_depth(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cv2.imshow("Face Depth Estimation", frame)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
