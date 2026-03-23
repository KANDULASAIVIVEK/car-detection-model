import cv2
import numpy as np
from ultralytics import YOLO

class TrafficDetector:
    def __init__(self, model_path='yolov8n.pt'):
        # Load the YOLOv8 model (downloads automatically if not present)
        self.model = YOLO(model_path)
        
        # COCO class IDs
        self.PERSON_CLASS = 0
        self.CAR_CLASS = 2
        
    @staticmethod
    def detect_color(roi: np.ndarray) -> str:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define color ranges in HSV (12 colors)
        color_ranges = {
            "Red": [
                (np.array([0, 50, 50]), np.array([10, 255, 255])),
                (np.array([160, 50, 50]), np.array([180, 255, 255]))
            ],
            "Orange": [(np.array([10, 50, 120]), np.array([20, 255, 255]))],
            "Brown": [(np.array([5, 50, 20]), np.array([25, 255, 120]))],
            "Yellow": [(np.array([20, 50, 120]), np.array([35, 255, 255]))],
            "Green": [(np.array([35, 50, 50]), np.array([85, 255, 255]))],
            "Cyan": [(np.array([85, 50, 50]), np.array([100, 255, 255]))],
            "Blue": [(np.array([100, 50, 50]), np.array([130, 255, 255]))],
            "Purple": [(np.array([130, 50, 50]), np.array([150, 255, 255]))],
            "Pink": [(np.array([150, 50, 50]), np.array([160, 255, 255]))],
            "White": [(np.array([0, 0, 200]), np.array([180, 50, 255]))],
            "Silver/Gray": [(np.array([0, 0, 50]), np.array([180, 50, 200]))],
            "Black": [(np.array([0, 0, 0]), np.array([180, 255, 50]))]
        }
        
        total_pixels = roi.shape[0] * roi.shape[1]
        if total_pixels == 0:
            return "Other"
            
        best_color = "Other"
        max_ratio = 0.0
        
        for color_name, ranges in color_ranges.items():
            combined_mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
            for lower, upper in ranges:
                mask = cv2.inRange(hsv, lower, upper)
                combined_mask = cv2.bitwise_or(combined_mask, mask)
                
            color_pixels = cv2.countNonZero(combined_mask)
            ratio = float(color_pixels) / float(total_pixels)
            
            if ratio > max_ratio and ratio > 0.15:  # 15% threshold for dominant color
                max_ratio = ratio
                best_color = color_name
                
        return best_color

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        # Run YOLO inference
        results = self.model(image)
        
        color_bgr = {
            "Red": (0, 0, 255),
            "Orange": (0, 165, 255),
            "Brown": (42, 42, 165),
            "Yellow": (0, 255, 255),
            "Green": (0, 255, 0),
            "Cyan": (255, 255, 0),
            "Blue": (255, 0, 0),
            "Purple": (128, 0, 128),
            "Pink": (203, 192, 255),
            "White": (255, 255, 255),
            "Silver/Gray": (192, 192, 192),
            "Black": (0, 0, 0),
            "Other": (255, 0, 255)
        }
        
        color_counts = {k: 0 for k in color_bgr.keys()}
        person_count: int = 0
        
        # Loop through detected boxes
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Class ID
                cls_id = int(box.cls[0].item())
                
                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Clamp coordinates to image dimensions
                h, w = image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Confidence
                conf = float(box.conf[0].item())
                
                if conf < 0.3:
                    continue
                    
                if cls_id == self.CAR_CLASS:
                    # Extract RoI
                    roi = image[y1:y2, x1:x2]
                    
                    if roi.size == 0:
                        continue
                        
                    color_name = self.detect_color(roi)
                    color_counts[color_name] += 1
                    
                    box_color = color_bgr.get(color_name, (255, 0, 255))
                    
                    # Draw rectangle
                    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
                    
                    # Better text visibility (black background)
                    label = f"{color_name} Car"
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(image, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 0, 0), -1)
                    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                        
                elif cls_id == self.PERSON_CLASS:
                    person_count += 1
                    # Draw GREEN rectangle for people
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    label = "Person"
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(image, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 0, 0), -1)
                    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
        total_cars: int = sum(color_counts.values())
        
        counts = {
            'total_cars': total_cars,
            'color_counts': color_counts,
            'people': person_count
        }
        
        return image, counts
