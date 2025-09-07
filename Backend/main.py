import cv2
import numpy as np
from detect_rice import DetectRice
from detect_wheat import DetectWheat
from GenerateSolution import GenerateSolution

class Main():
    def __init__(self):
        self.wheat_detector = DetectWheat()
        self.rice_detector = DetectRice()
        self.solution_generator = GenerateSolution()
        self.solution_generator.process_documents()
        self.solutions = {}
        self.frame_count = 0

    def detect_and_mark(self, frame, crop_type):
        # Detect every 30 frames to reduce processing load
        if self.frame_count % 30 == 0:
            cv2.imwrite("Live", frame)
            
            if crop_type == "wheat":
                disease = self.wheat_detector.detect_wheat_disease(frame)
            else:
                disease = self.rice_detector.detect_rice_disease(frame)
            
            # Generate solution if new disease detected
            if disease not in self.solutions:
                self.solution_generator.generate_with_context(crop_type, disease)
                self.solutions[disease] = self.solution_generator.response["answer"][:200] + "..."
        
        # Draw red box around center area (crop detection zone)
        h, w = frame.shape[:2]
        box_size = 200
        x1 = (w - box_size) // 2
        y1 = (h - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(frame, f"Scanning {crop_type.title()}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame

    def display_solutions(self, frame):
        y_pos = 30
        for disease, solution in self.solutions.items():
            # Disease name in red
            cv2.putText(frame, f"Disease: {disease}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Solution text in white with black background
            lines = [solution[i:i+60] for i in range(0, len(solution), 60)][:3]
            for i, line in enumerate(lines):
                y_text = y_pos + 25 + (i * 20)
                cv2.rectangle(frame, (10, y_text-15), (10 + len(line)*8, y_text+5), (0, 0, 0), -1)
                cv2.putText(frame, line, (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            y_pos += 100
        
        return frame

    def main(self):
        crop_type = input("Enter crop type (wheat/rice): ").lower()
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Camera ready. Press 'q' to quit, 's' to capture and analyze")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Mark detection area
            frame = self.detect_and_mark(frame, crop_type)
            
            # Display solutions
            frame = self.display_solutions(frame)
            
            # Show instructions
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Crop Disease Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    obj = Main()
    obj.main()