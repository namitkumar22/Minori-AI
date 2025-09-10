import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import tempfile
import time
from pathlib import Path
from detect_rice import DetectRice
from detect_wheat import DetectWheat


class MinoriApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Minori AI - Disease Detection")
        self.root.geometry("800x600")
        self.root.configure(bg='#2c3e50')
        
        # Application state
        self.models_loaded = False
        self.detection_active = False
        self.current_crop = tk.StringVar(value="Rice")
        
        # Results
        self.latest_result = "No Detection"
        self.is_healthy = True
        self.last_detection_time = 0
        self.detection_cooldown = 3.0
        
        # Camera
        self.cap = None
        self.temp_dir = Path(tempfile.mkdtemp(prefix="minori_"))
        
        # Setup UI
        self.setup_ui()
        
        # Initialize everything after UI is ready
        self.root.after(100, self.initialize_everything)
    
    def setup_ui(self):
        """Create the user interface"""
        # Header
        header_frame = tk.Frame(self.root, bg='#34495e', height=80)
        header_frame.pack(fill='x', padx=10, pady=10)
        header_frame.pack_propagate(False)
        
        tk.Label(
            header_frame,
            text="🌾 Minori AI - Disease Detection",
            font=('Arial', 18, 'bold'),
            fg='white',
            bg='#34495e'
        ).pack(expand=True)
        
        # Controls
        control_frame = tk.Frame(self.root, bg='#ecf0f1')
        control_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        # Crop selection
        tk.Label(
            control_frame,
            text="Select Crop:",
            font=('Arial', 12, 'bold'),
            bg='#ecf0f1'
        ).pack(side='left', padx=20, pady=15)
        
        self.crop_combo = ttk.Combobox(
            control_frame,
            textvariable=self.current_crop,
            values=['Rice', 'Wheat'],
            state='readonly',
            font=('Arial', 11),
            width=10
        )
        self.crop_combo.pack(side='left', pady=15)
        
        # Detection button
        self.detection_btn = tk.Button(
            control_frame,
            text="Start Detection",
            command=self.toggle_detection,
            font=('Arial', 12, 'bold'),
            bg='#27ae60',
            fg='white',
            padx=20,
            pady=8,
            state='disabled'
        )
        self.detection_btn.pack(side='left', padx=20, pady=15)
        
        # Detect Now button (for single detection)
        self.detect_now_btn = tk.Button(
            control_frame,
            text="Detect Now",
            command=self.detect_now,
            font=('Arial', 12, 'bold'),
            bg='#3498db',
            fg='white',
            padx=20,
            pady=8,
            state='disabled'
        )
        self.detect_now_btn.pack(side='left', padx=10, pady=15)
        
        # Status
        self.status_label = tk.Label(
            control_frame,
            text="Initializing...",
            font=('Arial', 11),
            bg='#ecf0f1',
            fg='#e74c3c'
        )
        self.status_label.pack(side='right', padx=20, pady=15)
        
        # Main content area
        content_frame = tk.Frame(self.root, bg='#2c3e50')
        content_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        # Camera panel
        camera_frame = tk.Frame(content_frame, bg='#34495e', relief='solid', bd=2)
        camera_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        self.camera_label = tk.Label(
            camera_frame,
            bg='black',
            text="📷 Camera Feed\n\nInitializing...",
            fg='white',
            font=('Arial', 12)
        )
        self.camera_label.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Results panel
        results_frame = tk.Frame(content_frame, bg='#34495e', width=250, relief='solid', bd=2)
        results_frame.pack(side='right', fill='y')
        results_frame.pack_propagate(False)
        
        # Results header
        tk.Label(
            results_frame,
            text="Detection Results",
            font=('Arial', 14, 'bold'),
            bg='#34495e',
            fg='white'
        ).pack(pady=10)
        
        # Results display
        self.result_text = tk.Text(
            results_frame,
            height=8,
            font=('Arial', 10),
            bg='#f8f9fa',
            relief='flat',
            wrap='word',
            state='disabled'
        )
        self.result_text.pack(fill='x', padx=10, pady=10)
        
        # Instructions
        instructions = """Instructions:
1. Select crop type
2. Click 'Detect Now' for single detection
   OR 'Start Detection' for continuous
3. Hold leaf in camera view
4. Keep steady for analysis
5. Green box = Healthy
6. Red box = Disease"""
        
        tk.Label(
            results_frame,
            text=instructions,
            font=('Arial', 9),
            bg='#34495e',
            fg='white',
            justify='left'
        ).pack(padx=10, pady=10)
    
    def initialize_everything(self):
        """Initialize camera and models after UI is ready"""
        try:
            # Update status
            self.status_label.config(text="Starting camera...", fg='#f39c12')
            self.root.update()
            
            # Initialize camera
            self.init_camera()
            
            # Load models
            self.status_label.config(text="Loading AI models...", fg='#f39c12')
            self.root.update()
            
            print("[INFO] Loading Rice detection model...")
            self.rice_detector = DetectRice()
            
            print("[INFO] Loading Wheat detection model...")
            self.wheat_detector = DetectWheat()
            
            self.models_loaded = True
            print("[INFO] All models loaded successfully!")
            
            # Enable buttons and start camera feed
            self.status_label.config(text="Ready for detection ✓", fg='#27ae60')
            self.detection_btn.config(state='normal')
            self.detect_now_btn.config(state='normal')
            
            # Start camera update loop
            self.update_camera()
            
        except Exception as e:
            error_msg = f"Initialization failed: {str(e)}"
            print(f"[ERROR] {error_msg}")
            self.status_label.config(text="Initialization failed ✗", fg='#e74c3c')
            messagebox.showerror("Initialization Error", error_msg)
    
    def init_camera(self):
        """Initialize camera"""
        try:
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
                        
            if not self.cap.isOpened():
                raise Exception("Could not access camera")
                
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            print("[INFO] Camera initialized successfully")
            
        except Exception as e:
            raise Exception(f"Camera initialization failed: {e}")
    
    def update_camera(self):
        """Update camera feed"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            
            if ret:
                frame = cv2.flip(frame, 1)  # Mirror effect
                
                # Auto-detect if continuous detection is active
                if (self.detection_active and 
                    time.time() - self.last_detection_time > self.detection_cooldown):
                    self.perform_detection_on_frame(frame)
                
                # Draw detection box if we have a result
                if self.latest_result != "No Detection":
                    frame = self.draw_detection_box(frame)
                
                # Display frame
                self.display_frame(frame)
            else:
                # Camera feed lost
                self.camera_label.config(
                    text="📷 Camera Feed\n\nCamera disconnected",
                    image=""
                )
        
        # Schedule next update
        self.root.after(30, self.update_camera)
    
    def detect_now(self):
        """Perform single detection"""
        if not self.models_loaded:
            messagebox.showwarning("Not Ready", "Models are still loading.")
            return
            
        if not (self.cap and self.cap.isOpened()):
            messagebox.showerror("Camera Error", "Camera is not available.")
            return
        
        # Capture current frame
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            self.perform_detection_on_frame(frame)
        else:
            messagebox.showerror("Capture Error", "Failed to capture frame.")
    
    def toggle_detection(self):
        """Toggle continuous detection on/off"""
        if not self.models_loaded:
            messagebox.showwarning("Not Ready", "Models are still loading.")
            return
        
        if not (self.cap and self.cap.isOpened()):
            messagebox.showerror("Camera Error", "Camera is not available.")
            return
        
        self.detection_active = not self.detection_active
        
        if self.detection_active:
            self.detection_btn.config(text="Stop Detection", bg='#e74c3c')
            self.status_label.config(text=f"Auto-detecting {self.current_crop.get()}...", fg='#e74c3c')
            self.crop_combo.config(state='disabled')
            self.detect_now_btn.config(state='disabled')
        else:
            self.detection_btn.config(text="Start Detection", bg='#27ae60')
            self.status_label.config(text="Ready for detection ✓", fg='#27ae60')
            self.crop_combo.config(state='readonly')
            self.detect_now_btn.config(state='normal')
            self.latest_result = "No Detection"
    
    def perform_detection_on_frame(self, frame):
        """Perform detection on given frame"""
        try:
            # Save frame temporarily
            timestamp = int(time.time() * 1000)
            temp_path = self.temp_dir / f"frame_{timestamp}.jpg"
            cv2.imwrite(str(temp_path), frame)
            
            # Update status
            self.status_label.config(text=f"Analyzing {self.current_crop.get()}...", fg='#f39c12')
            self.root.update()
            
            try:
                # Run detection based on crop type
                crop_type = self.current_crop.get()
                if crop_type == "Rice":
                    result = self.rice_detector.detect_rice_disease(str(temp_path))
                else:  # Wheat
                    result = self.wheat_detector.detect_wheat_disease(str(temp_path))
                
                # Process result
                self.process_detection_result(result, crop_type)
                self.last_detection_time = time.time()
                
            finally:
                # Clean up temp file
                if temp_path.exists():
                    temp_path.unlink()
                    
        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")
            self.status_label.config(text="Detection failed ✗", fg='#e74c3c')
    
    def process_detection_result(self, result, crop_type):
        """Process and display detection result"""
        self.latest_result = result
        self.is_healthy = "healthy" in result.lower()
        
        # Update results display
        self.update_results_display(result, crop_type)
        
        # Update status
        status_text = "✅ Healthy detected" if self.is_healthy else "🚨 Disease detected"
        if self.detection_active:
            status_text += f" - Auto-detecting {crop_type}..."
        else:
            status_text += " - Ready for next detection"
            
        color = '#27ae60' if self.is_healthy else '#e74c3c'
        self.status_label.config(text=status_text, fg=color)
        
        print(f"[DETECTION] {crop_type}: {result} ({'Healthy' if self.is_healthy else 'Disease'})")
    
    def update_results_display(self, result, crop_type):
        """Update the results text display"""
        self.result_text.config(state='normal')
        self.result_text.delete(1.0, tk.END)
        
        timestamp = time.strftime("%H:%M:%S")
        status = "✅ Healthy" if self.is_healthy else "🚨 Disease Detected"
        disease_name = result.replace('_', ' ').title()
        
        result_info = f"Time: {timestamp}\n"
        result_info += f"Crop: {crop_type}\n"
        result_info += f"Status: {status}\n"
        result_info += f"Detection: {disease_name}\n"
        
        if not self.is_healthy:
            result_info += "\n⚠️ Recommendation:\nConsult agricultural expert\nfor treatment options."
        else:
            result_info += "\n✅ Plant appears healthy.\nContinue regular monitoring."
        
        self.result_text.insert(1.0, result_info)
        self.result_text.config(state='disabled')
    
    def draw_detection_box(self, frame):
        """Draw detection box on frame"""
        h, w = frame.shape[:2]
        
        # Create centered detection box
        box_size = min(h, w) // 2
        x1 = (w - box_size) // 2
        y1 = (h - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size
        
        # Color based on health status
        color = (0, 255, 0) if self.is_healthy else (0, 0, 255)  # Green/Red
        
        # Draw main box with thicker border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        
        # Add corner markers for a more professional look
        corner_length = 30
        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, 6)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, 6)
        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, 6)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, 6)
        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, 6)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, 6)
        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, 6)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, 6)
        
        # Add label
        label = "HEALTHY" if self.is_healthy else "DISEASE DETECTED"
        font_scale = 0.8
        thickness = 2
        
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Label background
        cv2.rectangle(frame, (x1, y1 - text_h - 15), (x1 + text_w + 10, y1 - 5), color, -1)
        
        # Label text
        cv2.putText(frame, label, (x1 + 5, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def display_frame(self, frame):
        """Convert and display frame in tkinter"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Resize to fit label
            label_w = self.camera_label.winfo_width()
            label_h = self.camera_label.winfo_height()
            
            if label_w > 1 and label_h > 1:
                img = img.resize((label_w - 10, label_h - 10), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img)
            self.camera_label.configure(image=photo, text="")
            self.camera_label.image = photo
            
        except Exception as e:
            print(f"[ERROR] Display frame: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        self.detection_active = False
        
        if self.cap:
            self.cap.release()
        
        # Clean temp directory
        try:
            for temp_file in self.temp_dir.glob("*"):
                temp_file.unlink()
            self.temp_dir.rmdir()
        except Exception as e:
            print(f"[ERROR] Cleanup: {e}")
        
        cv2.destroyAllWindows()


def main():
    """Main application entry point"""
    print("[INFO] Starting Minori AI...")
    root = tk.Tk()
    app = MinoriApp(root)
    
    def on_closing():
        print("[INFO] Shutting down...")
        app.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("[INFO] Application interrupted by user")
        on_closing()


if __name__ == "__main__":
    main()