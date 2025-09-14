import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import cv2
from PIL import Image, ImageTk
import numpy as np
import tempfile
import time
from pathlib import Path
from detect_rice import DetectRice
from detect_wheat import DetectWheat
from GenerateSolution import GenerateSolution
import threading
from datetime import datetime


class MinoriApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Minori AI - Disease Detection & Solution")
        self.root.geometry("1400x800")
        self.root.configure(bg='#2c3e50')
        
        # Application state
        self.models_loaded = False
        self.solution_model_loaded = False
        self.detection_active = False
        self.current_crop = tk.StringVar(value="Rice")
        
        # Results
        self.latest_result = "No Detection"
        self.is_healthy = True
        self.last_detection_time = 0
        self.detection_cooldown = 3.0
        
        # Disease history and solutions
        self.disease_history = []  # List of detected diseases with timestamps
        self.disease_solutions = {}  # Dictionary to store solutions for each disease
        self.generating_solution = False
        
        # Camera
        self.cap = None
        self.temp_dir = Path(tempfile.mkdtemp(prefix="minori_"))
        
        # Setup UI
        self.setup_ui()
        
        # Initialize everything after UI is ready
        self.root.after(100, self.initialize_everything)
    
    def setup_ui(self):
        """Create the user interface with solution panel"""
        # Header
        header_frame = tk.Frame(self.root, bg='#34495e', height=80)
        header_frame.pack(fill='x', padx=10, pady=10)
        header_frame.pack_propagate(False)
        
        tk.Label(
            header_frame,
            text="🌾 Minori AI - Disease Detection & Solution",
            font=('Arial', 20, 'bold'),
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
        
        # Clear History button
        self.clear_history_btn = tk.Button(
            control_frame,
            text="Clear History",
            command=self.clear_history,
            font=('Arial', 12, 'bold'),
            bg='#e67e22',
            fg='white',
            padx=20,
            pady=8
        )
        self.clear_history_btn.pack(side='left', padx=10, pady=15)
        
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
        
        # Left side - Camera and current results
        left_frame = tk.Frame(content_frame, bg='#2c3e50')
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Camera panel
        camera_frame = tk.Frame(left_frame, bg='#34495e', relief='solid', bd=2)
        camera_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        self.camera_label = tk.Label(
            camera_frame,
            bg='black',
            text="📷 Camera Feed\n\nInitializing...",
            fg='white',
            font=('Arial', 12)
        )
        self.camera_label.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Current Detection Results panel
        current_results_frame = tk.Frame(left_frame, bg='#34495e', height=150, relief='solid', bd=2)
        current_results_frame.pack(fill='x', pady=(0, 5))
        current_results_frame.pack_propagate(False)
        
        # Current Results header
        tk.Label(
            current_results_frame,
            text="Current Detection",
            font=('Arial', 14, 'bold'),
            bg='#34495e',
            fg='white'
        ).pack(pady=5)
        
        # Current Results display
        self.result_text = tk.Text(
            current_results_frame,
            height=6,
            font=('Arial', 10),
            bg='#f8f9fa',
            relief='flat',
            wrap='word',
            state='disabled'
        )
        self.result_text.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        # Right side - Solutions panel
        solutions_frame = tk.Frame(content_frame, bg='#34495e', width=500, relief='solid', bd=2)
        solutions_frame.pack(side='right', fill='both', padx=(5, 0))
        solutions_frame.pack_propagate(False)
        
        # Solutions header
        solutions_header = tk.Frame(solutions_frame, bg='#34495e')
        solutions_header.pack(fill='x', pady=5)
        
        tk.Label(
            solutions_header,
            text="🔬 Disease Solutions & History",
            font=('Arial', 14, 'bold'),
            bg='#34495e',
            fg='white'
        ).pack(side='left', padx=10)
        
        # Solution status indicator
        self.solution_status = tk.Label(
            solutions_header,
            text="",
            font=('Arial', 10),
            bg='#34495e',
            fg='#f39c12'
        )
        self.solution_status.pack(side='right', padx=10)
        
        # Solutions display with scrollbar
        solutions_container = tk.Frame(solutions_frame, bg='#34495e')
        solutions_container.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        self.solutions_text = scrolledtext.ScrolledText(
            solutions_container,
            font=('Arial', 10),
            bg='#f8f9fa',
            relief='flat',
            wrap='word',
            state='disabled'
        )
        self.solutions_text.pack(fill='both', expand=True)
        
        # Instructions at bottom
        instructions = """Instructions:
1. Select crop type  2. Click 'Detect Now' for single detection OR 'Start Detection' for continuous
3. Hold leaf in camera view  4. Keep steady for analysis  5. Solutions appear automatically for diseases"""
        
        instruction_frame = tk.Frame(content_frame, bg='#2c3e50')
        instruction_frame.pack(side='bottom', fill='x', pady=5)
        
        tk.Label(
            instruction_frame,
            text=instructions,
            font=('Arial', 9),
            bg='#2c3e50',
            fg='white',
            justify='left'
        ).pack()
    
    def initialize_everything(self):
        """Initialize camera, detection models, and solution model after UI is ready"""
        try:
            # Update status
            self.status_label.config(text="Starting camera...", fg='#f39c12')
            self.root.update()
            
            # Initialize camera
            self.init_camera()
            
            # Load detection models
            self.status_label.config(text="Loading AI models...", fg='#f39c12')
            self.root.update()
            
            print("[INFO] Loading Rice detection model...")
            self.rice_detector = DetectRice()
            
            print("[INFO] Loading Wheat detection model...")
            self.wheat_detector = DetectWheat()
            
            self.models_loaded = True
            print("[INFO] Detection models loaded successfully!")
            
            # Load solution generation model
            self.status_label.config(text="Loading solution model...", fg='#f39c12')
            self.solution_status.config(text="Loading...")
            self.root.update()
            
            print("[INFO] Loading GenerateSolution model...")
            self.solution_generator = GenerateSolution()
            
            # Initialize solution generator in background thread
            def init_solution_model():
                try:
                    success = self.solution_generator.process_documents()
                    if success:
                        self.solution_model_loaded = True
                        print("[INFO] Solution model loaded successfully!")
                        self.root.after(0, lambda: self.solution_status.config(text="✅ Ready", fg='#27ae60'))
                    else:
                        print("[ERROR] Failed to load solution model")
                        self.root.after(0, lambda: self.solution_status.config(text="❌ Failed", fg='#e74c3c'))
                except Exception as e:
                    print(f"[ERROR] Solution model loading failed: {e}")
                    self.root.after(0, lambda: self.solution_status.config(text="❌ Error", fg='#e74c3c'))
            
            threading.Thread(target=init_solution_model, daemon=True).start()
            
            # Enable buttons and start camera feed
            self.status_label.config(text="Ready for detection ✅", fg='#27ae60')
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
            self.cap = cv2.VideoCapture(0)
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
            self.status_label.config(text="Ready for detection ✅", fg='#27ae60')
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
        
        # Add to history if it's a new disease detection
        if not self.is_healthy:
            self.add_to_disease_history(crop_type, result)
        
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
    
    def add_to_disease_history(self, crop_type, disease):
        """Add detected disease to history and generate solution"""
        # Create unique identifier for this disease-crop combination
        disease_key = f"{crop_type.lower()}_{disease.lower().replace(' ', '_')}"
        
        # Check if this is a new disease we haven't seen before
        existing_disease = next((d for d in self.disease_history if d['key'] == disease_key), None)
        
        if not existing_disease:
            # New disease detected
            detection_info = {
                'key': disease_key,
                'crop': crop_type,
                'disease': disease,
                'timestamp': datetime.now(),
                'count': 1
            }
            self.disease_history.append(detection_info)
            
            # Generate solution for this new disease
            self.generate_solution_for_disease(crop_type, disease, disease_key)
        else:
            # Update count and timestamp for existing disease
            existing_disease['count'] += 1
            existing_disease['timestamp'] = datetime.now()
        
        # Update the solutions display
        self.update_solutions_display()
    
    def generate_solution_for_disease(self, crop, disease, disease_key):
        """Generate solution for detected disease in background thread"""
        if not self.solution_model_loaded:
            self.disease_solutions[disease_key] = {
                'status': 'Model not loaded',
                'solution': 'Solution generation model is still loading. Please wait.',
                'timestamp': datetime.now()
            }
            self.update_solutions_display()
            return
        
        if self.generating_solution:
            return  # Avoid multiple simultaneous solution generations
        
        def generate_in_background():
            self.generating_solution = True
            self.root.after(0, lambda: self.solution_status.config(text="🔄 Generating...", fg='#f39c12'))
            
            try:
                # Add placeholder while generating
                self.disease_solutions[disease_key] = {
                    'status': 'Generating',
                    'solution': 'Generating solution, please wait...',
                    'timestamp': datetime.now()
                }
                self.root.after(0, self.update_solutions_display)
                
                # Generate solution using the GenerateSolution class
                print(f"[INFO] Generating solution for {crop} - {disease}")
                
                # Capture the output by temporarily redirecting stdout
                import io
                import contextlib
                
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    self.solution_generator.generate_with_context(crop.lower(), disease.lower())
                
                # Get the solution from the response
                solution_text = self.solution_generator.response.get("answer", "No solution found in context.")
                
                # Store the solution
                self.disease_solutions[disease_key] = {
                    'status': 'Generated',
                    'solution': solution_text,
                    'timestamp': datetime.now()
                }
                
                print(f"[INFO] Solution generated for {disease}")
                self.root.after(0, lambda: self.solution_status.config(text="✅ Ready", fg='#27ae60'))
                
            except Exception as e:
                print(f"[ERROR] Solution generation failed: {e}")
                self.disease_solutions[disease_key] = {
                    'status': 'Error',
                    'solution': f'Error generating solution: {str(e)}',
                    'timestamp': datetime.now()
                }
                self.root.after(0, lambda: self.solution_status.config(text="❌ Error", fg='#e74c3c'))
            
            finally:
                self.generating_solution = False
                self.root.after(0, self.update_solutions_display)
        
        # Start generation in background thread
        threading.Thread(target=generate_in_background, daemon=True).start()
    
    def format_solution_text(self, raw_solution):
        """Format the raw solution text for better readability"""
        if not raw_solution or raw_solution.strip() == "":
            return "No solution available."
        
        # Clean up the text
        formatted_text = raw_solution.strip()
        
        # Replace overly long separator lines with cleaner ones
        lines = formatted_text.split('\n')
        processed_lines = []
        
        for line in lines:
            original_line = line
            line = line.strip()
            
            if not line:
                processed_lines.append('')
                continue
            
            # Handle very long separator lines (like the one in your example)
            if len(line) > 100 and all(c in '─-_:' for c in line.replace(' ', '')):
                processed_lines.append('─' * 60)  # Replace with cleaner separator
                continue
            
            # Keep original bullet points and formatting
            if line.startswith(('•', '-', '*')):
                processed_lines.append(original_line)  # Preserve original spacing
                continue
            
            # Keep numbered lists
            if line.startswith(tuple(f"{i}." for i in range(1, 10))):
                processed_lines.append(original_line)
                continue
            
            # Handle section headers (lines that are all caps or end with colon)
            if (line.isupper() and len(line) > 10) or (line.endswith(':') and len(line.split()) <= 6):
                if not processed_lines or processed_lines[-1] != '':
                    processed_lines.append('')  # Add space before headers
                processed_lines.append(original_line)
                continue
            
            # Handle HTML-like tags (like <br>)
            if '<br>' in line:
                line = line.replace('<br>', '\n')
                for subline in line.split('\n'):
                    if subline.strip():
                        processed_lines.append(subline)
                continue
            
            # For tables with colons as separators, improve formatting
            if ' : ' in line and not line.startswith('http'):
                parts = line.split(' : ', 1)
                if len(parts) == 2:
                    label = parts[0].strip()
                    content = parts[1].strip()
                    # Format as a clean key-value pair
                    formatted_line = f"{label:<30} : {content}"
                    processed_lines.append(formatted_line)
                    continue
            
            # Keep everything else as is, but ensure consistent indentation
            processed_lines.append(original_line)
        
        return '\n'.join(processed_lines)

    def update_solutions_display(self):
        """Update the solutions display with all disease history and solutions"""
        self.solutions_text.config(state='normal')
        self.solutions_text.delete(1.0, tk.END)
        
        if not self.disease_history:
            welcome_text = """
🌾 Welcome to Disease Solution Center

No diseases detected yet.

📋 How it works:
  • When a disease is detected, it will appear here
  • Solutions are automatically generated from official sources
  • History is maintained for all detected diseases
  • Each disease shows treatment recommendations

🔍 Start detecting to see solutions here!
            """.strip()
            self.solutions_text.insert(tk.END, welcome_text)
        else:
            # Sort by most recent first
            sorted_history = sorted(self.disease_history, key=lambda x: x['timestamp'], reverse=True)
            
            for i, disease_info in enumerate(sorted_history):
                disease_key = disease_info['key']
                
                # Disease header with better formatting
                header = f"\n{'=' * 60}\n"
                header += f"🔍 DETECTION #{i+1} - {disease_info['disease'].replace('_', ' ').title()}\n"
                header += f"{'=' * 60}\n"
                header += f"🌾 Crop Type       : {disease_info['crop']}\n"
                header += f"📅 Last Detected  : {disease_info['timestamp'].strftime('%d/%m/%Y at %H:%M:%S')}\n"
                header += f"🔢 Detection Count : {disease_info['count']} time(s)\n"
                header += f"{'─' * 60}\n"
                
                self.solutions_text.insert(tk.END, header)
                
                # Solution content with better formatting
                if disease_key in self.disease_solutions:
                    solution_info = self.disease_solutions[disease_key]
                    status = solution_info['status']
                    
                    if status == 'Generating':
                        self.solutions_text.insert(tk.END, "\n🔄 GENERATING SOLUTION...\n\nPlease wait while we fetch the best treatment recommendations from official agricultural sources.\n\n")
                    elif status == 'Error':
                        self.solutions_text.insert(tk.END, f"\n❌ ERROR GENERATING SOLUTION\n\n{solution_info['solution']}\n\nPlease try detecting the disease again.\n\n")
                    elif status == 'Model not loaded':
                        self.solutions_text.insert(tk.END, f"\n⏳ SOLUTION MODEL LOADING\n\n{solution_info['solution']}\n\n")
                    else:  # Generated
                        formatted_solution = self.format_solution_text(solution_info['solution'])
                        self.solutions_text.insert(tk.END, f"\n💡 TREATMENT SOLUTION:\n{'─' * 40}\n{formatted_solution}\n\n")
                        
                        # Add a helpful footer
                        footer = "📌 Important Notes:\n"
                        footer += "  • Follow all safety guidelines when applying treatments\n"
                        footer += "  • Consult with local agricultural extension officers\n"
                        footer += "  • Monitor plant response after treatment\n"
                        self.solutions_text.insert(tk.END, f"{footer}\n")
                else:
                    self.solutions_text.insert(tk.END, "\n⏳ SOLUTION PENDING\n\nSolution will be generated shortly using official agricultural guidelines.\n\n")
                
                self.solutions_text.insert(tk.END, f"{'=' * 60}\n\n")
        
        self.solutions_text.config(state='disabled')
        # Auto-scroll to top to show latest detection
        self.solutions_text.see(1.0)
    
    def clear_history(self):
        """Clear disease history and solutions"""
        if messagebox.askyesno("Clear History", "Are you sure you want to clear all disease history and solutions?"):
            self.disease_history.clear()
            self.disease_solutions.clear()
            self.update_solutions_display()
            print("[INFO] Disease history and solutions cleared")
    
    def update_results_display(self, result, crop_type):
        """Update the current results text display"""
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
            result_info += "\n⚠️ Disease detected!\nSolution will be generated\nand shown in the side panel."
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
    print("[INFO] Starting Minori AI with Solution Generation...")
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