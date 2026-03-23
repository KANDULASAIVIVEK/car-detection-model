import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import threading
import os
from detector import TrafficDetector

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Car Color & Person Counter")
        self.root.geometry("1000x800")
        
        self.detector = None
        self.image_path = None
        
        self.setup_ui()
        
        # Load model in background to not freeze UI
        threading.Thread(target=self.init_detector, daemon=True).start()
        
    def init_detector(self):
        try:
            self.detector = TrafficDetector()
            def on_success():
                self.status_var.set("Model loaded successfully. Ready.")
                self.btn_run.config(state=tk.NORMAL)
            self.root.after(0, on_success)
        except Exception as e:
            err_msg = f"Error loading model: {str(e)}"
            self.root.after(0, lambda: self.status_var.set(err_msg))
            
    def setup_ui(self):
        # Top Frame for controls
        top_frame = tk.Frame(self.root, pady=10)
        top_frame.pack(fill=tk.X)
        
        self.btn_upload = tk.Button(top_frame, text="Upload Image", command=self.upload_image, width=15)
        self.btn_upload.pack(side=tk.LEFT, padx=10)
        
        self.btn_run = tk.Button(top_frame, text="Run Detection", command=self.run_detection, state=tk.DISABLED, width=15)
        self.btn_run.pack(side=tk.LEFT, padx=10)
        
        # Status Label
        self.status_var = tk.StringVar()
        self.status_var.set("Loading YOLOv8 model, please wait...")
        self.lbl_status = tk.Label(top_frame, textvariable=self.status_var, fg="blue")
        self.lbl_status.pack(side=tk.LEFT, padx=20)
        
        # Info Frame for counts
        info_frame = tk.Frame(self.root, pady=10)
        info_frame.pack(fill=tk.X)
        
        font_large = ("Helvetica", 12, "bold")
        
        self.lbl_total_cars = tk.Label(info_frame, text="Total Cars: 0", font=font_large, fg="black")
        self.lbl_total_cars.pack(side=tk.LEFT, padx=15)
        
        self.lbl_people = tk.Label(info_frame, text="People: 0", font=font_large, fg="green")
        self.lbl_people.pack(side=tk.LEFT, padx=15)
        
        self.lbl_colors = tk.Label(info_frame, text="Colors: -", font=("Helvetica", 11), fg="purple")
        self.lbl_colors.pack(side=tk.LEFT, padx=15)
        
        # Canvas Frame for image display
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Traffic Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.image_path = file_path
            self.display_image(self.image_path)
            self.status_var.set(f"Loaded: {os.path.basename(self.image_path)}")
            
    def display_image(self, img_source):
        # img_source can be a path (string) or a cv2 image (numpy array)
        if isinstance(img_source, str):
            with Image.open(img_source) as tmp_img:
                img = tmp_img.copy()
        else:
            # Convert OpenCV BGR to RGB for PIL
            img = cv2.cvtColor(img_source, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            
        # Resize image to fit canvas while maintaining aspect ratio
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Fallback if canvas is not drawn yet
        if canvas_width <= 1:
            canvas_width, canvas_height = 960, 600
            
        img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        
        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        
        # Center image
        x_centered = (canvas_width - img.width) // 2
        y_centered = (canvas_height - img.height) // 2
        
        self.canvas.create_image(x_centered, y_centered, anchor=tk.NW, image=self.tk_image)
        
    def run_detection(self):
        if not self.image_path:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return
            
        if not self.detector:
            messagebox.showwarning("Model Loading", "Please wait for the model to finish loading.")
            return
            
        self.status_var.set("Running detection...")
        self.root.update()
        
        try:
            processed_img, counts = self.detector.process_image(self.image_path)
            
            # Update counts UI
            self.lbl_total_cars.config(text=f"Total Cars: {counts['total_cars']}")
            self.lbl_people.config(text=f"People: {counts['people']}")
            
            # Format color counts string (exclude zeros)
            color_str = ", ".join([f"{k}: {v}" for k, v in counts['color_counts'].items() if v > 0])
            if not color_str:
                color_str = "-"
            self.lbl_colors.config(text=f"Colors: {color_str}")
            
            # Display processed image
            self.display_image(processed_img)
            self.status_var.set("Detection complete.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed during detection: {str(e)}")
            self.status_var.set("Detection failed.")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
