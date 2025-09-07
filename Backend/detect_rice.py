from tensorflow.keras.models import load_model
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image


class DetectRice():
    def __init__(self):
        self.model = load_model("Backend\\Rice Disease Model and Classes\\rice_disease_model.h5", compile=False)

        with open("Backend\\Rice Disease Model and Classes\\rice_class_indices.pkl", "rb") as f:
            class_indices = pickle.load(f)

        self.class_labels = list(class_indices.keys())
        

    def detect_rice_disease(self, frame):


        img = image.load_img(frame, target_size=(128,128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = self.model.predict(img_array)
        predicted_class = self.class_labels[np.argmax(prediction)]

        print("Predicted:", predicted_class)
        return predicted_class
    

obj = DetectRice()
result = obj.detect_rice_disease("D:\\Namit Kumar\\Hackathons\\Project Exhibition 2025\\Minori AI\\Backend\\rice.jpg")