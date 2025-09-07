from tensorflow.keras.models import load_model
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image


class DetectWheat():
    def __init__(self):
        self.model = load_model("Wheat Disease Model and Classes\\wheat_disease_model.h5")

        with open("Wheat Disease Model and Classes\\wheat_class_indices.pkl", "rb") as f:
            class_indices = pickle.load(f)

        self.class_labels = list(class_indices.keys())
        

    def detect_wheat_disease(self, frame):


        img = image.load_img(frame, target_size=(128,128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = self.model.predict(img_array)
        predicted_class = self.class_labels[np.argmax(prediction)]

        print("Predicted:", predicted_class)
        return predicted_class
    

obj = DetectWheat()
result = obj.detect_wheat_disease("D:\\Namit Kumar\\Hackathons\\Project Exhibition 2025\\Minori AI\\Backend\\rice.jpg")