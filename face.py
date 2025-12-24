from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.filechooser import FileChooserIconView
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.logger import Logger

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import os

from keras_facenet import FaceNet

class CamApp(App):

    def build(self):
        self.web_cam = Image(size_hint=(1, .8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1, .1))
        self.verification_label = Label(text="Verification uninitiated", size_hint=(1, .1))
   
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        self.facenet = FaceNet()
        self.mtcnn = MTCNN()

        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)

        return layout
    def update(self, *args):

        ret, frame = self.capture.read()
        frame = frame[350:350+250,500:500+250, :]

        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt='bgr'
            )
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    def get_face_embedding(self, img_path):
        """Extract face embedding using FaceNet"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                return None
            
            detections = self.mtcnn.detect_faces(img)
            if len(detections) == 0:
                return None
            
            best_detection = max(detections, key=lambda x: x['confidence'])
            x, y, w, h = best_detection['box']
            x, y = max(0, x), max(0, y)
            face = img[y:y+h, x:x+w]
            
            face_resized = cv2.resize(face, (160, 160))
            face_resized = face_resized.astype('float32')
            
            embedding = self.facenet.embeddings([face_resized])[0]
            return embedding
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    def verify(self, *args):
        similarity_threshold = 0.6

        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[350:350+250, 500:500+250, :]
        cv2.imwrite(SAVE_PATH, frame)
        
        input_embedding = self.get_face_embedding(SAVE_PATH)
        if input_embedding is None:
            self.verification_label.text = "Unverified (No face detected)"
            return [], False
        
        verification_images_path = os.path.join('application_data', 'verification_images')
        verification_images = os.listdir(verification_images_path)
        
        if len(verification_images) == 0:
            self.verification_label.text = "Unverified (No verification images)"
            return [], False
        
        results = []
        for image in verification_images:
            val_embedding = self.get_face_embedding(os.path.join(verification_images_path, image))
            if val_embedding is None:
                continue
            
            similarity = np.dot(input_embedding, val_embedding) / (
                np.linalg.norm(input_embedding) * np.linalg.norm(val_embedding)
            )
            results.append(similarity)
        
        if len(results) == 0:
            self.verification_label.text = "Unverified (No valid faces in verification images)"
            return [], False
        
        matches = np.sum(np.array(results) > similarity_threshold)
        verification = matches / len(results)
        verified = verification > 0.5

        print("Similarities: ", results)
        print("Matches: ", matches)
        print("Verification: ", verification)
        print("Verified: ", verified)

        self.verification_label.text = "Verified" if verified else "Unverified"
    
        return results, verified

if __name__ == '__main__':
    CamApp().run()