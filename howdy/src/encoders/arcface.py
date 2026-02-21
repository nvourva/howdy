import cv2
import numpy as np
import os

class ArcFaceEncoder:
    """Face encoder using ArcFace ONNX model."""
    
    def __init__(self, model_path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
    def preprocess(self, frame, landmarks):
        """Align and crop face based on landmarks."""
        # For now, we'll do a simple crop. 
        # In a full implementation, we would use the 5-point landmarks to warp the face.
        x, y, w, h = landmarks.rect.left(), landmarks.rect.top(), landmarks.rect.width(), landmarks.rect.height()
        face_img = frame[max(0, y):y+h, max(0, x):x+w]
        face_img = cv2.resize(face_img, (112, 112))
        
        # Normalize
        face_img = face_img.astype(np.float32)
        face_img = (face_img / 255.0 - 0.5) / 0.5
        face_img = np.transpose(face_img, (2, 0, 1))
        face_img = np.expand_dims(face_img, axis=0)
        return face_img

    def encode(self, frame, landmarks):
        """Generate 512D embedding."""
        blob = self.preprocess(frame, landmarks)
        net_out = self.session.run(None, {self.input_name: blob})
        embeddings = net_out[0]
        
        # L2 Normalize
        norm = np.linalg.norm(embeddings)
        if norm > 1e-6:
            embeddings = embeddings / norm
            
        return embeddings.flatten()
