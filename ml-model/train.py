"""
Script para entrenar el modelo con datos reales
Incluye funciones para detección de regiones sospechosas
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from model import BreastCancerModel
import matplotlib.pyplot as plt

class RegionDetector:
    """Clase para detectar regiones sospechosas en mamografías"""
    
    def __init__(self):
        self.model = None
    
    def detect_suspicious_regions(self, image, prediction_score, threshold=0.7):
        """
        Detectar regiones sospechosas usando técnicas de computer vision
        Esta es una implementación simplificada - en un caso real usarías
        técnicas más avanzadas como Grad-CAM o modelos de detección de objetos
        """
        
        if prediction_score < threshold:
            return None
        
        # Convertir a grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Aplicar filtros para resaltar regiones anómalas
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detectar bordes
        edges = cv2.Canny(blurred, 50, 150)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Encontrar el contorno más grande (simplificación)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Obtener bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Filtrar regiones muy pequeñas
        if w < 30 or h < 30:
            return None
        
        return {
            'x': int(x),
            'y': int(y),
            'width': int(w),
            'height': int(h),
            'area': int(w * h)
        }
    
    def visualize_detection(self, image, bbox, save_path=None):
        """Visualizar la detección con bounding box"""
        
        if bbox is None:
            return image
        
        # Crear copia de la imagen
        vis_image = image.copy()
        
        # Dibujar bounding box
        cv2.rectangle(
            vis_image,
            (bbox['x'], bbox['y']),
            (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']),
            (255, 0, 0),  # Color rojo
            3
        )
        
        # Agregar texto
        cv2.putText(
            vis_image,
            'Región Sospechosa',
            (bbox['x'], bbox['y'] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image

def load_mammography_dataset(data_dir):
    """
    Cargar dataset de mamografías
    Estructura esperada:
    data_dir/
    ├── benign/
    │   ├── image1.jpg
    │   ├── image2.jpg
    └── malignant/
        ├── image1.jpg
        ├── image2.jpg
    """
    
    images = []
    labels = []
    
    # Cargar imágenes benignas
    benign_dir = os.path.join(data_dir, 'benign')
    if os.path.exists(benign_dir):
        for filename in os.listdir(benign_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(benign_dir, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    labels.append(0)  # 0 = benigno
    
    # Cargar imágenes malignas
    malignant_dir = os.path.join(data_dir, 'malignant')
    if os.path.exists(malignant_dir):
        for filename in os.listdir(malignant_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(malignant_dir, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    labels.append(1)  # 1 = maligno
    
    return np.array(images), np.array(labels)

def preprocess_images(images, target_size=(224, 224)):
    """Preprocesar imágenes para el modelo"""
    
    processed = []
    for img in images:
        # Redimensionar
        resized = cv2.resize(img, target_size)
        # Normalizar
        normalized = resized.astype('float32') / 255.0
        processed.append(normalized)
    
    return np.array(processed)

def train_model_with_real_data(data_dir, model_save_path='trained_breast_cancer_model.h5'):
    """Entrenar modelo con datos reales"""
    
    print("Cargando dataset...")
    images, labels = load_mammography_dataset(data_dir)
    
    if len(images) == 0:
        print("No se encontraron imágenes. Creando modelo con datos sintéticos...")
        # Usar datos sintéticos si no hay datos reales
        from model import create_synthetic_data
        images, labels = create_synthetic_data(1000)
    else:
        print(f"Dataset cargado: {len(images)} imágenes")
        # Preprocesar imágenes reales
        images = preprocess_images(images)
    
    # Dividir dataset
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Crear y entrenar modelo
    print("Creando modelo...")
    model = BreastCancerModel()
    
    print("Entrenando modelo...")
    history = model.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=16,
        verbose=1
    )
    
    # Evaluar modelo
    print("Evaluando modelo...")
    test_loss, test_acc, test_prec, test_rec = model.model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_rec:.4f}")
    
    # Guardar modelo
    model.save_model(model_save_path)
    
    return model, history

def test_complete_pipeline():
    """Probar pipeline completo incluyendo detección de regiones"""
    
    print("Probando pipeline completo...")
    
    # Crear modelo
    model = BreastCancerModel()
    
    # Crear detector de regiones
    detector = RegionDetector()
    
    # Crear imagen de prueba
    test_image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    
    # Realizar predicción
    result = model.predict(test_image)
    print(f"Predicción: {result}")
    
    # Detectar regiones sospechosas
    bbox = detector.detect_suspicious_regions(test_image, result['probability'])
    print(f"Región detectada: {bbox}")
    
    # Visualizar resultado
    if bbox:
        vis_image = detector.visualize_detection(test_image, bbox)
        print("Imagen con detección creada")
    
    return model, detector

if __name__ == "__main__":
    # Para datos reales, descomenta la siguiente línea y proporciona la ruta
    # train_model_with_real_data('/path/to/mammography/dataset')
    
    # Para prueba con datos sintéticos
    test_complete_pipeline()
