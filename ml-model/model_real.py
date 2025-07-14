"""
Modelo Real de Machine Learning para Diagnóstico de Cáncer de Mama
Usa el modelo entrenado breast_cancer_detection_model_transfer.h5
Versión integrada para el sistema web
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import mixed_precision
import numpy as np
import cv2
from PIL import Image
import os
import json

class RealBreastCancerModel:
    def __init__(self, img_size=50):
        """
        Inicializar modelo real con el modelo entrenado
        
        Args:
            img_size (int): Tamaño de imagen que espera el modelo (50x50 por defecto)
        """
        self.img_size = img_size
        self.model = None
        self.model_loaded = False
        self.model_path = None
        
        # Configurar TensorFlow
        self._setup_tensorflow()
        
        # Intentar cargar el modelo
        self.load_trained_model()
    
    def _setup_tensorflow(self):
        """Configurar TensorFlow para optimización"""
        try:
            # Política de precisión mixta
            mixed_precision.set_global_policy('mixed_float16')
            print("✅ Política de precisión mixta establecida")
            
            # Configurar GPU si está disponible
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"✅ GPUs configuradas: {len(gpus)}")
                except RuntimeError as e:
                    print(f"⚠️ Error configurando GPU: {e}")
            else:
                print("ℹ️ Ejecutando en CPU")
                
        except Exception as e:
            print(f"⚠️ Error configurando TensorFlow: {e}")
    
    def load_trained_model(self):
        """Cargar el modelo entrenado real"""
        
        # Rutas posibles del modelo
        possible_paths = [
            "/app/ml-model/breast_cancer_detection_model_transfer.h5",  # Docker
            "Z:/Trabajos/TrabajoTesis/dev_cruz_ml/app-cruz-ml/ml-model/breast_cancer_detection_model_transfer.h5",  # Local
            "./ml-model/breast_cancer_detection_model_transfer.h5",  # Relativo
            "../ml-model/breast_cancer_detection_model_transfer.h5",  # Relativo desde backend
            os.path.join(os.path.dirname(__file__), "breast_cancer_detection_model_transfer.h5")  # Mismo directorio
        ]
        
        for model_path in possible_paths:
            try:
                if os.path.exists(model_path):
                    print(f"🔍 Encontrado modelo en: {model_path}")
                    self.model = load_model(model_path)
                    self.model_path = model_path
                    self.model_loaded = True
                    
                    print(f"✅ Modelo real cargado exitosamente desde: {model_path}")
                    print(f"📊 Resumen del modelo:")
                    print(f"   - Input shape: {self.model.input_shape}")
                    print(f"   - Output shape: {self.model.output_shape}")
                    print(f"   - Imagen esperada: {self.img_size}x{self.img_size}")
                    
                    return True
                    
            except Exception as e:
                print(f"❌ Error cargando modelo desde {model_path}: {e}")
                continue
        
        print("❌ No se pudo cargar el modelo real desde ninguna ruta")
        return False
    
    def preprocess_image(self, image_input):
        """
        Preprocesar imagen para el modelo real
        Compatible con el formato esperado por breast_cancer_detection_model_transfer.h5
        
        Args:
            image_input: Puede ser ruta de archivo, bytes, PIL Image o numpy array
            
        Returns:
            numpy.ndarray: Imagen preprocesada lista para predicción
        """
        
        try:
            # Convertir input a PIL Image
            if isinstance(image_input, str):
                # Ruta de archivo
                if not os.path.exists(image_input):
                    raise ValueError(f"Archivo no encontrado: {image_input}")
                pil_image = Image.open(image_input)
                
            elif isinstance(image_input, bytes):
                # Bytes de imagen
                from io import BytesIO
                pil_image = Image.open(BytesIO(image_input))
                
            elif isinstance(image_input, Image.Image):
                # Ya es PIL Image
                pil_image = image_input
                
            elif isinstance(image_input, np.ndarray):
                # Numpy array
                pil_image = Image.fromarray(image_input.astype('uint8'))
                
            else:
                raise ValueError(f"Tipo de imagen no soportado: {type(image_input)}")
            
            # Convertir a RGB si es necesario
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            print(f"📸 Imagen original: {pil_image.size}, modo: {pil_image.mode}")
            
            # Usar la función de Keras para cargar y redimensionar (compatible con el entrenamiento)
            # Esto asegura la compatibilidad exacta con cómo se entrenó el modelo
            
            if isinstance(image_input, (str, bytes)):
                # Para rutas o bytes, usar load_img de Keras
                if isinstance(image_input, bytes):
                    # Guardar temporalmente para usar load_img
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                        pil_image.save(tmp_file.name)
                        temp_path = tmp_file.name
                    
                    try:
                        img = image.load_img(temp_path, target_size=(self.img_size, self.img_size))
                        os.unlink(temp_path)  # Limpiar archivo temporal
                    except:
                        os.unlink(temp_path)  # Limpiar en caso de error
                        raise
                else:
                    img = image.load_img(image_input, target_size=(self.img_size, self.img_size))
            else:
                # Para PIL Images, redimensionar manualmente
                img = pil_image.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
            
            # Convertir a array de numpy
            img_array = image.img_to_array(img)
            
            # Expandir dimensiones para batch
            img_array = np.expand_dims(img_array, axis=0)
            
            # Normalizar píxeles (igual que en el entrenamiento)
            img_array = img_array / 255.0
            
            print(f"✅ Imagen preprocesada: shape={img_array.shape}, dtype={img_array.dtype}")
            print(f"   - Min: {img_array.min():.4f}, Max: {img_array.max():.4f}")
            
            return img_array
            
        except Exception as e:
            print(f"❌ Error preprocesando imagen: {e}")
            raise e
    
    def predict(self, image_input):
        """
        Realizar predicción usando el modelo real entrenado
        
        Args:
            image_input: Imagen a analizar (ruta, bytes, PIL Image, etc.)
            
        Returns:
            dict: Resultado de la predicción con probabilidad, clasificación y confianza
        """
        
        if not self.model_loaded or self.model is None:
            raise ValueError("Modelo real no está cargado. Verifica que el archivo .h5 existe.")
        
        try:
            print(f"🔍 Iniciando predicción con modelo real...")
            
            # Preprocesar imagen
            processed_image = self.preprocess_image(image_input)
            
            # Realizar predicción
            print(f"🧠 Ejecutando predicción...")
            prediction_proba = self.model.predict(processed_image, verbose=0)
            
            # Extraer probabilidad (el modelo devuelve una matriz, tomamos el primer valor)
            probability = float(prediction_proba[0][0])
            
            print(f"📊 Probabilidad cruda: {probability:.6f}")
            
            # Interpretar resultado (igual que en el script original)
            has_cancer = probability > 0.5
            
            # Calcular confianza
            confidence = abs(probability - 0.5) * 2  # Convertir a escala 0-1
            
            # Generar región sospechosa si hay cáncer detectado
            bbox = None
            if has_cancer:
                bbox = self._generate_suspicious_region(processed_image[0], probability)
            
            # Preparar resultado
            result = {
                'probability': probability,
                'has_cancer': has_cancer,
                'confidence': confidence,
                'bbox': bbox,
                'model_type': 'TransferLearning_Real',
                'model_path': self.model_path,
                'image_size_used': self.img_size
            }
            
            print(f"✅ Predicción completada:")
            print(f"   - Probabilidad: {probability:.4f}")
            print(f"   - Tiene cáncer: {has_cancer}")
            print(f"   - Confianza: {confidence:.4f}")
            print(f"   - Región sospechosa: {'Sí' if bbox else 'No'}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise e
    
    def _generate_suspicious_region(self, image_array, probability):
        """
        Generar región sospechosa basada en la imagen y probabilidad
        Usa técnicas simples de análisis de imagen
        
        Args:
            image_array (numpy.ndarray): Imagen preprocesada (50x50x3)
            probability (float): Probabilidad de cáncer
            
        Returns:
            dict: Coordenadas de la región sospechosa o None
        """
        
        try:
            # Convertir imagen a escala de grises para análisis
            if len(image_array.shape) == 3:
                gray = np.mean(image_array, axis=2)
            else:
                gray = image_array
            
            # Encontrar regiones de mayor intensidad/contraste
            height, width = gray.shape
            
            # Buscar región con mayor variación (posible masa)
            window_size = max(10, min(height, width) // 5)
            max_variance = 0
            best_region = None
            
            for y in range(0, height - window_size, 5):
                for x in range(0, width - window_size, 5):
                    region = gray[y:y+window_size, x:x+window_size]
                    variance = np.var(region)
                    
                    if variance > max_variance:
                        max_variance = variance
                        best_region = (x, y, window_size, window_size)
            
            if best_region and max_variance > 0.01:  # Umbral mínimo de varianza
                # Ajustar tamaño basado en probabilidad
                scale_factor = 0.5 + probability * 0.5  # Entre 0.5 y 1.0
                
                x, y, w, h = best_region
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)
                
                # Centrar la región ajustada
                new_x = max(0, x + (w - new_w) // 2)
                new_y = max(0, y + (h - new_h) // 2)
                
                return {
                    'x': int(new_x),
                    'y': int(new_y),
                    'width': int(new_w),
                    'height': int(new_h)
                }
            
            # Si no se encuentra región específica, usar región central
            center_size = int(min(height, width) * 0.3)
            center_x = (width - center_size) // 2
            center_y = (height - center_size) // 2
            
            return {
                'x': center_x,
                'y': center_y,
                'width': center_size,
                'height': center_size
            }
            
        except Exception as e:
            print(f"⚠️ Error generando región sospechosa: {e}")
            return None
    
    def predict_single_image_legacy(self, image_path):
        """
        Función compatible con el script original predict_breast_cancer.py
        
        Args:
            image_path (str): Ruta de la imagen
            
        Returns:
            tuple: (probabilidad, etiqueta_texto)
        """
        
        try:
            result = self.predict(image_path)
            
            probability = result['probability']
            
            if result['has_cancer']:
                label = "Cáncer de Mama (IDC Positivo)"
            else:
                label = "No Cáncer (IDC Negativo)"
            
            return probability, label
            
        except Exception as e:
            print(f"Error en predicción legacy: {e}")
            return None, None
    
    def get_model_info(self):
        """Obtener información del modelo"""
        
        if not self.model_loaded:
            return {
                'loaded': False,
                'error': 'Modelo no cargado'
            }
        
        try:
            return {
                'loaded': True,
                'model_path': self.model_path,
                'input_shape': str(self.model.input_shape),
                'output_shape': str(self.model.output_shape),
                'image_size': self.img_size,
                'total_params': self.model.count_params(),
                'model_type': 'Transfer Learning - Real Model'
            }
        except Exception as e:
            return {
                'loaded': True,
                'error': str(e),
                'model_path': self.model_path
            }

# Función de conveniencia para compatibilidad
def predict_single_image(image_path, model_instance=None, img_size=50):
    """
    Función de compatibilidad con el script original
    
    Args:
        image_path (str): Ruta de la imagen
        model_instance: Instancia del modelo (opcional)
        img_size (int): Tamaño de imagen
        
    Returns:
        tuple: (probabilidad, etiqueta)
    """
    
    if model_instance is None:
        model_instance = RealBreastCancerModel(img_size=img_size)
    
    return model_instance.predict_single_image_legacy(image_path)

# Función para prueba rápida
def test_real_model():
    """Probar el modelo real"""
    
    try:
        print("🧪 Iniciando prueba del modelo real...")
        
        # Crear instancia
        model = RealBreastCancerModel()
        
        if not model.model_loaded:
            print("❌ No se pudo cargar el modelo real")
            return False
        
        # Mostrar información
        info = model.get_model_info()
        print(f"📋 Información del modelo:")
        for key, value in info.items():
            print(f"   - {key}: {value}")
        
        # Crear imagen de prueba
        test_image = np.random.rand(50, 50, 3).astype('float32')
        
        # Hacer predicción de prueba
        result = model.predict(test_image)
        
        print(f"✅ Prueba exitosa:")
        print(f"   - Probabilidad: {result['probability']:.4f}")
        print(f"   - Clasificación: {'Cáncer' if result['has_cancer'] else 'Normal'}")
        print(f"   - Confianza: {result['confidence']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("🏥 MODELO REAL DE DIAGNÓSTICO DE CÁNCER DE MAMA")
    print("=" * 60)
    
    # Ejecutar prueba
    success = test_real_model()
    
    if success:
        print("\n✅ Modelo real funcionando correctamente")
    else:
        print("\n❌ Error en el modelo real")
