"""
Adaptación DIRECTA de tu script que funciona
Solo cambiamos para recibir imagen desde web en lugar de ruta
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import mixed_precision
from PIL import Image
import io
import tempfile

class DirectWebModel:
    def __init__(self):
        """Inicializar usando exactamente tu configuración que funciona"""
        
        # Tu configuración exacta
        mixed_precision.set_global_policy('mixed_float16')
        print("Política de precisión mixta establecida a 'mixed_float16'.")
        
        # Verificar GPU (tu código exacto)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPUs detectadas y configuradas: {gpus}")
            except RuntimeError as e:
                print(e)
        else:
            print("No se detectaron GPUs. La predicción se realizará en CPU.")
        
        # Rutas del modelo (tu configuración)
        self.model_paths = [
            "/app/ml-model/breast_cancer_detection_model_transfer.h5",  # Docker
            "Z:/Trabajos/TrabajoTesis/dev_cruz_ml/app-cruz-ml/ml-model/breast_cancer_detection_model_transfer.h5",  # Local
            "../ml-model/breast_cancer_detection_model_transfer.h5",
            "./ml-model/breast_cancer_detection_model_transfer.h5"
        ]
        
        self.IMG_SIZE = 50  # Tu tamaño exacto
        self.model = None
        self.model_loaded = False
        self.model_path = None
        
        # Cargar modelo usando tu método exacto
        self.load_model()
    
    def load_model(self):
        """Cargar modelo usando exactamente tu código CON manejo de batch_shape"""
        
        for model_path in self.model_paths:
            if os.path.exists(model_path):
                try:
                    print(f"Intentando cargar modelo desde: {model_path}")
                    
                    # MÉTODO 1: Cargar directamente
                    try:
                        self.model = load_model(model_path)
                        self.model_path = model_path
                        self.model_loaded = True
                        print(f"✅ Modelo cargado exitosamente desde: {model_path}")
                        self.model.summary()  # Tu línea exacta
                        return True
                    except Exception as e:
                        if 'batch_shape' in str(e):
                            print(f"⚠️ Error de batch_shape detectado, intentando con compile=False...")
                            # MÉTODO 2: Cargar sin compilar
                            try:
                                self.model = load_model(model_path, compile=False)
                                # Recompilar manualmente
                                self.model.compile(
                                    optimizer='adam',
                                    loss='binary_crossentropy',
                                    metrics=['accuracy']
                                )
                                self.model_path = model_path
                                self.model_loaded = True
                                print(f"✅ Modelo cargado exitosamente (sin compilar) desde: {model_path}")
                                self.model.summary()
                                return True
                            except Exception as e2:
                                print(f"❌ También falló sin compilar: {e2}")
                                continue
                        else:
                            print(f"❌ Error cargando desde {model_path}: {e}")
                            continue
                            
                except Exception as e:
                    print(f"❌ Error general cargando desde {model_path}: {e}")
                    continue
        
        print("❌ No se pudo cargar el modelo desde ninguna ubicación")
        return False
    
    def predict_single_image_bytes(self, image_bytes):
        """
        Adaptación de tu función predict_single_image para trabajar con bytes
        Mantiene EXACTAMENTE tu lógica de predicción
        """
        
        if not self.model_loaded:
            print("❌ Modelo no cargado")
            return None, None
        
        try:
            # Convertir bytes a archivo temporal para usar tu código exacto
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                tmp_file.write(image_bytes)
                temp_path = tmp_file.name
            
            try:
                # TU CÓDIGO EXACTO desde aquí
                
                # Cargar la imagen y redimensionarla (TU LÍNEA EXACTA)
                img = image.load_img(temp_path, target_size=(self.IMG_SIZE, self.IMG_SIZE))
                
                # Convertir la imagen a un array de NumPy (TU LÍNEA EXACTA)
                img_array = image.img_to_array(img)
                
                # Expandir las dimensiones (TU LÍNEA EXACTA)
                img_array = np.expand_dims(img_array, axis=0)
                
                # Normalizar los valores de los píxeles (TU LÍNEA EXACTA)
                img_array /= 255.0

                # Realizar la predicción (TU LÍNEA EXACTA)
                prediction_proba = self.model.predict(img_array)[0][0]

                # Interpretar la predicción (TU CÓDIGO EXACTO)
                if prediction_proba > 0.5:
                    prediction_label = "Cáncer de Mama (IDC Positivo)"
                    has_cancer = True
                else:
                    prediction_label = "No Cáncer (IDC Negativo)"
                    has_cancer = False

                print(f"🔍 Predicción directa:")
                print(f"   - Probabilidad: {prediction_proba:.6f}")
                print(f"   - Clasificación: {prediction_label}")

                return prediction_proba, prediction_label, has_cancer

            finally:
                # Limpiar archivo temporal
                os.unlink(temp_path)
                
        except Exception as e:
            print(f"❌ Error procesando imagen: {e}")
            return None, None, None
    
    def predict(self, image_input):
        """
        Función para el sistema web que usa tu lógica exacta
        """
        
        try:
            # Usar tu función adaptada
            result = self.predict_single_image_bytes(image_input)
            
            if result[0] is not None:
                probability, label, has_cancer = result
                
                # Calcular confianza
                confidence = abs(probability - 0.5) * 2
                
                # Generar región sospechosa simple si hay cáncer
                bbox = None
                if has_cancer:
                    bbox = {
                        'x': 15,
                        'y': 15,
                        'width': 20,
                        'height': 20
                    }
                
                # Formato para el backend
                return {
                    'probability': float(probability),
                    'has_cancer': bool(has_cancer),
                    'confidence': float(confidence),
                    'bbox': bbox,
                    'model_type': 'DirectWebModel_YourScript',
                    'model_path': self.model_path,
                    'image_size_used': self.IMG_SIZE
                }
            else:
                raise Exception("Error en predicción")
                
        except Exception as e:
            print(f"❌ Error en predict: {e}")
            raise e
    
    def get_model_info(self):
        """Información del modelo"""
        if not self.model_loaded:
            return {
                'loaded': False,
                'error': 'Modelo no cargado'
            }
        
        return {
            'loaded': True,
            'model_path': self.model_path,
            'image_size': self.IMG_SIZE,
            'model_type': 'Direct Web Model - Tu Script Exacto',
            'tf_version': tf.__version__
        }

# Exportar para el sistema
DirectRealModel = DirectWebModel
RealBreastCancerModel = DirectWebModel

def test_direct_web_model():
    """Probar el modelo web directo"""
    try:
        print("🧪 Probando modelo web directo...")
        
        model = DirectWebModel()
        
        if not model.model_loaded:
            print("❌ Modelo no cargado")
            return False
        
        # Crear imagen de prueba
        test_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        pil_image = Image.fromarray(test_image)
        
        # Convertir a bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        image_bytes = img_byte_arr.getvalue()
        
        result = model.predict(image_bytes)
        
        print(f"✅ Resultado:")
        print(f"   - Probabilidad: {result['probability']:.6f}")
        print(f"   - Tiene cáncer: {result['has_cancer']}")
        print(f"   - Confianza: {result['confidence']:.4f}")
        print(f"   - Modelo: {result['model_type']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        return False

if __name__ == "__main__":
    print("🏥 MODELO WEB DIRECTO - USANDO TU SCRIPT EXACTO")
    print("=" * 60)
    
    success = test_direct_web_model()
    
    if success:
        print("\\n✅ Modelo web directo funcionando correctamente")
    else:
        print("\\n❌ Error en modelo web directo")
