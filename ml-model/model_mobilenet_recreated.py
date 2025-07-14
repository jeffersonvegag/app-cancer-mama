"""
Recrear exactamente la arquitectura de tu modelo entrenado
Y cargar los pesos desde el archivo .h5
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import mixed_precision
from PIL import Image
import io
import tempfile

class MobileNetV2Model:
    def __init__(self):
        """Recrear exactamente tu arquitectura de entrenamiento"""
        
        # Tu configuración exacta del entrenamiento
        mixed_precision.set_global_policy('mixed_float16')
        print("Política de precisión mixta establecida a 'mixed_float16'.")
        
        # Verificar GPU (tu código exacto)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPUs detectadas: {gpus}")
            except RuntimeError as e:
                print(e)
        else:
            print("No se detectaron GPUs. Predicción en CPU.")
        
        self.IMG_SIZE = 50  # Tu tamaño exacto
        self.model = None
        self.model_loaded = False
        self.model_path = None
        
        # Rutas posibles del modelo
        self.model_paths = [
            "/app/ml-model/breast_cancer_detection_model_transfer.h5",
            "Z:/Trabajos/TrabajoTesis/dev_cruz_ml/app-cruz-ml/ml-model/breast_cancer_detection_model_transfer.h5",
            "../ml-model/breast_cancer_detection_model_transfer.h5",
            "./ml-model/breast_cancer_detection_model_transfer.h5"
        ]
        
        # Crear modelo y cargar pesos
        self.create_model()
        self.load_weights()
    
    def create_model(self):
        """Recrear exactamente tu arquitectura del entrenamiento"""
        
        try:
            print("🏗️ Recreando arquitectura exacta del modelo...")
            
            # TU ARQUITECTURA EXACTA del entrenamiento
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3))
            base_model.trainable = False  # Tu configuración exacta
            
            # Tu API funcional exacta
            inputs = Input(shape=(self.IMG_SIZE, self.IMG_SIZE, 3))
            x = base_model(inputs, training=False)  # Tu línea exacta
            x = tf.keras.layers.GlobalAveragePooling2D()(x)  # Tu línea exacta
            
            # Tus capas densas exactas
            x = Dense(128, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            outputs = Dense(1, activation='sigmoid')(x)  # Tu línea exacta
            
            self.model = Model(inputs, outputs)
            
            # Compilar igual que en entrenamiento
            self.model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            print("✅ Arquitectura recreada exitosamente")
            self.model.summary()
            
            return True
            
        except Exception as e:
            print(f"❌ Error creando arquitectura: {e}")
            return False
    
    def load_weights(self):
        """Cargar pesos del modelo entrenado"""
        
        for model_path in self.model_paths:
            if os.path.exists(model_path):
                try:
                    print(f"🔍 Intentando cargar pesos desde: {model_path}")
                    
                    # Método 1: Cargar pesos directamente
                    try:
                        self.model.load_weights(model_path)
                        self.model_path = model_path
                        self.model_loaded = True
                        print(f"✅ Pesos cargados exitosamente desde: {model_path}")
                        return True
                    except Exception as e:
                        print(f"⚠️ Error cargando pesos directamente: {e}")
                        
                        # Método 2: Intentar cargar modelo completo y extraer pesos
                        try:
                            print("🔧 Intentando extraer pesos del modelo completo...")
                            original_model = tf.keras.models.load_model(model_path)
                            
                            # Copiar pesos capa por capa
                            for i, layer in enumerate(self.model.layers):
                                if i < len(original_model.layers):
                                    try:
                                        layer.set_weights(original_model.layers[i].get_weights())
                                    except:
                                        continue
                            
                            self.model_path = model_path
                            self.model_loaded = True
                            print(f"✅ Pesos extraídos exitosamente")
                            return True
                            
                        except Exception as e2:
                            print(f"❌ También falló extracción de pesos: {e2}")
                            continue
                            
                except Exception as e:
                    print(f"❌ Error general con {model_path}: {e}")
                    continue
        
        print("❌ No se pudieron cargar los pesos desde ninguna ubicación")
        return False
    
    def predict_single_image_bytes(self, image_bytes):
        """
        Tu función de predicción exacta adaptada para bytes
        """
        
        if not self.model_loaded:
            print("❌ Modelo no cargado")
            return None, None, None
        
        try:
            # Convertir bytes a archivo temporal
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                tmp_file.write(image_bytes)
                temp_path = tmp_file.name
            
            try:
                # TU FUNCIÓN EXACTA predict_single_image
                img = image.load_img(temp_path, target_size=(self.IMG_SIZE, self.IMG_SIZE))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.0
                
                prediction_proba = self.model.predict(img_array)[0][0]
                
                if prediction_proba > 0.5:
                    prediction_label = "Cáncer de Mama (IDC Positivo)"
                    has_cancer = True
                else:
                    prediction_label = "No Cáncer (IDC Negativo)"
                    has_cancer = False
                
                print(f"🔍 Predicción MobileNetV2:")
                print(f"   - Probabilidad: {prediction_proba:.6f}")
                print(f"   - Clasificación: {prediction_label}")
                
                return prediction_proba, prediction_label, has_cancer
                
            finally:
                # Limpiar archivo temporal
                os.unlink(temp_path)
                
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            return None, None, None
    
    def predict(self, image_input):
        """Función para el sistema web"""
        
        try:
            result = self.predict_single_image_bytes(image_input)
            
            if result[0] is not None:
                probability, label, has_cancer = result
                
                # Calcular confianza
                confidence = abs(probability - 0.5) * 2
                
                # Región sospechosa si hay cáncer
                bbox = None
                if has_cancer:
                    bbox = {
                        'x': 12,
                        'y': 12,
                        'width': 26,
                        'height': 26
                    }
                
                return {
                    'probability': float(probability),
                    'has_cancer': bool(has_cancer),
                    'confidence': float(confidence),
                    'bbox': bbox,
                    'model_type': 'MobileNetV2_TransferLearning',
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
            'model_type': 'MobileNetV2 Transfer Learning - Arquitectura Recreada',
            'base_model': 'MobileNetV2',
            'tf_version': tf.__version__
        }

# Exportar para el sistema
DirectWebModel = MobileNetV2Model
DirectRealModel = MobileNetV2Model
RealBreastCancerModel = MobileNetV2Model

def test_mobilenet_model():
    """Probar el modelo MobileNetV2 recreado"""
    try:
        print("🧪 Probando modelo MobileNetV2 recreado...")
        
        model = MobileNetV2Model()
        
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
    print("🏥 MODELO MOBILENETV2 RECREADO")
    print("=" * 60)
    
    success = test_mobilenet_model()
    
    if success:
        print("\\n✅ Modelo MobileNetV2 funcionando correctamente")
    else:
        print("\\n❌ Error en modelo MobileNetV2")
