import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import mixed_precision # Para mantener la consistencia con el entrenamiento

# --- 0. Configuración de Aceleración y Verificación de Hardware ---
# Se mantiene la política de precisión mixta por si se ejecuta en un entorno con GPU.
# En CPU no tendrá un impacto significativo.
mixed_precision.set_global_policy('mixed_float16')
print("Política de precisión mixta establecida a 'mixed_float16'. Nota: Esto acelera principalmente en GPUs.")

# Verificar si TensorFlow está usando la GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs detectadas y configuradas para la predicción: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("No se detectaron GPUs. La predicción se realizará en CPU.")

# --- 1. Configuración de Rutas y Parámetros del Modelo ---

# Ruta donde se guardó tu modelo entrenado
MODEL_PATH = r"Z:\Trabajos\TrabajoTesis\dev_cruz_ml\app-cruz-ml\ml-model\breast_cancer_detection_model_transfer.h5"
IMG_SIZE = 50 # El tamaño de imagen con el que se entrenó el modelo

# --- 2. Cargar el Modelo Entrenado ---
try:
    model = load_model(MODEL_PATH)
    print(f"Modelo cargado exitosamente desde: {MODEL_PATH}")
    model.summary() # Muestra un resumen del modelo cargado
except Exception as e:
    print(f"Error al cargar el modelo desde {MODEL_PATH}. Asegúrate de que la ruta es correcta y el modelo existe.")
    print(f"Error: {e}")
    exit() # Salir si el modelo no se puede cargar

# --- 3. Función de Predicción para una Sola Imagen ---

def predict_single_image(image_path, model, img_size):
    """
    Realiza una predicción en una sola imagen utilizando el modelo cargado.

    Args:
        image_path (str): Ruta completa de la imagen a predecir.
        model (keras.Model): El modelo de Keras entrenado.
        img_size (int): Tamaño al que se deben redimensionar las imágenes (img_size x img_size).

    Returns:
        tuple: (probabilidad_de_cancer, etiqueta_predicha_texto)
               Retorna (None, None) si ocurre un error.
    """
    if not os.path.exists(image_path):
        print(f"Error: La imagen no se encontró en la ruta especificada: {image_path}")
        return None, None

    try:
        # Cargar la imagen y redimensionarla
        img = image.load_img(image_path, target_size=(img_size, img_size))
        # Convertir la imagen a un array de NumPy
        img_array = image.img_to_array(img)
        # Expandir las dimensiones para que coincida con el formato de entrada del modelo (batch_size, height, width, channels)
        img_array = np.expand_dims(img_array, axis=0)
        # Normalizar los valores de los píxeles (igual que durante el entrenamiento)
        img_array /= 255.0

        # Realizar la predicción
        prediction_proba = model.predict(img_array)[0][0] # Obtener la probabilidad de la clase positiva

        # Interpretar la predicción
        if prediction_proba > 0.5:
            prediction_label = "Cáncer de Mama (IDC Positivo)"
        else:
            prediction_label = "No Cáncer (IDC Negativo)"

        return prediction_proba, prediction_label

    except Exception as e:
        print(f"Ocurrió un error al procesar la imagen '{image_path}': {e}")
        return None, None

# --- 4. Interfaz de Usuario para la Predicción ---

if __name__ == "__main__":
    print("\n--- Herramienta de Predicción de Cáncer de Mama ---")
    print("Introduce la ruta completa de la imagen (ej: C:\\Imagenes\\mi_imagen.png)")
    print("O escribe 'salir' para terminar.")

    while True:
        image_path_input = input("\nRuta de la imagen: ").strip()

        if image_path_input.lower() == 'salir':
            print("Saliendo de la herramienta de predicción.")
            break

        if not image_path_input:
            print("Por favor, introduce una ruta de imagen válida.")
            continue

        prob, label = predict_single_image(image_path_input, model, IMG_SIZE)

        if prob is not None:
            print(f"\nResultados de la Predicción para: {os.path.basename(image_path_input)}")
            print(f"Nivel de Confianza (Probabilidad de Cáncer): {prob:.4f}")
            print(f"Predicción: {label}")
        else:
            print("No se pudo realizar la predicción para esta imagen.")

