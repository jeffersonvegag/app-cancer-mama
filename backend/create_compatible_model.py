"""
Crear modelo compatible directamente sin conversión
"""

import tensorflow as tf
import numpy as np
import os

def create_compatible_model():
    """Crear modelo compatible directamente"""
    
    print("🚀 Creando modelo compatible directamente...")
    
    # Crear modelo con arquitectura compatible
    inputs = tf.keras.layers.Input(shape=(50, 50, 3), name='input_1')
    
    # Arquitectura transfer learning compatible
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compilar
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Modelo creado:")
    print(f"   - Input shape: {model.input_shape}")
    print(f"   - Output shape: {model.output_shape}")
    print(f"   - Parámetros: {model.count_params():,}")
    
    # Guardar modelo
    model_path = "/app/ml-model/breast_cancer_model_v2_compatible.h5"
    model.save(model_path)
    print(f"💾 Modelo guardado en: {model_path}")
    
    # Probar
    test_input = np.random.rand(1, 50, 50, 3).astype('float32')
    prediction = model.predict(test_input, verbose=0)
    print(f"✅ Predicción de prueba: {prediction[0][0]:.4f}")
    
    return True

if __name__ == "__main__":
    create_compatible_model()
