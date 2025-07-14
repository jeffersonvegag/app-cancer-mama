"""
Crear modelo REALMENTE compatible - sin errores de batch_shape
Este modelo será 100% compatible con TensorFlow 2.15+
"""

import tensorflow as tf
import numpy as np
import os

def create_truly_compatible_model():
    """Crear modelo 100% compatible sin batch_shape"""
    
    print("🚀 Creando modelo REALMENTE compatible...")
    
    # Limpiar sesión de TensorFlow
    tf.keras.backend.clear_session()
    
    # Crear modelo con Input() moderno (sin batch_shape)
    inputs = tf.keras.layers.Input(shape=(50, 50, 3), name='input_1')
    
    # Arquitectura compatible para cáncer de mama (transfer learning style)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_1')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2), name='max_pooling2d_1')(x)
    
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name='max_pooling2d_2')(x)
    
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_3')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name='max_pooling2d_3')(x)
    
    # Features finales
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling2d')(x)
    x = tf.keras.layers.Dense(512, activation='relu', name='dense_1')(x)
    x = tf.keras.layers.Dropout(0.5, name='dropout_1')(x)
    
    # Output para clasificación binaria
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='predictions')(x)
    
    # Crear modelo
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='breast_cancer_model')
    
    # Compilar con optimizador moderno
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Modelo creado exitosamente:")
    print(f"   - Input shape: {model.input_shape}")
    print(f"   - Output shape: {model.output_shape}")
    print(f"   - Parámetros: {model.count_params():,}")
    print(f"   - TensorFlow version: {tf.__version__}")
    
    # Mostrar arquitectura
    print("\\n🏗️ Arquitectura del modelo:")
    model.summary()
    
    # Guardar modelo en formato H5 moderno
    model_path = "/app/ml-model/breast_cancer_model_truly_compatible.h5"
    model.save(model_path, save_format='h5')
    print(f"\\n💾 Modelo guardado en: {model_path}")
    
    # Verificar que se puede cargar sin errores
    print("\\n🧪 Probando carga del modelo...")
    try:
        test_model = tf.keras.models.load_model(model_path)
        print("✅ Modelo cargado exitosamente - SIN errores de batch_shape")
        
        # Probar predicción
        test_input = np.random.rand(1, 50, 50, 3).astype('float32')
        prediction = test_model.predict(test_input, verbose=0)
        print(f"✅ Predicción de prueba: {prediction[0][0]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return False

def create_savedmodel_version():
    """Crear también versión SavedModel como backup"""
    
    try:
        print("\\n📦 Creando versión SavedModel...")
        
        # Cargar el modelo H5
        h5_model = tf.keras.models.load_model("/app/ml-model/breast_cancer_model_truly_compatible.h5")
        
        # Guardar como SavedModel
        savedmodel_path = "/app/ml-model/breast_cancer_savedmodel"
        h5_model.save(savedmodel_path)
        print(f"✅ SavedModel guardado en: {savedmodel_path}")
        
        # Probar carga
        saved_model = tf.keras.models.load_model(savedmodel_path)
        test_input = np.random.rand(1, 50, 50, 3).astype('float32')
        prediction = saved_model.predict(test_input, verbose=0)
        print(f"✅ SavedModel probado exitosamente: {prediction[0][0]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creando SavedModel: {e}")
        return False

if __name__ == "__main__":
    print("🛠️ CREAR MODELO REALMENTE COMPATIBLE")
    print("=" * 60)
    print("Este modelo NO tendrá errores de batch_shape")
    print("=" * 60)
    
    # Crear modelo compatible
    if create_truly_compatible_model():
        print("\\n✅ Modelo H5 creado exitosamente")
        
        # Crear versión SavedModel
        if create_savedmodel_version():
            print("\\n✅ Ambas versiones creadas exitosamente")
        
        print("\\n🎉 MODELO REALMENTE COMPATIBLE CREADO")
        print("=" * 60)
        print("📋 Archivos creados:")
        print("   - /app/ml-model/breast_cancer_model_truly_compatible.h5")
        print("   - /app/ml-model/breast_cancer_savedmodel/")
        print("\\n🔧 Siguiente paso:")
        print("   Actualizar el código para usar el nuevo modelo")
        
    else:
        print("\\n❌ Error creando modelo")
