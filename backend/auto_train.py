"""
Auto-entrenamiento del modelo al iniciar el contenedor
Este script se ejecuta automáticamente para asegurar que tenemos un modelo funcional
"""

import os
import sys
import numpy as np

def auto_train_model():
    """Entrenar modelo automáticamente si no existe"""
    
    model_path = '/app/ml-model/trained_breast_cancer_model.h5'
    
    # Si ya existe el modelo, no hacer nada
    if os.path.exists(model_path):
        print(f"✅ Modelo pre-entrenado encontrado: {model_path}")
        return True
    
    print("🤖 Iniciando auto-entrenamiento del modelo...")
    
    try:
        # Importar el modelo
        sys.path.append('/app/ml-model')
        from model import BreastCancerModel, create_synthetic_data
        
        print("📦 Creando modelo EfficientNet...")
        model = BreastCancerModel()
        
        print("🎲 Generando datos sintéticos de entrenamiento...")
        X_train, y_train = create_synthetic_data(500)
        X_val, y_val = create_synthetic_data(100)
        
        print("🏋️ Entrenando modelo (esto puede tomar unos minutos)...")
        
        # Entrenamiento básico y silencioso
        history = model.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=3,  # Pocas épocas para rapidez
            batch_size=32,
            verbose=0  # Silencioso
        )
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Guardar modelo
        model.save_model(model_path)
        
        print(f"✅ Modelo entrenado y guardado: {model_path}")
        return True
        
    except Exception as e:
        print(f"⚠️ Error en auto-entrenamiento: {e}")
        print("🔄 Continuando sin modelo pre-entrenado...")
        return False

if __name__ == "__main__":
    auto_train_model()
