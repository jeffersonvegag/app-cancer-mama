"""
Script para entrenar y guardar el modelo para el sistema de diagnóstico
Este script entrenará el modelo EfficientNet y lo guardará para uso en producción
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml-model'))

import numpy as np
import tensorflow as tf
from model import BreastCancerModel, create_synthetic_data
import json
from datetime import datetime

def train_and_save_model():
    """Entrenar modelo y guardarlo para uso en producción"""
    
    print("=" * 60)
    print("ENTRENAMIENTO DEL MODELO DE DIAGNÓSTICO DE CÁNCER DE MAMA")
    print("=" * 60)
    
    # 1. Crear el modelo
    print("\n1. Creando modelo EfficientNet...")
    model = BreastCancerModel(input_shape=(224, 224, 3))
    
    # 2. Crear datos sintéticos para entrenamiento
    print("\n2. Generando datos sintéticos para entrenamiento...")
    print("   (En un caso real, aquí cargarías tu dataset de mamografías)")
    
    X_train, y_train = create_synthetic_data(1600)  # Más datos para mejor entrenamiento
    X_val, y_val = create_synthetic_data(400)
    X_test, y_test = create_synthetic_data(200)
    
    print(f"   - Datos de entrenamiento: {X_train.shape}")
    print(f"   - Datos de validación: {X_val.shape}")
    print(f"   - Datos de prueba: {X_test.shape}")
    
    # 3. Entrenar el modelo
    print("\n3. Entrenando modelo...")
    print("   Esto puede tomar varios minutos...")
    
    # Configurar callbacks mejorados
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model_checkpoint.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]
    
    # Entrenar
    history = model.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,  # Más epochs para mejor entrenamiento
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # 4. Evaluar modelo
    print("\n4. Evaluando modelo...")
    test_loss, test_acc, test_prec, test_rec = model.model.evaluate(X_test, y_test, verbose=0)
    
    print(f"   - Test Loss: {test_loss:.4f}")
    print(f"   - Test Accuracy: {test_acc:.4f}")
    print(f"   - Test Precision: {test_prec:.4f}")
    print(f"   - Test Recall: {test_rec:.4f}")
    
    # 5. Calcular F1-Score
    f1_score = 2 * (test_prec * test_rec) / (test_prec + test_rec) if (test_prec + test_rec) > 0 else 0
    print(f"   - Test F1-Score: {f1_score:.4f}")
    
    # 6. Guardar modelo
    print("\n5. Guardando modelo...")
    
    # Crear directorio si no existe
    model_dir = "../backend/ml-model"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "trained_breast_cancer_model.h5")
    model.save_model(model_path)
    
    # 7. Guardar métricas de entrenamiento
    metrics_path = os.path.join(model_dir, "model_metrics.json")
    metrics = {
        "training_date": datetime.now().isoformat(),
        "model_architecture": "EfficientNetB0",
        "input_shape": list(model.input_shape),
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "test_metrics": {
            "loss": float(test_loss),
            "accuracy": float(test_acc),
            "precision": float(test_prec),
            "recall": float(test_rec),
            "f1_score": float(f1_score)
        },
        "training_history": {
            "final_train_loss": float(history.history['loss'][-1]),
            "final_val_loss": float(history.history['val_loss'][-1]),
            "final_train_acc": float(history.history['accuracy'][-1]),
            "final_val_acc": float(history.history['val_accuracy'][-1]),
            "epochs_trained": len(history.history['loss'])
        }
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"   - Modelo guardado en: {model_path}")
    print(f"   - Métricas guardadas en: {metrics_path}")
    
    # 8. Probar el modelo guardado
    print("\n6. Probando modelo guardado...")
    
    # Crear nueva instancia y cargar modelo
    test_model = BreastCancerModel()
    test_model.load_model(model_path)
    
    # Crear imagen de prueba
    test_image = np.random.rand(224, 224, 3).astype('float32')
    result = test_model.predict(test_image)
    
    print("   Resultado de prueba:")
    print(f"   - Probabilidad: {result['probability']:.4f}")
    print(f"   - Tiene cáncer: {result['has_cancer']}")
    print(f"   - Confianza: {result['confidence']:.4f}")
    
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 60)
    
    return model, metrics

def quick_test():
    """Prueba rápida del modelo para desarrollo"""
    
    print("=" * 50)
    print("PRUEBA RÁPIDA DEL MODELO")
    print("=" * 50)
    
    # Crear modelo
    print("\nCreando modelo...")
    model = BreastCancerModel()
    
    # Entrenar con pocos datos para prueba rápida
    print("\nGenerando datos de prueba...")
    X_train, y_train = create_synthetic_data(100)
    X_val, y_val = create_synthetic_data(50)
    
    print("\nEntrenamiento rápido (3 epochs)...")
    history = model.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=3,
        batch_size=16,
        verbose=1
    )
    
    # Guardar modelo de prueba
    model_path = "../backend/ml-model/quick_test_model.h5"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    
    print(f"\nModelo de prueba guardado en: {model_path}")
    print("Listo para usar en el sistema!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenar modelo de diagnóstico de cáncer de mama')
    parser.add_argument('--quick', action='store_true', help='Entrenamiento rápido para pruebas')
    parser.add_argument('--full', action='store_true', help='Entrenamiento completo')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    elif args.full:
        train_and_save_model()
    else:
        print("Selecciona una opción:")
        print("  --quick: Entrenamiento rápido para pruebas")
        print("  --full: Entrenamiento completo")
        
        choice = input("\n¿Quieres hacer un entrenamiento rápido? (y/n): ").lower()
        if choice in ['y', 'yes', 's', 'si']:
            quick_test()
        else:
            train_and_save_model()
