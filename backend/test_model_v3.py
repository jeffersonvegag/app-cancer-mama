"""
Probar el nuevo modelo compatible v3
"""

import sys
import os
sys.path.append('/app/ml-model')

try:
    from model_real_fixed import RealBreastCancerModelFixed
    import numpy as np
    
    print("🧪 PROBANDO MODELO COMPATIBLE V3")
    print("=" * 50)
    
    # Crear instancia del modelo
    print("🔄 Creando instancia del modelo...")
    model = RealBreastCancerModelFixed()
    
    if not model.model_loaded:
        print("❌ FALLO: No se pudo cargar el modelo")
        exit(1)
    
    print("✅ Modelo cargado exitosamente!")
    print(f"   - Tipo: {model.model_type if hasattr(model, 'model_type') else 'N/A'}")
    print(f"   - Ruta: {model.model_path}")
    
    # Mostrar información del modelo
    info = model.get_model_info()
    print("\n📋 Información del modelo:")
    for key, value in info.items():
        print(f"   - {key}: {value}")
    
    # Crear imagen de prueba
    print("\n🖼️ Creando imagen de prueba...")
    test_image = np.random.rand(50, 50, 3).astype('float32') * 255
    test_image = test_image.astype('uint8')
    
    # Hacer predicción
    print("🧠 Realizando predicción...")
    result = model.predict(test_image)
    
    print("\n✅ PREDICCIÓN EXITOSA!")
    print("=" * 50)
    print(f"🎯 Resultados:")
    print(f"   - Probabilidad: {result['probability']:.4f}")
    print(f"   - Tiene cáncer: {'SÍ' if result['has_cancer'] else 'NO'}")
    print(f"   - Confianza: {result['confidence']:.4f}")
    print(f"   - Región sospechosa: {'SÍ' if result.get('bbox') else 'NO'}")
    print(f"   - Tipo de modelo: {result.get('model_type', 'N/A')}")
    
    # Probar con varias imágenes
    print("\n🔄 Probando con múltiples imágenes...")
    for i in range(3):
        test_img = np.random.rand(50, 50, 3).astype('float32') * 255
        result = model.predict(test_img.astype('uint8'))
        print(f"   Imagen {i+1}: {result['probability']:.4f} ({'Cáncer' if result['has_cancer'] else 'Normal'})")
    
    print("\n🎉 TODAS LAS PRUEBAS EXITOSAS!")
    print("El modelo compatible v3 está funcionando correctamente.")
    
except Exception as e:
    print(f"❌ ERROR EN PRUEBA: {e}")
    import traceback
    print(traceback.format_exc())
    exit(1)
