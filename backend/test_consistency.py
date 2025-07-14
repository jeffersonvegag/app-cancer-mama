"""
Script de prueba para verificar consistencia del modelo
Este script prueba que la misma imagen produzca resultados consistentes
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml-model'))

import numpy as np
from PIL import Image
import io
import time

def create_test_image():
    """Crear una imagen de prueba específica"""
    # Crear imagen con patrones específicos para pruebas consistentes
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Agregar algunos patrones
    img[50:150, 50:150] = [200, 200, 200]  # Cuadrado gris claro
    img[200:300, 200:300] = [100, 100, 100]  # Cuadrado gris oscuro
    img[150:250, 150:250] = [150, 150, 150]  # Cuadrado gris medio
    
    # Agregar líneas para simular estructuras mamográficas
    img[100:105, :] = [180, 180, 180]  # Línea horizontal
    img[:, 100:105] = [180, 180, 180]  # Línea vertical
    
    return Image.fromarray(img)

def image_to_bytes(image):
    """Convertir imagen PIL a bytes"""
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='PNG')
    return byte_arr.getvalue()

def test_consistency_with_real_model():
    """Probar consistencia con el modelo real"""
    print("🤖 PROBANDO MODELO REAL (EfficientNet)")
    print("-" * 50)
    
    try:
        from model import BreastCancerModel
        
        # Crear modelo
        model = BreastCancerModel()
        
        # Crear imagen de prueba
        test_image = create_test_image()
        
        # Realizar múltiples predicciones de la misma imagen
        results = []
        print("Realizando 5 predicciones de la misma imagen...")
        
        for i in range(5):
            result = model.predict(test_image)
            results.append(result)
            print(f"Predicción {i+1}: {result['probability']:.4f} | Cáncer: {result['has_cancer']} | Confianza: {result['confidence']:.4f}")
            time.sleep(0.1)  # Pequeña pausa
        
        # Analizar consistencia
        probabilities = [r['probability'] for r in results]
        max_prob = max(probabilities)
        min_prob = min(probabilities)
        variance = max_prob - min_prob
        
        print(f"\nAnálisis de consistencia:")
        print(f"- Probabilidad mínima: {min_prob:.4f}")
        print(f"- Probabilidad máxima: {max_prob:.4f}")
        print(f"- Varianza: {variance:.4f}")
        
        if variance < 0.001:  # Menos de 0.1% de variación
            print("✅ EXCELENTE: Resultados muy consistentes")
        elif variance < 0.01:  # Menos de 1% de variación
            print("✅ BUENO: Resultados consistentes")
        elif variance < 0.05:  # Menos de 5% de variación
            print("⚠️ ACEPTABLE: Ligera variación en resultados")
        else:
            print("❌ PROBLEMA: Demasiada variación en resultados")
        
        return variance < 0.05
        
    except ImportError:
        print("❌ No se pudo importar el modelo real")
        return False
    except Exception as e:
        print(f"❌ Error probando modelo real: {e}")
        return False

def test_consistency_with_fixed_backend():
    """Probar consistencia con el backend corregido"""
    print("\n🔧 PROBANDO BACKEND CORREGIDO")
    print("-" * 50)
    
    try:
        # Importar la clase del backend corregido
        sys.path.append(".")
        
        # Leer el archivo main_fixed.py y ejecutar la clase
        if os.path.exists("main_fixed.py"):
            with open("main_fixed.py", "r") as f:
                exec(f.read(), globals())
            
            # Crear instancia del modelo corregido
            ml_model = RealMLModel()
            
            # Crear imagen de prueba
            test_image = create_test_image()
            image_bytes = image_to_bytes(test_image)
            
            # Realizar múltiples predicciones
            results = []
            print("Realizando 5 predicciones con backend corregido...")
            
            for i in range(5):
                result = ml_model.predict(image_bytes)
                results.append(result)
                print(f"Predicción {i+1}: {result['prediction_score']:.4f} | Cáncer: {result['has_cancer']} | Tipo: {result.get('model_type', 'Unknown')}")
                time.sleep(0.1)
            
            # Analizar consistencia
            scores = [r['prediction_score'] for r in results]
            max_score = max(scores)
            min_score = min(scores)
            variance = max_score - min_score
            
            print(f"\nAnálisis de consistencia del backend:")
            print(f"- Score mínimo: {min_score:.4f}")
            print(f"- Score máximo: {max_score:.4f}")
            print(f"- Varianza: {variance:.4f}")
            
            if variance == 0:
                print("✅ PERFECTO: Resultados idénticos (determinísticos)")
            elif variance < 0.001:
                print("✅ EXCELENTE: Resultados muy consistentes")
            else:
                print("❌ PROBLEMA: Resultados inconsistentes")
            
            return variance < 0.001
            
        else:
            print("❌ main_fixed.py no encontrado")
            return False
            
    except Exception as e:
        print(f"❌ Error probando backend corregido: {e}")
        return False

def test_consistency_with_original_backend():
    """Probar con el backend original para mostrar el problema"""
    print("\n⚠️ PROBANDO BACKEND ORIGINAL (para comparación)")
    print("-" * 50)
    
    try:
        if os.path.exists("main.py"):
            # Leer archivo main.py original
            with open("main.py", "r") as f:
                content = f.read()
            
            # Verificar si tiene el problema del ruido aleatorio
            if "random.uniform" in content and "SimplifiedMLModel" in content:
                print("❌ Backend original detectado con problema de ruido aleatorio")
                print("   (No ejecutando prueba para evitar resultados inconsistentes)")
                print("   → El problema está en la línea que contiene 'random.uniform'")
                return False
            else:
                print("✅ Backend parece estar corregido")
                return True
        else:
            print("❌ main.py no encontrado")
            return False
            
    except Exception as e:
        print(f"Error verificando backend original: {e}")
        return False

def generate_test_report():
    """Generar reporte de pruebas"""
    print("\n📊 REPORTE DE PRUEBAS DE CONSISTENCIA")
    print("=" * 60)
    
    # Realizar todas las pruebas
    real_model_ok = test_consistency_with_real_model()
    fixed_backend_ok = test_consistency_with_fixed_backend()
    original_backend_ok = test_consistency_with_original_backend()
    
    print("\n📋 RESUMEN:")
    print("-" * 30)
    
    status_real = "✅ PASS" if real_model_ok else "❌ FAIL"
    status_fixed = "✅ PASS" if fixed_backend_ok else "❌ FAIL"
    status_original = "✅ PASS" if original_backend_ok else "❌ FAIL"
    
    print(f"Modelo Real (EfficientNet): {status_real}")
    print(f"Backend Corregido:         {status_fixed}")
    print(f"Backend Original:          {status_original}")
    
    print("\n💡 RECOMENDACIONES:")
    if not real_model_ok:
        print("- Entrenar el modelo real: python train_model.py --quick")
    
    if not fixed_backend_ok:
        print("- Usar el backend corregido: cp main_fixed.py main.py")
    
    if not original_backend_ok:
        print("- Reemplazar el backend original que tiene problemas")
    
    if real_model_ok and fixed_backend_ok:
        print("✅ Sistema funcionando correctamente!")
        print("   Los resultados deberían ser consistentes ahora.")

def main():
    """Función principal"""
    print("🧪 PRUEBAS DE CONSISTENCIA DEL MODELO")
    print("=" * 60)
    print("Este script verifica que el modelo produzca resultados consistentes")
    print("para la misma imagen en múltiples ejecuciones.")
    print()
    
    generate_test_report()
    
    print("\n" + "=" * 60)
    print("PRUEBAS COMPLETADAS")
    print("=" * 60)

if __name__ == "__main__":
    main()
