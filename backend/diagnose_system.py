"""
Script de diagnóstico para el sistema de cáncer de mama
Identifica problemas y ayuda a solucionarlos
"""

import os
import sys
import json
import sqlite3
from datetime import datetime

def check_file_exists(path, description):
    """Verificar si un archivo existe"""
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {path}")
    return exists

def check_directory_structure():
    """Verificar estructura de directorios"""
    print("\n🔍 VERIFICANDO ESTRUCTURA DE DIRECTORIOS")
    print("=" * 50)
    
    base_dir = "../"
    dirs_to_check = [
        ("backend/", "Directorio Backend"),
        ("frontend/", "Directorio Frontend"), 
        ("ml-model/", "Directorio ML Model"),
        ("backend/data/", "Directorio de Datos"),
        ("backend/ml-model/", "Directorio de Modelos Entrenados")
    ]
    
    all_exist = True
    for dir_path, description in dirs_to_check:
        full_path = os.path.join(base_dir, dir_path)
        exists = os.path.exists(full_path)
        status = "✅" if exists else "❌"
        print(f"{status} {description}: {full_path}")
        
        if not exists:
            all_exist = False
            try:
                os.makedirs(full_path, exist_ok=True)
                print(f"   → Directorio creado automáticamente")
            except Exception as e:
                print(f"   → Error creando directorio: {e}")
    
    return all_exist

def check_files():
    """Verificar archivos importantes"""
    print("\n📁 VERIFICANDO ARCHIVOS IMPORTANTES")
    print("=" * 50)
    
    base_dir = "../"
    files_to_check = [
        ("backend/main.py", "Backend Principal"),
        ("backend/main_fixed.py", "Backend Corregido"),
        ("backend/requirements.txt", "Dependencias Backend"),
        ("ml-model/model.py", "Definición del Modelo"),
        ("ml-model/train.py", "Script de Entrenamiento"),
        ("frontend/src/App.tsx", "Frontend Principal"),
        ("backend/ml-model/trained_breast_cancer_model.h5", "Modelo Entrenado"),
        ("backend/ml-model/model_metrics.json", "Métricas del Modelo")
    ]
    
    results = {}
    for file_path, description in files_to_check:
        full_path = os.path.join(base_dir, file_path)
        exists = check_file_exists(full_path, description)
        results[file_path] = exists
    
    return results

def analyze_current_backend():
    """Analizar el backend actual para detectar problemas"""
    print("\n🔍 ANALIZANDO BACKEND ACTUAL")
    print("=" * 50)
    
    backend_path = "../backend/main.py"
    
    if not os.path.exists(backend_path):
        print("❌ Backend principal no encontrado")
        return False
    
    try:
        with open(backend_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Buscar indicadores de problemas
        issues = []
        
        if "random.uniform" in content:
            issues.append("PROBLEMA: Código contiene ruido aleatorio (random.uniform)")
            
        if "SimplifiedMLModel" in content:
            issues.append("PROBLEMA: Usando modelo simulado en lugar del real")
            
        if "import model" not in content and "from model import" not in content:
            issues.append("PROBLEMA: No importa el modelo EfficientNet real")
            
        if "noise = " in content:
            issues.append("PROBLEMA: Agregando ruido artificial a las predicciones")
        
        if issues:
            print("❌ PROBLEMAS DETECTADOS:")
            for issue in issues:
                print(f"   - {issue}")
            
            print("\n💡 SOLUCIÓN:")
            print("   1. Usar el archivo main_fixed.py que corrige estos problemas")
            print("   2. O reemplazar main.py con la versión corregida")
            return False
        else:
            print("✅ Backend parece estar correcto")
            return True
            
    except Exception as e:
        print(f"❌ Error analizando backend: {e}")
        return False

def check_model_availability():
    """Verificar disponibilidad del modelo"""
    print("\n🤖 VERIFICANDO MODELO DE ML")
    print("=" * 50)
    
    try:
        sys.path.append("../ml-model")
        from model import BreastCancerModel
        
        print("✅ Modelo BreastCancerModel importado correctamente")
        
        # Intentar crear instancia
        model = BreastCancerModel()
        print("✅ Modelo instanciado correctamente")
        
        # Verificar si existe modelo entrenado
        model_path = "../backend/ml-model/trained_breast_cancer_model.h5"
        if os.path.exists(model_path):
            print("✅ Modelo entrenado encontrado")
            
            # Verificar métricas
            metrics_path = "../backend/ml-model/model_metrics.json"
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                print("✅ Métricas del modelo disponibles:")
                print(f"   - Accuracy: {metrics.get('test_metrics', {}).get('accuracy', 'N/A')}")
                print(f"   - Fecha entrenamiento: {metrics.get('training_date', 'N/A')}")
            else:
                print("⚠️ Métricas del modelo no encontradas")
        else:
            print("❌ Modelo entrenado no encontrado")
            print("💡 Ejecuta: python train_model.py --quick")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error importando modelo: {e}")
        print("💡 Verifica las dependencias: pip install tensorflow")
        return False
    except Exception as e:
        print(f"❌ Error con el modelo: {e}")
        return False

def check_database():
    """Verificar base de datos"""
    print("\n💾 VERIFICANDO BASE DE DATOS")
    print("=" * 50)
    
    db_path = "../backend/data/patients.db"
    
    try:
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Verificar tablas
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            print(f"✅ Base de datos encontrada: {db_path}")
            print(f"   Tablas: {[table[0] for table in tables]}")
            
            # Contar pacientes
            cursor.execute("SELECT COUNT(*) FROM patients;")
            patient_count = cursor.fetchone()[0]
            print(f"   Pacientes registrados: {patient_count}")
            
            # Contar diagnósticos
            cursor.execute("SELECT COUNT(*) FROM diagnoses;")
            diagnosis_count = cursor.fetchone()[0]
            print(f"   Diagnósticos realizados: {diagnosis_count}")
            
            conn.close()
            return True
            
        else:
            print("❌ Base de datos no encontrada")
            print("💡 Se creará automáticamente al iniciar el backend")
            return False
            
    except Exception as e:
        print(f"❌ Error verificando base de datos: {e}")
        return False

def generate_recommendations():
    """Generar recomendaciones basadas en el diagnóstico"""
    print("\n💡 RECOMENDACIONES")
    print("=" * 50)
    
    # Verificar estado general
    file_results = check_files()
    backend_ok = analyze_current_backend()
    model_ok = check_model_availability()
    
    print("\n🎯 PASOS RECOMENDADOS:")
    
    if not file_results.get("backend/main_fixed.py", False):
        print("1. ❌ Archivo main_fixed.py no encontrado - necesario para solución")
    else:
        print("1. ✅ Archivo main_fixed.py disponible")
    
    if not backend_ok:
        print("2. 🔧 REEMPLAZAR backend actual:")
        print("   cd backend")
        print("   cp main_fixed.py main.py")
    else:
        print("2. ✅ Backend parece correcto")
    
    if not model_ok:
        print("3. 🤖 ENTRENAR modelo:")
        print("   cd backend")
        print("   python train_model.py --quick")
    else:
        print("3. ✅ Modelo disponible")
    
    if not file_results.get("backend/ml-model/trained_breast_cancer_model.h5", False):
        print("4. 📊 EJECUTAR entrenamiento para generar modelo .h5")
    else:
        print("4. ✅ Modelo .h5 disponible")
    
    print("\n5. 🚀 PROBAR sistema:")
    print("   - Iniciar backend: python main.py")
    print("   - Subir la misma imagen varias veces")
    print("   - Verificar que resultados sean consistentes")

def main():
    """Función principal de diagnóstico"""
    print("🏥 DIAGNÓSTICO DEL SISTEMA DE CÁNCER DE MAMA")
    print("=" * 60)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    check_directory_structure()
    check_files()
    analyze_current_backend()
    check_model_availability()
    check_database()
    generate_recommendations()
    
    print("\n" + "=" * 60)
    print("DIAGNÓSTICO COMPLETADO")
    print("=" * 60)

if __name__ == "__main__":
    main()
