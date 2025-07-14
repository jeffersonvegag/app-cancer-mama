"""
Verificar que el modelo v3 esté en la ubicación correcta en Docker
"""

import os

def check_model_locations():
    """Verificar dónde está el modelo dentro del contenedor"""
    
    print("🔍 VERIFICANDO UBICACIONES DEL MODELO V3")
    print("=" * 50)
    
    # Ubicaciones a verificar
    locations = [
        "/app/ml-model/breast_cancer_model_compatible_v3.h5",
        "./ml-model/breast_cancer_model_compatible_v3.h5", 
        "../ml-model/breast_cancer_model_compatible_v3.h5",
        "/app/ml-model/",  # Listar directorio
    ]
    
    for location in locations:
        print(f"📂 Verificando: {location}")
        
        if location.endswith("/"):
            # Es un directorio, listar contenido
            if os.path.exists(location):
                try:
                    files = os.listdir(location)
                    print(f"   ✅ Directorio existe, archivos:")
                    for file in files:
                        if file.endswith('.h5'):
                            print(f"      🔹 {file}")
                except Exception as e:
                    print(f"   ❌ Error listando: {e}")
            else:
                print(f"   ❌ Directorio no existe")
        else:
            # Es un archivo específico
            if os.path.exists(location):
                size = os.path.getsize(location)
                print(f"   ✅ ENCONTRADO! Tamaño: {size:,} bytes")
            else:
                print(f"   ❌ No encontrado")
    
    print("\n📋 Información del entorno:")
    print(f"   - Directorio actual: {os.getcwd()}")
    print(f"   - Contenido directorio actual:")
    try:
        for item in os.listdir("."):
            print(f"     - {item}")
    except:
        pass

if __name__ == "__main__":
    check_model_locations()
