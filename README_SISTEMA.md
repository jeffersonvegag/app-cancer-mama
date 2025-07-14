# Sistema de Diagnóstico de Cáncer de Mama

## 📋 Descripción

Este sistema utiliza Machine Learning (EfficientNet) para analizar mamografías y ayudar en el diagnóstico temprano de cáncer de mama. El sistema incluye:

- **Backend**: API REST con FastAPI y modelo EfficientNet
- **Frontend**: Interfaz web con React/TypeScript  
- **Base de datos**: SQLite para gestión de pacientes y diagnósticos
- **ML Model**: EfficientNetB0 pre-entrenado con transfer learning

## 🔍 ¿Qué Modelo Estás Usando?

### Arquitectura del Modelo
- **Base**: EfficientNetB0 (pre-entrenado en ImageNet)
- **Transfer Learning**: Capas congeladas + capas personalizadas
- **Entrada**: Imágenes de 224x224x3 píxeles
- **Salida**: Probabilidad de cáncer (0-1) + clasificación binaria

### Componentes del Modelo
```
Input (224x224x3)
    ↓
EfficientNetB0 (congelado)
    ↓
GlobalAveragePooling2D
    ↓
Dropout(0.3)
    ↓
Dense(128, activation='relu')
    ↓
Dropout(0.2)
    ↓
Dense(1, activation='sigmoid') → Predicción final
```

## ⚠️ Problema Actual: Resultados Variables

### Causa del Problema
El problema de variación en resultados se debe a que **actualmente el sistema NO está usando el modelo EfficientNet real**. En su lugar, usa una simulación con ruido aleatorio:

```python
# En main.py (línea 185-190) - PROBLEMA:
noise = random.uniform(-0.2, 0.2)  # ±20% de ruido aleatorio
prediction_score = max(0.05, min(0.95, base_score + noise))
```

Cada predicción agrega ruido aleatorio del -20% al +20%, causando resultados inconsistentes para la misma imagen.

### Solución Implementada
He creado `main_fixed.py` que:
1. **Usa el modelo EfficientNet real** cuando está disponible
2. **Simulación determinística** (sin aleatoriedad) como fallback
3. **Resultados consistentes** para la misma imagen

## 🚀 Instalación y Configuración

### 1. Prerrequisitos
```bash
# Python 3.8+, Node.js 16+, Docker (opcional)
pip install tensorflow fastapi uvicorn opencv-python pillow numpy sqlite3
```

### 2. Entrenar el Modelo
```bash
# Ir al directorio backend
cd app-cruz-ml/backend

# Entrenamiento rápido (para pruebas)
python train_model.py --quick

# Entrenamiento completo (recomendado)
python train_model.py --full
```

### 3. Usar el Modelo Corregido
```bash
# Reemplazar el archivo main.py actual
cp main_fixed.py main.py

# O usar directamente
python main_fixed.py
```

### 4. Ejecutar el Sistema
```bash
# Backend
cd app-cruz-ml/backend
python main.py

# Frontend (nueva terminal)
cd app-cruz-ml/frontend
npm install
npm run dev
```

## 📊 Métricas del Modelo

Después del entrenamiento, el sistema guarda métricas en `model_metrics.json`:
```json
{
  "test_metrics": {
    "accuracy": 0.85,
    "precision": 0.82,
    "recall": 0.78,
    "f1_score": 0.80
  }
}
```

## 🔧 Arquitectura del Sistema

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │    │                 │
│    Frontend     │────│    Backend      │────│   ML Model      │
│   (React TS)    │    │   (FastAPI)     │    │ (EfficientNet)  │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                │
                         ┌─────────────────┐
                         │                 │
                         │   Database      │
                         │   (SQLite)      │
                         │                 │
                         └─────────────────┘
```

## 🎯 Flujo de Predicción

1. **Usuario sube imagen** → Frontend valida formato/tamaño
2. **Envío al backend** → Recibe imagen + ID paciente
3. **Preprocesamiento** → Redimensiona a 224x224, normaliza
4. **Predicción** → EfficientNet procesa imagen
5. **Post-procesamiento** → Calcula confianza, detecta regiones
6. **Respuesta** → JSON con score, diagnóstico, recomendaciones

## 📝 API Endpoints

### Diagnóstico
```http
POST /api/diagnose
Content-Type: multipart/form-data

{
  "file": <imagen>,
  "patient_id": <id>
}
```

### Respuesta
```json
{
  "prediction_score": 0.65,
  "has_cancer": true,
  "confidence": 0.8,
  "diagnosis": "Probabilidad moderada de cáncer",
  "recommendation": "Se recomienda consulta con especialista",
  "bbox": {
    "x": 120, "y": 80, 
    "width": 60, "height": 45
  }
}
```

## 🔬 Detección de Regiones

El sistema incluye detección básica de regiones sospechosas usando:
- **Computer Vision**: Filtros adaptativos, detección de contornos
- **Bounding Boxes**: Coordenadas de regiones anómalas
- **Visualización**: Overlay rojo en la interfaz

## 📈 Mejoras Pendientes

### Para Producción Real:
1. **Dataset Real**: Reemplazar datos sintéticos con mamografías reales
2. **Validación Médica**: Revisar con radiólogos especializados
3. **Métricas Clínicas**: Sensibilidad, especificidad por categorías
4. **Explicabilidad**: Implementar Grad-CAM para visualización
5. **Seguridad**: Cifrado de imágenes, auditoría de accesos

### Mejoras Técnicas:
1. **Data Augmentation**: Rotaciones, zoom, contrast para más datos
2. **Ensemble Models**: Combinar múltiples modelos
3. **Active Learning**: Mejorar con feedback médico
4. **Edge Cases**: Manejo de imágenes de baja calidad

## 🚨 Limitaciones Actuales

1. **Datos Sintéticos**: El modelo actual usa datos generados, no reales
2. **Sin Validación Médica**: No ha sido evaluado por especialistas
3. **Uso Experimental**: NO debe usarse para diagnósticos reales
4. **Detección Básica**: La localización de regiones es simplificada

## 📚 Documentación Técnica

### Estructura de Archivos
```
app-cruz-ml/
├── backend/
│   ├── main.py              # API original (con problema)
│   ├── main_fixed.py        # API corregida
│   ├── train_model.py       # Script de entrenamiento
│   └── requirements.txt
├── ml-model/
│   ├── model.py            # Definición del modelo
│   ├── train.py            # Scripts de entrenamiento
│   └── requirements.txt
└── frontend/
    ├── src/App.tsx         # Interfaz principal
    └── package.json
```

## 🔄 Próximos Pasos

1. **Ejecutar el entrenamiento**: `python train_model.py --quick`
2. **Usar el backend corregido**: `cp main_fixed.py main.py`
3. **Probar consistencia**: Subir la misma imagen varias veces
4. **Validar resultados**: Los scores deben ser consistentes
5. **Preparar datos reales**: Para entrenamiento con mamografías reales

## 📞 Contacto y Soporte

Para preguntas técnicas sobre el modelo o implementación, revisar:
- Logs del backend para errores de carga del modelo
- Métricas en `model_metrics.json` después del entrenamiento
- Estado del modelo en endpoint `/api/health`

---

**⚠️ IMPORTANTE**: Este sistema es experimental y NO debe usarse para diagnósticos médicos reales sin validación clínica apropiada.
