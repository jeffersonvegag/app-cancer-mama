from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import json
from typing import Optional
import sqlite3
import os
from datetime import datetime
import sys

# Agregar el directorio ml-model al path para importar el modelo
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml-model'))

try:
    from model import BreastCancerModel
    ML_MODEL_AVAILABLE = True
    print("Modelo de ML importado exitosamente")
except ImportError as e:
    print(f"Error importando modelo de ML: {e}")
    ML_MODEL_AVAILABLE = False

def clean_numpy_types(obj):
    """Convertir tipos de NumPy a tipos nativos de Python para serialización JSON"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: clean_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_numpy_types(item) for item in obj]
    else:
        return obj

# Configuración de la base de datos
DATABASE_PATH = os.getenv('DATABASE_PATH', '/app/data/patients.db')

# Asegurar que el directorio existe
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)

app = FastAPI(title="Sistema de Diagnóstico de Cáncer de Mama", version="2.0.0")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RealMLModel:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.load_model()
    
    def load_model(self):
        """Cargar el modelo de ML real"""
        try:
            if ML_MODEL_AVAILABLE:
                print("Inicializando modelo EfficientNet...")
                self.model = BreastCancerModel()
                
                # Verificar si existe un modelo pre-entrenado
                model_path = '/app/ml-model/trained_breast_cancer_model.h5'
                if os.path.exists(model_path):
                    print(f"Cargando modelo pre-entrenado desde {model_path}")
                    self.model.load_model(model_path)
                else:
                    print("No se encontró modelo pre-entrenado. Usando modelo base.")
                
                self.model_loaded = True
                print("Modelo cargado exitosamente")
            else:
                print("Modelo de ML no disponible, usando modo simulación")
                self.model_loaded = False
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            self.model_loaded = False
    
    def predict(self, image_bytes):
        """Realizar predicción con el modelo real o simulación mejorada"""
        
        if self.model_loaded and self.model is not None:
            return self._predict_with_real_model(image_bytes)
        else:
            return self._predict_with_improved_simulation(image_bytes)
    
    def _predict_with_real_model(self, image_bytes):
        """Predicción usando el modelo EfficientNet real"""
        try:
            # Convertir bytes a imagen PIL
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Usar el método predict del modelo
            result = self.model.predict(image)
            
            # Agregar detección de bounding box si hay cáncer detectado
            bbox = None
            if result['has_cancer']:
                bbox = self._detect_suspicious_region(image, result['probability'])
            
            return {
                'prediction_score': float(result['probability']),
                'has_cancer': bool(result['has_cancer']),
                'confidence': float(result['confidence']),
                'bbox': bbox,
                'model_type': 'EfficientNet Real'
            }
            
        except Exception as e:
            print(f"Error en predicción real: {e}")
            return self._predict_with_improved_simulation(image_bytes)
    
    def _predict_with_improved_simulation(self, image_bytes):
        """Simulación mejorada y consistente (sin aleatoriedad)"""
        try:
            # Convertir bytes a imagen
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            img_array = np.array(image)
            
            # Calcular características determinísticas de la imagen
            avg_intensity = float(np.mean(img_array))
            contrast = float(np.std(img_array))
            height, width = img_array.shape[:2]
            
            # Crear un hash básico de la imagen para consistencia
            image_hash = hash(img_array.tobytes()) % 1000000
            
            # Predicción determinística basada en características de la imagen
            # (sin componente aleatoria)
            intensity_factor = avg_intensity / 255.0
            contrast_factor = min(contrast / 128.0, 1.0)
            hash_factor = (image_hash % 100) / 100.0
            
            # Combinar factores de manera determinística
            prediction_score = (
                0.2 * intensity_factor + 
                0.4 * contrast_factor + 
                0.4 * hash_factor
            )
            
            # Normalizar entre 0.1 y 0.9
            prediction_score = 0.1 + prediction_score * 0.8
            
            has_cancer = bool(prediction_score > 0.5)
            confidence = float(abs(prediction_score - 0.5) * 2)
            
            # Simular detección de región si hay "cáncer"
            bbox = None
            if has_cancer:
                bbox = self._simulate_bbox(width, height, image_hash)
            
            return {
                'prediction_score': float(prediction_score),
                'has_cancer': has_cancer,
                'confidence': confidence,
                'bbox': bbox,
                'model_type': 'Deterministic Simulation'
            }
            
        except Exception as e:
            print(f"Error en simulación: {e}")
            # Fallback básico
            return {
                'prediction_score': 0.3,
                'has_cancer': False,
                'confidence': 0.6,
                'bbox': None,
                'model_type': 'Fallback'
            }
    
    def _detect_suspicious_region(self, image, prediction_score):
        """Detectar región sospechosa usando técnicas básicas de CV"""
        try:
            import cv2
            
            # Convertir PIL a numpy array
            img_array = np.array(image)
            
            # Convertir a grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Aplicar threshold adaptativo para resaltar regiones anómalas
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Encontrar contornos
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Encontrar contorno con área significativa
                valid_contours = [c for c in contours if cv2.contourArea(c) > 100]
                
                if valid_contours:
                    # Tomar un contorno representativo
                    contour = max(valid_contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    return {
                        'x': int(x),
                        'y': int(y),
                        'width': int(w),
                        'height': int(h)
                    }
            
            return None
            
        except Exception as e:
            print(f"Error en detección de región: {e}")
            return None
    
    def _simulate_bbox(self, width, height, image_hash):
        """Simular bounding box de manera determinística"""
        # Usar hash para generar coordenadas consistentes
        x = (image_hash % 100) * (width - 100) // 100
        y = ((image_hash // 100) % 100) * (height - 100) // 100
        w = 50 + (image_hash % 50)
        h = 50 + ((image_hash // 50) % 50)
        
        return {
            'x': max(0, int(x)),
            'y': max(0, int(y)),
            'width': min(int(w), width - int(x)),
            'height': min(int(h), height - int(y))
        }

# Inicializar modelo real
ml_model = RealMLModel()

# Inicializar base de datos (código igual que antes)
def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cedula TEXT UNIQUE NOT NULL,
            nombre TEXT NOT NULL,
            edad INTEGER NOT NULL,
            tipo_sangre TEXT NOT NULL,
            detalles TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS diagnoses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            image_path TEXT,
            prediction_score REAL,
            has_cancer BOOLEAN,
            bbox_x INTEGER,
            bbox_y INTEGER,
            bbox_width INTEGER,
            bbox_height INTEGER,
            model_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients (id)
        )
    ''')
    
    # Insertar pacientes de prueba
    test_patients = [
        ('0988123456', 'María González', 45, 'O+', 'Paciente con antecedentes familiares'),
        ('0987654321', 'Ana Rodríguez', 52, 'A+', 'Control rutinario'),
        ('0912345678', 'Carmen López', 38, 'B-', 'Primera mamografía'),
        ('0998765432', 'Elena Martínez', 41, 'AB+', 'Seguimiento post-tratamiento'),
        ('0956789123', 'Patricia Silva', 55, 'O-', 'Revisión anual'),
        ('0928728777', 'Leydi Quimi', 22, 'O+', 'Paciente joven para control preventivo'),
    ]
    
    for patient in test_patients:
        cursor.execute('''
            INSERT OR IGNORE INTO patients (cedula, nombre, edad, tipo_sangre, detalles)
            VALUES (?, ?, ?, ?, ?)
        ''', patient)
    
    conn.commit()
    conn.close()

@app.on_event("startup")
async def startup_event():
    init_db()
    print("Sistema iniciado correctamente")
    print(f"Modelo ML cargado: {ml_model.model_loaded}")

@app.get("/")
async def root():
    return {
        "message": "Sistema de Diagnóstico de Cáncer de Mama API v2.0",
        "version": "2.0.0",
        "ml_model_loaded": ml_model.model_loaded,
        "model_type": "EfficientNet" if ml_model.model_loaded else "Simulation",
        "status": "active"
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "ml_model_loaded": ml_model.model_loaded,
        "database": "connected"
    }

@app.get("/api/patient/{cedula}")
async def get_patient(cedula: str):
    """Obtener información del paciente por cédula"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, cedula, nombre, edad, tipo_sangre, detalles
        FROM patients WHERE cedula = ?
    ''', (cedula,))
    
    patient = cursor.fetchone()
    conn.close()
    
    if not patient:
        raise HTTPException(status_code=404, detail="Paciente no encontrado")
    
    return {
        "id": patient[0],
        "cedula": patient[1],
        "nombre": patient[2],
        "edad": patient[3],
        "tipo_sangre": patient[4],
        "detalles": patient[5] or ""
    }

@app.post("/api/diagnose")
async def diagnose_image(
    file: UploadFile = File(...),
    patient_id: int = Form(...)
):
    """Realizar diagnóstico de imagen"""
    
    # Validar tipo de archivo
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
    
    # Validar tamaño del archivo (máximo 10MB)
    content = await file.read()
    file_size = len(content)
    
    if file_size > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(status_code=400, detail="El archivo es demasiado grande (máximo 10MB)")
    
    try:
        # Realizar predicción con el modelo real
        result = ml_model.predict(content)
        
        # Limpiar tipos de NumPy
        result = clean_numpy_types(result)
        
        # Guardar diagnóstico en base de datos
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        bbox = result.get('bbox')
        cursor.execute('''
            INSERT INTO diagnoses (
                patient_id, prediction_score, has_cancer,
                bbox_x, bbox_y, bbox_width, bbox_height, model_type
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            patient_id, 
            result['prediction_score'], 
            result['has_cancer'],
            bbox['x'] if bbox else None,
            bbox['y'] if bbox else None,
            bbox['width'] if bbox else None,
            bbox['height'] if bbox else None,
            result.get('model_type', 'Unknown')
        ))
        
        conn.commit()
        conn.close()
        
        # Generar recomendaciones basadas en el resultado
        diagnosis_text, recommendation = generate_diagnosis_recommendation(result)
        
        # Preparar respuesta final
        response = {
            "prediction_score": result['prediction_score'],
            "has_cancer": result['has_cancer'],
            "confidence": result['confidence'],
            "bbox": result['bbox'],
            "diagnosis": diagnosis_text,
            "recommendation": recommendation,
            "model_used": result.get('model_type', 'Unknown')
        }
        
        return clean_numpy_types(response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")

def generate_diagnosis_recommendation(result):
    """Generar diagnóstico y recomendaciones basadas en el resultado"""
    
    score = result['prediction_score']
    has_cancer = result['has_cancer']
    confidence = result['confidence']
    
    if has_cancer:
        if score > 0.8:
            diagnosis = f"Alta probabilidad de cáncer detectado (Score: {score:.2f})"
            recommendation = "Se recomienda consulta urgente con oncólogo y estudios adicionales"
        elif score > 0.6:
            diagnosis = f"Probabilidad moderada de cáncer (Score: {score:.2f})"
            recommendation = "Se recomienda consulta con especialista y segunda opinión"
        else:
            diagnosis = f"Posible anomalía detectada (Score: {score:.2f})"
            recommendation = "Se recomienda seguimiento cercano y estudios complementarios"
    else:
        if score < 0.2:
            diagnosis = f"No se detecta evidencia de cáncer (Score: {score:.2f})"
            recommendation = "Continuar con controles rutinarios según protocolo"
        else:
            diagnosis = f"Baja probabilidad de cáncer (Score: {score:.2f})"
            recommendation = "Mantener seguimiento regular y controles preventivos"
    
    # Agregar nota sobre confianza
    if confidence < 0.7:
        recommendation += ". Nota: Confianza del modelo moderada, se sugiere validación adicional"
    
    return diagnosis, recommendation

@app.get("/api/patient/{patient_id}/history")
async def get_patient_history(patient_id: int):
    """Obtener historial de diagnósticos del paciente"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT prediction_score, has_cancer, created_at,
               bbox_x, bbox_y, bbox_width, bbox_height, model_type
        FROM diagnoses WHERE patient_id = ?
        ORDER BY created_at DESC
    ''', (patient_id,))
    
    history = cursor.fetchall()
    conn.close()
    
    results = []
    for h in history:
        bbox = None
        if h[3] is not None:  # bbox_x existe
            bbox = {
                'x': h[3],
                'y': h[4],
                'width': h[5],
                'height': h[6]
            }
        
        results.append({
            "prediction_score": h[0],
            "has_cancer": h[1],
            "date": h[2],
            "bbox": bbox,
            "model_type": h[7] if len(h) > 7 else "Unknown"
        })
    
    return results

@app.get("/api/patients")
async def get_all_patients():
    """Obtener lista de todos los pacientes"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, cedula, nombre, edad, tipo_sangre
        FROM patients
        ORDER BY nombre
    ''')
    
    patients = cursor.fetchall()
    conn.close()
    
    return [
        {
            "id": p[0],
            "cedula": p[1],
            "nombre": p[2],
            "edad": p[3],
            "tipo_sangre": p[4]
        } for p in patients
    ]

@app.post("/api/patient")
async def create_patient(
    cedula: str = Form(...),
    nombre: str = Form(...),
    edad: int = Form(...),
    tipo_sangre: str = Form(...),
    detalles: str = Form("")
):
    """Crear nuevo paciente"""
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO patients (cedula, nombre, edad, tipo_sangre, detalles)
            VALUES (?, ?, ?, ?, ?)
        ''', (cedula, nombre, edad, tipo_sangre, detalles))
        
        patient_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return {
            "id": patient_id,
            "cedula": cedula,
            "nombre": nombre,
            "edad": edad,
            "tipo_sangre": tipo_sangre,
            "detalles": detalles,
            "message": "Paciente creado exitosamente"
        }
        
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=400, detail="Ya existe un paciente con esa cédula")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
