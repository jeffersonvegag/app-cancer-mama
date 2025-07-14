import React, { useState } from 'react';
import axios from 'axios';

interface Patient {
  id: number;
  cedula: string;
  nombre: string;
  edad: number;
  tipo_sangre: string;
  detalles: string;
}

interface DiagnosisResult {
  prediction_score: number;
  has_cancer: boolean;
  confidence: number;
  bbox?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  diagnosis: string;
  recommendation: string;
}

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [cedula, setCedula] = useState('');
  const [patient, setPatient] = useState<Patient | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [diagnosisResult, setDiagnosisResult] = useState<DiagnosisResult | null>(null);
  const [diagnosisLoading, setDiagnosisLoading] = useState(false);

  const searchPatient = async () => {
    if (!cedula.trim()) {
      setError('Por favor ingrese un número de cédula');
      return;
    }

    setLoading(true);
    setError('');
    setPatient(null);

    try {
      const response = await axios.get(`${API_BASE_URL}/api/patient/${cedula}`);
      setPatient(response.data);
    } catch (err: any) {
      if (err.response?.status === 404) {
        setError('Paciente no encontrado');
      } else {
        setError('Error al buscar paciente. Verifique la conexión.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      
      // Crear preview de la imagen
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
      
      // Limpiar resultados anteriores
      setDiagnosisResult(null);
    }
  };

  const performDiagnosis = async () => {
    if (!selectedFile || !patient) {
      setError('Seleccione una imagen y asegúrese de tener un paciente cargado');
      return;
    }

    setDiagnosisLoading(true);
    setError('');

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('patient_id', patient.id.toString());

    try {
      const response = await axios.post(`${API_BASE_URL}/api/diagnose`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setDiagnosisResult(response.data);
    } catch (err: any) {
      setError('Error al procesar la imagen. Intente nuevamente.');
      console.error('Diagnosis error:', err);
    } finally {
      setDiagnosisLoading(false);
    }
  };

  const renderBoundingBox = () => {
    if (!diagnosisResult?.bbox || !imagePreview) return null;

    const bbox = diagnosisResult.bbox;
    return (
      <div
        className="bbox-overlay"
        style={{
          left: `${bbox.x}px`,
          top: `${bbox.y}px`,
          width: `${bbox.width}px`,
          height: `${bbox.height}px`,
        }}
      />
    );
  };

  return (
    <div className="container">
      <div className="header">
        <h1>Sistema de Diagnóstico de Cáncer de Mama</h1>
        <p>Centro Obstétrico - Guayaquil</p>
      </div>

      <div className="content">
        {/* Sección de búsqueda */}
        <div className="search-section">
          <h2>Búsqueda de Paciente</h2>
          <div className="search-form">
            <input
              type="text"
              className="input-field"
              placeholder="Ingrese número de cédula"
              value={cedula}
              onChange={(e) => setCedula(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && searchPatient()}
            />
            <button 
              className="btn btn-primary" 
              onClick={searchPatient}
              disabled={loading}
            >
              {loading ? 'Buscando...' : 'Buscar'}
            </button>
          </div>
        </div>

        {error && <div className="error">{error}</div>}

        {patient && (
          <>
            {/* Información del paciente y mamografía */}
            <div className="patient-info">
              {/* Sección de mamografía */}
              <div className="mammography-section">
                {imagePreview ? (
                  <div className="image-preview">
                    <img 
                      src={imagePreview} 
                      alt="Mamografía" 
                      className="mammography-image"
                    />
                    {renderBoundingBox()}
                  </div>
                ) : (
                  <div className="mammography-placeholder">
                    <h3>Mamografía del Paciente</h3>
                    <p>La imagen aparecerá aquí cuando suba un archivo</p>
                  </div>
                )}
              </div>

              {/* Datos del paciente */}
              <div className="patient-details">
                <h3>Datos del Paciente</h3>
                <div className="detail-row">
                  <span className="detail-label">Cédula:</span>
                  <span className="detail-value">{patient.cedula}</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">Nombre:</span>
                  <span className="detail-value">{patient.nombre}</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">Edad:</span>
                  <span className="detail-value">{patient.edad} años</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">Tipo de Sangre:</span>
                  <span className="detail-value">{patient.tipo_sangre}</span>
                </div>
                
                <textarea
                  className="details-textarea"
                  placeholder="Detalles adicionales (opcional)"
                  defaultValue={patient.detalles}
                  readOnly
                />
              </div>
            </div>

            {/* Sección de carga de imagen */}
            <div className="upload-section">
              <h3>Subir Mamografía</h3>
              <div className="file-input">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                />
              </div>
              
              {selectedFile && (
                <button 
                  className="btn btn-success" 
                  onClick={performDiagnosis}
                  disabled={diagnosisLoading}
                >
                  {diagnosisLoading ? 'Analizando...' : 'Obtener Resultados'}
                </button>
              )}
            </div>

            {/* Resultados del diagnóstico */}
            {diagnosisResult && (
              <div className={`results-section ${
                diagnosisResult.has_cancer ? 'results-danger' : 'results-success'
              }`}>
                <h3>Resultados del Diagnóstico</h3>
                
                <div className="prediction-score">
                  Puntuación de Predicción: {(diagnosisResult.prediction_score * 100).toFixed(1)}%
                </div>
                
                <div className="confidence-bar">
                  <div 
                    className="confidence-fill"
                    style={{ width: `${diagnosisResult.confidence * 100}%` }}
                  />
                </div>
                
                <p><strong>Diagnóstico:</strong> {diagnosisResult.diagnosis}</p>
                <p><strong>Recomendación:</strong> {diagnosisResult.recommendation}</p>
                
                {diagnosisResult.has_cancer && diagnosisResult.bbox && (
                  <p><strong>Región sospechosa detectada</strong> (marcada en rojo en la imagen)</p>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default App;
