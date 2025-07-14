@echo off
echo ========================================
echo   SOLUCIONADOR DE PROBLEMAS DEL SISTEMA
echo ========================================
echo.

cd /d "%~dp0"

echo [1/4] Diagnosticando sistema...
python diagnose_system.py

echo.
echo [2/4] Solucionando problema del backend...
if exist main_fixed.py (
    copy main_fixed.py main.py
    echo ✅ Backend actualizado con version corregida
) else (
    echo ❌ main_fixed.py no encontrado
)

echo.
echo [3/4] Entrenando modelo rapido...
python train_model.py --quick

echo.
echo [4/4] Verificando solucion...
python diagnose_system.py

echo.
echo ========================================
echo   SOLUCION COMPLETADA
echo ========================================
echo Para probar el sistema:
echo   1. python main.py
echo   2. Abrir frontend en otra terminal
echo   3. Subir la misma imagen varias veces
echo   4. Verificar que los resultados sean consistentes
echo.
pause
