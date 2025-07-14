#!/bin/bash

echo "========================================"
echo "  SOLUCIONADOR DE PROBLEMAS DEL SISTEMA"
echo "========================================"
echo

cd "$(dirname "$0")"

echo "[1/4] Diagnosticando sistema..."
python3 diagnose_system.py

echo
echo "[2/4] Solucionando problema del backend..."
if [ -f "main_fixed.py" ]; then
    cp main_fixed.py main.py
    echo "✅ Backend actualizado con version corregida"
else
    echo "❌ main_fixed.py no encontrado"
fi

echo
echo "[3/4] Entrenando modelo rapido..."
python3 train_model.py --quick

echo
echo "[4/4] Verificando solucion..."
python3 diagnose_system.py

echo
echo "========================================"
echo "  SOLUCION COMPLETADA"
echo "========================================"
echo "Para probar el sistema:"
echo "  1. python3 main.py"
echo "  2. Abrir frontend en otra terminal"
echo "  3. Subir la misma imagen varias veces"
echo "  4. Verificar que los resultados sean consistentes"
echo

read -p "Presiona Enter para continuar..."
