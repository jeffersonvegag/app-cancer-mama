@echo off
echo.
echo 🧹 LIMPIEZA COMPLETA DEL SISTEMA
echo ================================
echo.

cd /d "%~dp0"

echo 🛑 Deteniendo contenedores existentes...
docker-compose down 2>nul

echo 🗑️ Eliminando contenedores viejos...
docker rm -f app-cruz-ml-backend app-cruz-ml-frontend app-cruz-ml-backend-v2 app-cruz-ml-frontend-v2 2>nul

echo 🗑️ Eliminando imágenes viejas...
docker rmi -f app-cruz-ml-backend:latest app-cruz-ml-frontend:latest app-cruz-ml-backend:v2 app-cruz-ml-frontend:v2 2>nul

echo 🗑️ Eliminando volúmenes viejos (incluyendo base de datos)...
docker volume rm app-cruz-ml-db-data app-cruz-ml-node-modules app-cruz-ml-db-data-v2 app-cruz-ml-node-modules-v2 2>nul

echo 🗑️ Eliminando redes viejas...
docker network rm app-cruz-ml-network app-cruz-ml-network-v2 2>nul

echo 🧹 Limpieza de Docker completa...
docker system prune -f

echo.
echo 🚀 Iniciando sistema limpio...
docker-compose up --build -d

echo.
echo ✅ Sistema iniciado desde cero!
echo.
echo 🔗 URLs disponibles:
echo    - Backend API: http://localhost:8000
echo    - Frontend: http://localhost:3000
echo    - Health Check: http://localhost:8000/api/health
echo.
echo 📊 Para verificar logs:
echo    docker-compose logs -f backend
echo    docker-compose logs -f frontend
echo.
pause
