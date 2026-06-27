#!/bin/bash
# SentinexFace: Script de inicio automático
# Este script mata cualquier proceso anterior y levanta todo desde cero.

echo "======================================"
echo "SentinexFace - Iniciando servicios..."
echo "======================================"

# 1. Matar procesos anteriores (por puerto)
echo "1. Limpiando servicios anteriores..."

# Matar backend Vision (Puerto 8890)
PID_VISION=$(lsof -t -i:8890)
if [ ! -z "$PID_VISION" ]; then
    echo " -> Matando Backend Vision (PID $PID_VISION)"
    kill -9 $PID_VISION
fi

# Matar RTSP Bridge (Puerto 8787)
PID_RTSP=$(lsof -t -i:8787)
if [ ! -z "$PID_RTSP" ]; then
    echo " -> Matando RTSP Bridge (PID $PID_RTSP)"
    kill -9 $PID_RTSP
fi

# Matar Frontend Vite (Puerto 5173)
PID_VITE=$(lsof -t -i:5173)
if [ ! -z "$PID_VITE" ]; then
    echo " -> Matando Frontend Vite (PID $PID_VITE)"
    kill -9 $PID_VITE
fi

echo "Limpieza completada."
echo ""

# 2. Iniciar procesos
echo "2. Levantando servicios..."

# Iniciar RTSP Server en background
echo " -> Iniciando RTSP Bridge (npm run dev:rtsp)..."
npm run dev:rtsp > rtsp.log 2>&1 &
sleep 2

# Iniciar Backend Vision en background
echo " -> Iniciando Backend Vision (npm run dev:vision)..."
npm run dev:vision > vision.log 2>&1 &
sleep 2

# Iniciar Frontend Vite en background
echo " -> Iniciando Frontend React (npm run dev)..."
npm run dev > vite.log 2>&1 &

echo ""
echo "======================================"
echo "¡Todo listo! Los servicios están corriendo."
echo "Frontend UI : http://localhost:5173"
echo "Backend API : http://localhost:8890"
echo "RTSP Bridge : http://localhost:8787"
echo "======================================"
echo "Puedes cerrar esta terminal, los procesos quedaron en background."
echo "Si quieres ver los logs, revisa vite.log, vision.log y rtsp.log"
