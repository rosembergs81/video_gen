#!/bin/bash
# ============================================================
#  setup.sh — Instala todo en una instancia Vast.ai
#  Ejecutar con: bash setup.sh
#
#  IMPORTANTE: Si tienes un token de HuggingFace, expórtalo
#  ANTES de ejecutar este script:
#    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#    bash setup.sh
#
#  O pásalo como argumento: bash setup.sh hf_xxxxxxxxxxxxxxxx
# ============================================================
set -e

echo "=============================================="
echo " 🎬  Video Generator — Setup para Vast.ai"
echo "=============================================="

# ── 0. HuggingFace Token ──────────────────────────
# Acepta el token como argumento o de la variable de entorno
if [ -n "$1" ]; then
    export HF_TOKEN="$1"
fi

if [ -n "$HF_TOKEN" ]; then
    echo "  ✅ HF_TOKEN detectado (longitud=${#HF_TOKEN})"
else
    echo "  ⚠️  HF_TOKEN no definido."
    echo "     Para descargar modelos con gating, ejecuta:"
    echo "     export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    echo "     bash setup.sh"
fi

# ── 1. Actualizar sistema ──────────────────────
echo "[1/6] Actualizando paquetes del sistema…"
apt-get update -qq && apt-get install -y -qq \
    git wget curl ffmpeg libgl1 libglib2.0-0 \
    python3-pip python3-venv > /dev/null 2>&1
echo "  ✅ Sistema actualizado"

# ── 2. Entorno virtual ────────────────────────
echo "[2/6] Creando entorno virtual Python…"
# Agregamos --system-site-packages para heredar el PyTorch global de la imagen Docker de Vast.ai
python3 -m venv --system-site-packages /workspace/venv
source /workspace/venv/bin/activate
pip install --upgrade pip -q

# Persistir HF_TOKEN en el entorno virtual para sesiones futuras
if [ -n "$HF_TOKEN" ]; then
    echo "export HF_TOKEN=\"${HF_TOKEN}\"" >> /workspace/venv/bin/activate
    echo "export HUGGINGFACE_TOKEN=\"${HF_TOKEN}\"" >> /workspace/venv/bin/activate
    echo "  ✅ HF_TOKEN guardado en el entorno virtual"
fi
echo "  ✅ Entorno listo en /workspace/venv"

# ── 3. PyTorch con CUDA ───────────────────────
# Omitido: La imagen Docker de Vast.ai ya incluye PyTorch+CUDA por defecto.
echo "[3/6] Omitiendo instalacion de PyTorch (Incluido en imagen Vast.ai)..."

# ── 4. Dependencias del proyecto ──────────────
echo "[4/6] Instalando dependencias…"
pip install -r /workspace/video_gen/requirements.txt -q
echo "  ✅ Dependencias instaladas"

# ── 5. Crear directorios necesarios ──────────────
echo "[5/6] Configurando estructura del proyecto…"
mkdir -p /workspace/video_gen/outputs
mkdir -p /workspace/video_gen/loras

# Si no existe app.py en /workspace, copiar desde directorio actual
if [ ! -f /workspace/video_gen/app.py ]; then
    cp -r ./* /workspace/video_gen/ 2>/dev/null || true
fi
echo "  ✅ Directorios creados"

# ── 6. Configurar inicio automático ──────────
echo "[6/6] Configurando servicio de inicio…"
cat > /workspace/start.sh << 'EOF'
#!/bin/bash
source /workspace/venv/bin/activate
cd /workspace/video_gen
# HF_TOKEN ya está en el entorno virtual si se configuró durante el setup.
# Si quieres sobreescribirlo en tiempo de ejecución:
#   export HF_TOKEN=hf_nueva_clave && bash /workspace/start.sh
python app.py --share
EOF
chmod +x /workspace/start.sh
echo "  ✅ Script de inicio creado en /workspace/start.sh"

echo ""
echo "=============================================="
echo "  ✅  Setup completado con éxito!"
echo "=============================================="
echo ""
echo "  Para iniciar la app:"
echo "    bash /workspace/start.sh"
echo ""
echo "  Con link público (Gradio Share):"
echo "    cd /workspace/video_gen && python app.py --share"
echo ""
echo "  Con autenticación:"
echo "    cd /workspace/video_gen && python app.py --auth usuario password"
echo ""
echo "  Token HuggingFace en tiempo de ejecución:"
echo "    export HF_TOKEN=hf_xxxxx && python app.py"
echo "    # O con generate.py:"
echo "    python generate.py --prompt '...' --hf-token hf_xxxxx"
echo ""
echo "  La UI estará disponible en:"
echo "    http://<IP_VASTAI>:7860"
echo "=============================================="
