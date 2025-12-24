#!/bin/bash
# =============================================================================
# Pack for RunPod - Cria arquivo ZIP para upload
# =============================================================================
# Uso (Windows com Git Bash ou WSL):
#   cd extracao
#   bash deploy/pack_for_runpod.sh
# =============================================================================

echo "Criando pacote para RunPod..."

# Diretório base
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

# Nome do arquivo
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="rag-pipeline-${TIMESTAMP}.zip"

# Criar ZIP excluindo arquivos desnecessários
zip -r "$OUTPUT_FILE" \
    src/ \
    scripts/ \
    data/*.pdf \
    deploy/ \
    pyproject.toml \
    -x "*.pyc" \
    -x "*__pycache__*" \
    -x "*.egg-info*" \
    -x ".venv/*" \
    -x "*.log" \
    -x "data/output/*" \
    -x "*.tmp"

echo ""
echo "=============================================="
echo "Pacote criado: $OUTPUT_FILE"
echo "Tamanho: $(du -h "$OUTPUT_FILE" | cut -f1)"
echo "=============================================="
echo ""
echo "Para fazer upload no RunPod:"
echo "  scp $OUTPUT_FILE root@<pod-ip>:/workspace/"
echo ""
echo "No RunPod, descompacte com:"
echo "  cd /workspace && unzip $OUTPUT_FILE -d rag-pipeline"
echo ""
