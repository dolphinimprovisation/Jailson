#!/usr/bin/env bash
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}╔══════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     Instalação do Jailson Agent      ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════╝${NC}"

# Verificar Python 3.11+
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required="3.11"
if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
    echo -e "${GREEN}✓ Python $python_version encontrado${NC}"
else
    echo -e "${RED}✗ Python 3.11+ é necessário. Versão atual: $python_version${NC}"
    exit 1
fi

# Criar ambiente virtual
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}→ Criando ambiente virtual...${NC}"
    python3 -m venv venv
fi
echo -e "${GREEN}✓ Ambiente virtual pronto${NC}"

# Ativar e instalar dependências
source venv/bin/activate
echo -e "${YELLOW}→ Instalando dependências (pode demorar na primeira vez)...${NC}"
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo -e "${GREEN}✓ Dependências instaladas${NC}"

# Criar diretório de dados
DATA_DIR="$HOME/.jailson"
mkdir -p "$DATA_DIR"/{memory,logs}
echo -e "${GREEN}✓ Diretório de dados criado: $DATA_DIR${NC}"

# Criar .env se não existir
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${YELLOW}⚠ Ficheiro .env criado — edita a tua ANTHROPIC_API_KEY!${NC}"
else
    echo -e "${GREEN}✓ Ficheiro .env já existe${NC}"
fi

echo ""
echo -e "${GREEN}╔══════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         Instalação Completa!         ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════╝${NC}"
echo ""
echo -e "Para começar:"
echo -e "  ${YELLOW}1. Edita .env e adiciona a tua ANTHROPIC_API_KEY${NC}"
echo -e "  ${YELLOW}2. source venv/bin/activate${NC}"
echo -e "  ${YELLOW}3. python main.py${NC}"
echo ""
echo -e "Editores recomendados:"
echo -e "  • ${GREEN}Cursor${NC}  — https://cursor.sh  (melhor para AI)"
echo -e "  • ${GREEN}VSCode${NC}  — https://code.visualstudio.com"
echo ""
echo -e "${YELLOW}Nota:${NC} ChromaDB irá descarregar o modelo de embeddings (~80MB) na primeira execução."
