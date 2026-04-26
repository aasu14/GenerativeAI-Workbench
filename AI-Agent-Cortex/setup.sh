#!/bin/bash
set -e

echo "================================================="
echo "  AI Agent Orchestration Platform - Setup"
echo "================================================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed."
    exit 1
fi

if ! command -v node &> /dev/null; then
    echo "Node.js is required but not installed."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
NODE_VERSION=$(node --version 2>&1)
echo -e "${GREEN}Python: $PYTHON_VERSION${NC}"
echo -e "${GREEN}Node.js: $NODE_VERSION${NC}"

# Backend setup
echo -e "\n${BLUE}Setting up backend...${NC}"
cd backend

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created${NC}"
fi

source venv/bin/activate
pip install -r requirements.txt --quiet
echo -e "${GREEN}Backend dependencies installed${NC}"

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${YELLOW}Created .env from template. Please edit backend/.env with your API keys.${NC}"
fi

cd ..

# Frontend setup
echo -e "\n${BLUE}Setting up frontend...${NC}"
cd frontend
npm install --silent
echo -e "${GREEN}Frontend dependencies installed${NC}"
cd ..

echo -e "\n${GREEN}=================================================${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}=================================================${NC}"
echo -e ""
echo -e "To start the platform:"
echo -e ""
echo -e "  ${BLUE}Terminal 1 (Backend):${NC}"
echo -e "    cd backend && source venv/bin/activate"
echo -e "    uvicorn main:app --reload --port 8000"
echo -e ""
echo -e "  ${BLUE}Terminal 2 (Frontend):${NC}"
echo -e "    cd frontend && npm run dev"
echo -e ""
echo -e "  Then open ${GREEN}http://localhost:5173${NC} in your browser"
echo -e ""
echo -e "${YELLOW}Don't forget to set your OPENAI_API_KEY in backend/.env${NC}"
echo -e "${YELLOW}For Telegram integration, set TELEGRAM_BOT_TOKEN too${NC}"
