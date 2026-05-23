#!/bin/bash
# =========================================================
#  AI Personal Finance Advisor v2.0 — INR Edition
#  Quick Start Script
# =========================================================
echo ""
echo "  ╔═══════════════════════════════════════════╗"
echo "  ║   AI PERSONAL FINANCE ADVISOR v2.0        ║"
echo "  ║   INR Edition  |  ML-Powered               ║"
echo "  ╠═══════════════════════════════════════════╣"
echo "  ║  Demo Login:  demo / demo123               ║"
echo "  ║  URL:  http://127.0.0.1:5000              ║"
echo "  ╚═══════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found. Please install Python 3.9+."
    exit 1
fi

# Install requirements
echo "[*] Installing dependencies..."
pip install Flask Flask-CORS Flask-SQLAlchemy bcrypt PyJWT scikit-learn pandas numpy --break-system-packages -q

# Run the app
echo "[*] Starting server at http://127.0.0.1:5000 ..."
cd "$(dirname "$0")/backend"
python3 app.py
