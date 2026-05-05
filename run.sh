#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║   SVM Margin Visualiser — macOS Apple CLI Launcher              ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# Usage:
#   chmod +x run.sh && ./run.sh
#   ./run.sh --port 8502
#   ./run.sh --browser false     # headless server mode
#   ./run.sh --reset-venv        # wipe & recreate virtual environment

set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Colours ──────────────────────────────────────────────────────────
YLW='\033[0;33m'; GRN='\033[0;32m'; CYN='\033[0;36m'
RED='\033[0;31m'; DIM='\033[2m'; NC='\033[0m'
hdr()  { echo -e "\n${YLW}$*${NC}"; }
ok()   { echo -e "  ${GRN}✔${NC}  $*"; }
info() { echo -e "  ${CYN}▸${NC}  $*"; }
die()  { echo -e "  ${RED}✘${NC}  $*"; exit 1; }

echo ""
echo -e "${YLW}╔══════════════════════════════════════════════════════╗"
echo -e "║   SVM MARGIN VISUALISER  ·  Kite Connect Edition    ║"
echo -e "╚══════════════════════════════════════════════════════╝${NC}"

# ── Parse flags ──────────────────────────────────────────────────────
PORT=8501; OPEN_BROWSER=true; RESET_VENV=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)       PORT=$2;           shift 2 ;;
        --browser)    OPEN_BROWSER=$2;   shift 2 ;;
        --reset-venv) RESET_VENV=true;   shift   ;;
        *) die "Unknown argument: $1" ;;
    esac
done

# ── Python check ─────────────────────────────────────────────────────
command -v python3 &>/dev/null || die "python3 not found. brew install python"
PY=$( command -v python3)
PV=$($PY -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python $PV  ($PY)"
[[ "$(printf '%s\n' 3.9 "$PV" | sort -V | head -1)" == "3.9" ]] || \
    die "Python >= 3.9 required"

# ── Virtual environment ───────────────────────────────────────────────
VENV="$DIR/.venv"
if $RESET_VENV && [[ -d "$VENV" ]]; then
    info "Removing existing venv …"; rm -rf "$VENV"
fi
if [[ ! -d "$VENV" ]]; then
    info "Creating virtual environment …"
    $PY -m venv "$VENV"
    ok "venv created"
fi
PIP="$VENV/bin/pip"
PYTHON="$VENV/bin/python"
STREAMLIT="$VENV/bin/streamlit"

# ── Dependencies ──────────────────────────────────────────────────────
info "Syncing dependencies …"
"$PIP" install --quiet --upgrade pip
"$PIP" install --quiet -r "$DIR/requirements.txt"
ok "Dependencies ready"

# ── .env ─────────────────────────────────────────────────────────────
if [[ -f "$DIR/.env" ]]; then
    set -o allexport; source "$DIR/.env"; set +o allexport
    ok ".env loaded"
else
    echo -e "\n  ${YLW}⚠${NC}  .env not found — copy .env.example → .env and fill in credentials"
fi

[[ -n "${KITE_API_KEY:-}"      ]] && ok "KITE_API_KEY detected" || \
    echo -e "  ${DIM}KITE_API_KEY not set — enter on the Auth page${NC}"
[[ -n "${KITE_ACCESS_TOKEN:-}" ]] && ok "KITE_ACCESS_TOKEN detected" || \
    echo -e "  ${DIM}KITE_ACCESS_TOKEN not set — enter on the Auth page${NC}"

# ── Launch ───────────────────────────────────────────────────────────
hdr "Launching Streamlit on port $PORT …"
echo -e "  ${CYN}URL:${NC} http://localhost:$PORT\n"

"$STREAMLIT" run "$DIR/app.py" \
    --server.port          "$PORT" \
    --server.headless      "$( $OPEN_BROWSER && echo false || echo true )" \
    --theme.base           dark \
    --theme.primaryColor   "#ff8c00" \
    --theme.backgroundColor "#0b0f14" \
    --theme.secondaryBackgroundColor "#111820" \
    --theme.textColor      "#cdd6e0" \
    --theme.font           monospace
