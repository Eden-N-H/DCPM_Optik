#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Defect Inspector — one-shot install script
# Usage: bash install.sh [--cpu]   (pass --cpu to skip CUDA PyTorch)
# ─────────────────────────────────────────────────────────────────────────────
set -e

# ── Colours & helpers ─────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'
YELLOW='\033[1;33m'; BOLD='\033[1m'; RESET='\033[0m'

step()  { echo -e "\n${CYAN}${BOLD}▶  $*${RESET}"; }
ok()    { echo -e "${GREEN}✔  $*${RESET}"; }
warn()  { echo -e "${YELLOW}⚠  $*${RESET}"; }
die()   { echo -e "${RED}✘  $*${RESET}"; exit 1; }

# ── Progress bar (pure bash) ──────────────────────────────────────────────────
progress_bar() {
  local cur=$1 total=$2 label="${3:-}"
  local width=40
  local filled=$(( cur * width / total ))
  local empty=$(( width - filled ))
  local bar=""
  for ((i=0; i<filled; i++)); do bar+="█"; done
  for ((i=0; i<empty;  i++)); do bar+="░"; done
  printf "\r  [${GREEN}%s${RESET}] %3d%%  %s" "$bar" "$(( cur * 100 / total ))" "$label"
  [[ $cur -eq $total ]] && echo || true
}

# ── Wrap pip with a spinner so long installs feel alive ──────────────────────
pip_install() {
  local label="$1"; shift
  echo -e "  ${CYAN}Installing:${RESET} $label"
  pip install --quiet --progress-bar off "$@" &
  local pid=$! i=0
  local spin=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
  while kill -0 "$pid" 2>/dev/null; do
    printf "\r  ${spin[$((i % 10))]}  waiting…"
    sleep 0.12
    (( i++ )) || true
  done
  wait "$pid" && printf "\r  ${GREEN}✔${RESET}  %-40s\n" "$label" \
              || { echo; die "pip install failed for: $label"; }
}

# ── Banner ────────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}╔══════════════════════════════════════╗"
echo -e "║     Defect Inspector — Installer     ║"
echo -e "╚══════════════════════════════════════╝${RESET}"

CPU_ONLY=false
[[ "${1:-}" == "--cpu" ]] && CPU_ONLY=true

# ── Total steps for top-level progress ───────────────────────────────────────
TOTAL=6; CUR=0
show_progress() {
  (( CUR++ )) || true
  progress_bar "$CUR" "$TOTAL" "$1"
  echo
}

# ── Step 1: Python version check ─────────────────────────────────────────────
step "Step 1/6 — Checking Python version"
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)

if [[ $PY_MAJOR -lt 3 || ($PY_MAJOR -eq 3 && $PY_MINOR -lt 10) ]]; then
  die "Python 3.10+ required (found $PY_VER). Install via conda or pyenv."
fi
ok "Python $PY_VER detected"
show_progress "Python check"

# ── Step 2: pip / wheel bootstrap ────────────────────────────────────────────
step "Step 2/6 — Upgrading pip, setuptools, wheel"
pip_install "pip + setuptools + wheel" --upgrade pip setuptools wheel
show_progress "pip bootstrap"

# ── Step 3: PyTorch ──────────────────────────────────────────────────────────
step "Step 3/6 — Installing PyTorch"

if $CPU_ONLY; then
  warn "CPU-only mode requested — SAM3 will be slow (~20-60 s/image)"
  pip_install "torch + torchvision (CPU)" torch torchvision
elif [[ "$(uname -m)" == "arm64" && "$(uname)" == "Darwin" ]]; then
  warn "Apple Silicon detected — using default pip torch (MPS backend)"
  pip_install "torch + torchvision (MPS)" torch torchvision
else
  echo -e "  ${CYAN}Target:${RESET} CUDA 12.8 — edit this script for other CUDA versions"
  pip_install "torch + torchvision (CUDA 12.8)" \
    torch==2.10.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu128
fi
show_progress "PyTorch"

# ── Step 4: SAM3 from GitHub ─────────────────────────────────────────────────
step "Step 4/6 — Installing SAM3 from GitHub"
pip_install "SAM3 (Meta, from source)" \
  "git+https://github.com/facebookresearch/sam3.git"
show_progress "SAM3"

# ── Step 5: remaining requirements ───────────────────────────────────────────
step "Step 5/6 — Installing remaining dependencies"
REQS=(
  "numpy>=1.26,<2"
  "flask>=3.0.0"
  "flask-cors>=4.0.0"
  "ultralytics>=8.3.237"
  "opencv-python-headless==4.10.0.84"
  "Pillow>=10.0.0"
  "huggingface_hub>=0.24.0"
)
TOTAL_REQS=${#REQS[@]}
for i in "${!REQS[@]}"; do
  pkg="${REQS[$i]}"
  progress_bar "$i" "$TOTAL_REQS" "$pkg"
  pip install --quiet --progress-bar off "$pkg" 2>/dev/null \
    || die "Failed to install $pkg"
done
progress_bar "$TOTAL_REQS" "$TOTAL_REQS" "Done"
show_progress "Dependencies"

# ── Step 6: HuggingFace login ─────────────────────────────────────────────────
step "Step 6/6 — HuggingFace authentication"
echo -e "  SAM3 weights (~1.7 GB) are gated on HuggingFace."
echo -e "  1. Visit ${CYAN}https://huggingface.co/facebook/sam3${RESET} and request access"
echo -e "  2. Generate a token at ${CYAN}https://huggingface.co/settings/tokens${RESET}"
echo -e "  3. Run: ${BOLD}huggingface-cli login${RESET}"
echo
read -rp "  Run 'huggingface-cli login' now? [Y/n] " yn
if [[ "${yn:-Y}" =~ ^[Yy]$ ]]; then
  huggingface-cli login || warn "Login failed — run manually before starting the server"
fi
show_progress "HuggingFace"

# ── Done ─────────────────────────────────────────────────────────────────────
echo -e "\n${GREEN}${BOLD}✔  Installation complete!${RESET}"
echo -e "\n  Start the server:\n    ${BOLD}python app.py${RESET}"
echo -e "  Then open:      ${CYAN}http://localhost:5001${RESET}\n"