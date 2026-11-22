#!/usr/bin/env bash
# install.sh — full setup for Kinect 360 (model 1473) on Linux
#
# Installs: system deps, patched libfreenect 0.7.x -> ~/.kinect,
#           python venv (.venv) with pyfreenect, kinect_tilt binary,
#           audios.bin firmware (~/.libfreenect), udev rules + blacklist.
#
# Usage: ./install.sh          (will ask sudo for apt/udev steps)
#        ./install.sh --no-firmware   (skip the ~116MB firmware download)

set -euo pipefail

cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"
PREFIX="${HOME}/.kinect"
KINECT_USER="$(id -un)"
FW_DIR="${HOME}/.libfreenect"
SKIP_FW=0
[[ "${1:-}" == "--no-firmware" ]] && SKIP_FW=1

info()  { printf '\033[1;32m[+]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }
fail()  { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- deps
info "checking system dependencies"
MISSING=()
for pkg in build-essential cmake git libusb-1.0-0-dev python3 python3-venv python3-pip udev; do
    dpkg -s "$pkg" >/dev/null 2>&1 || MISSING+=("$pkg")
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
    warn "installing missing packages: ${MISSING[*]}"
    sudo apt-get update -qq
    sudo apt-get install -y "${MISSING[@]}"
fi

# ------------------------------------------------- libfreenect + patches
if [[ ! -d libfreenect ]]; then
    info "cloning libfreenect (master)"
    git clone --depth 1 https://github.com/OpenKinect/libfreenect.git
elif git -C libfreenect describe --always >/dev/null 2>&1; then
    info "libfreenect already cloned"
fi

info "applying K4W/1473 patches"
git -C libfreenect apply --check patches/0001-libfreenect-k4w-1473-fixes.patch 2>/dev/null \
    && git -C libfreenect apply patches/0001-libfreenect-k4w-1473-fixes.patch \
    || info "patch already applied (or dirty tree), skipping"

info "building libfreenect -> ${PREFIX}"
cmake -S libfreenect -B libfreenect/build \
      -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,${PREFIX}/lib" \
      -DBUILD_EXAMPLES=OFF -DBUILD_FAKENECT=OFF -DBUILD_REDIST_PACK=OFF
make -C libfreenect/build -j"$(nproc)" install

# ---------------------------------------------------------------- venv
info "setting up python venv with pyfreenect"
[[ -d .venv ]] || python3 -m venv .venv
.venv/bin/pip install -q -U pip setuptools wheel numpy
.venv/bin/pip install -q "cython<3.1"          # setup.py of 0.7.5 breaks with Cython >= 3.1

CFLAGS="-I${PREFIX}/include/libfreenect" \
LDFLAGS="-L${PREFIX}/lib -Wl,-rpath,${PREFIX}/lib" \
    .venv/bin/pip install --no-build-isolation -q ./libfreenect/wrappers/python

# ------------------------------------------------------- kinect_tilt
info "building kinect_tilt"
gcc -I"${PREFIX}/include/libfreenect" kinect_tilt.c -o kinect_tilt \
    -L"${PREFIX}/lib" -Wl,-rpath,"${PREFIX}/lib" -lfreenect

info "building kinect_hub (camera + mics + motor daemon for jarvis.py)"
gcc -O2 -I"${PREFIX}/include/libfreenect" kinect_hub.c -o kinect_hub \
    -L"${PREFIX}/lib" -Wl,-rpath,"${PREFIX}/lib" -lfreenect -lm

info "installing jarvis deps (opencv, faster-whisper)"
.venv/bin/pip install -q opencv-python-headless faster-whisper

# ------------------------------------------------------------- firmware
if [[ $SKIP_FW -eq 1 ]]; then
    warn "--no-firmware: skipping audios.bin download"
elif [[ -f "${FW_DIR}/audios.bin" ]]; then
    info "audios.bin already present in ${FW_DIR}"
else
    info "fetching audios.bin from xbox.com (~116 MB, one time)"
    info "this firmware powers tilt/LED on the model 1473; without it only depth/RGB work"
    mkdir -p "${FW_DIR}"
    ( cd libfreenect && python3 src/fwfetcher.py )
    cp libfreenect/audios.bin "${FW_DIR}/audios.bin"
fi

# ------------------------------------------------------------ udev etc.
info "writing udev rules (permissions + no autosuspend for Kinect)"
sudo tee /etc/udev/rules.d/99-kinect.rules >/dev/null <<'EOF'
# Microsoft Kinect 360 / Kinect for Windows (libfreenect)
SUBSYSTEM=="usb", ATTRS{idVendor}=="045e", ATTRS{idProduct}=="02ad", MODE="0666", ATTR{power/control}:="on"
SUBSYSTEM=="usb", ATTRS{idVendor}=="045e", ATTRS{idProduct}=="02ae", MODE="0666", ATTR{power/control}:="on"
SUBSYSTEM=="usb", ATTRS{idVendor}=="045e", ATTRS{idProduct}=="02c2", MODE="0666", ATTR{power/control}:="on"
EOF

info "blacklisting gspca_kinect (kernel driver fights libfreenect for the camera)"
echo "blacklist gspca_kinect" | sudo tee /etc/modprobe.d/no-kinect-gspca.conf >/dev/null
if lsmod | grep -q gspca_kinect; then
    sudo rmmod gspca_kinect 2>/dev/null || true
fi
sudo udevadm control --reload
sudo udevadm trigger --subsystem-match=usb --attr-match=idVendor=045e 2>/dev/null || true

# ---------------------------------------------------------------- check
info "smoke test: pyfreenect import"
.venv/bin/python - <<'PY'
import ctypes, os
lib = os.path.expanduser("~/.kinect/lib")
ctypes.CDLL(os.path.join(lib, "libfreenect.so.0"), mode=ctypes.RTLD_GLOBAL)
import freenect
print("    pyfreenect:", freenect.__file__)
PY

cat <<EOF

Done! Replug the Kinect (or not) and run:

    .venv/bin/python color_depth.py      # truecolor half-block visualizer
    .venv/bin/python ascii_depth.py      # plain ASCII depth art
    ./kinect_tilt 15                     # tilt the motor manually
    .venv/bin/python jarvis.py           # voice assistant (needs an OpenAI-compatible LLM on :8080)

Notes:
  * depth+RGB do not stream simultaneously on the 1473; toggle with 'm'
  * audio firmware (tilt/LED) re-flashes on the first open after replug
  * if lsusb shows only 'Motor', the Kinect needs its 12V power adapter
EOF
