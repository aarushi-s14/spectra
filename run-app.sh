#!/bin/zsh
# Launch SPECTRA. Requires Python 3.12+ with numpy, scipy, opencv-python, sklearn, fastapi, uvicorn, pillow.
# Example setup:
#   uv venv --python 3.12 .venv
#   uv pip install --python .venv/bin/python numpy scipy opencv-python-headless scikit-learn fastapi "uvicorn[standard]" pillow
#   ./.venv/bin/python -m uvicorn server:app --port 8642
cd "$(dirname "$0")"
exec python3 -m uvicorn server:app --port 8642 "${@}"
