# ---- config ----
REPO_NAME  := $(notdir $(CURDIR))
SCRATCH    ?= /scratch/general/vast/$(USER)
ENV_DIR    := $(SCRATCH)/conda_envs/$(REPO_NAME)
CONDA_BASE ?= $(SCRATCH)/miniconda3
CONDA_SH   := $(CONDA_BASE)/etc/profile.d/conda.sh

SHELL := /bin/bash

JUPY_PORT ?= 8888
TMUX_SESS ?= jupy
JUPY_LOG  ?= .jupyter_$(TMUX_SESS).log

# Always use a clean shell environment for Python runs.
PY_CLEAN = env -u LD_LIBRARY_PATH

.PHONY: help
help:
	@echo "Targets:"
	@echo "  make env        - create env from scratch only if missing"
	@echo "  make rebuild    - fully remove and recreate env at same path"
	@echo "  make activate   - open shell with env activated"
	@echo "  make update     - update env from environment.yml"
	@echo "  make verify     - verify core imports/runtime"
	@echo "  make lock       - export yaml + explicit lock files"
	@echo "  make kernel     - register Jupyter kernel"
	@echo "  make destroy    - remove env completely"
	@echo "  make clone      - clone current env to a test env"
	@echo "  make cpu        - request CPU interactive job"
	@echo "  make gpu        - request GPU interactive job"
	@echo "  make jup        - launch Jupyter Lab"

.PHONY: env
env:
	@source "$(CONDA_SH)" && \
	mkdir -p "$(SCRATCH)/conda_envs" && \
	if [ -d "$(ENV_DIR)" ]; then \
	  echo "[ERROR] Env already exists at $(ENV_DIR)"; \
	  echo "        Use 'make rebuild' to recreate it cleanly."; \
	  exit 1; \
	fi && \
	echo "[INFO] Creating conda env at $(ENV_DIR)" && \
	conda env create --prefix "$(ENV_DIR)" -f environment.yml && \
	$(MAKE) verify

.PHONY: rebuild
rebuild:
	@source "$(CONDA_SH)" && \
	echo "[INFO] Rebuilding conda env at $(ENV_DIR)" && \
	conda deactivate >/dev/null 2>&1 || true && \
	conda env remove --prefix "$(ENV_DIR)" -y >/dev/null 2>&1 || true && \
	rm -rf "$(ENV_DIR)" && \
	mkdir -p "$(SCRATCH)/conda_envs" && \
	conda env create --prefix "$(ENV_DIR)" -f environment.yml && \
	$(MAKE) verify

.PHONY: activate
activate:
	@echo "[INFO] Activating conda env at $(ENV_DIR)"
	@source "$(CONDA_SH)" && \
	conda activate "$(ENV_DIR)" && \
	exec $$SHELL

.PHONY: update
update:
	@echo "[INFO] Updating conda env at $(ENV_DIR)"
	@source "$(CONDA_SH)" && \
	conda env update --prefix "$(ENV_DIR)" -f environment.yml --prune && \
	$(MAKE) verify

.PHONY: verify
verify:
	@echo "[INFO] Verifying runtime and core imports"
	@source "$(CONDA_SH)" && \
	conda activate "$(ENV_DIR)" && \
	$(PY_CLEAN) python - <<'PY'
import sys
print("python:", sys.executable)

import ssl
import ctypes
import six
import packaging
from dateutil.tz import gettz
import pandas
import torch
import torch.nn.functional as F
import transformers
import peft
import huggingface_hub

print("ssl OK")
print("ctypes OK")
print("six OK")
print("packaging OK:", packaging.__file__)
print("gettz OK:", bool(gettz("UTC")))
print("pandas:", pandas.__version__)
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("peft:", peft.__version__)
print("huggingface_hub:", huggingface_hub.__version__)
print("torch.nn.functional OK:", F.relu(torch.tensor([-1.0, 2.0])))
PY

.PHONY: lock
lock:
	@echo "[INFO] Exporting locked environment"
	@source "$(CONDA_SH)" && \
	conda env export --prefix "$(ENV_DIR)" > environment.lock.yml && \
	conda list --explicit --prefix "$(ENV_DIR)" > environment.explicit.txt

.PHONY: kernel
kernel:
	@echo "[INFO] Registering Jupyter kernel"
	@source "$(CONDA_SH)" && \
	conda activate "$(ENV_DIR)" && \
	$(PY_CLEAN) python -m ipykernel install --user \
	  --name "$(REPO_NAME)" \
	  --display-name "CHPC: $(REPO_NAME)"

.PHONY: destroy
destroy:
	@echo "[INFO] Removing conda env at $(ENV_DIR)"
	@source "$(CONDA_SH)" && \
	conda deactivate >/dev/null 2>&1 || true && \
	conda env remove --prefix "$(ENV_DIR)" -y >/dev/null 2>&1 || true && \
	rm -rf "$(ENV_DIR)"

.PHONY: clone
clone:
	@source "$(CONDA_SH)" && \
	TEST_ENV="$(ENV_DIR)-test" && \
	echo "[INFO] Cloning env to $$TEST_ENV" && \
	conda create -y --prefix "$$TEST_ENV" --clone "$(ENV_DIR)"

.PHONY: cpu
cpu:
	salloc -t 01:00:00 \
	--ntasks=1 --nodes=1 -c 4 --mem=32G \
	--partition=coe-class-grn --qos=coe-class-grn --account=cs6953

.PHONY: gpu
gpu:
	salloc --time=01:00:00 \
	--ntasks=1 --nodes=1 -c 4 --mem=32G --gres=gpu:1 \
	--partition=coe-gpu-class-grn --qos=coe-gpu-students-grn --account=cs6953

.PHONY: jup
jup:
	srun --pty bash -lc '\
	  source "$(CONDA_SH)" && \
	  conda activate "$(ENV_DIR)" && \
	  env -u LD_LIBRARY_PATH jupyter lab --no-browser --ip=0.0.0.0 --port=$(JUPY_PORT) \
	'