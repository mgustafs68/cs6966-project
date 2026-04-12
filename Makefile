# ---- config ----
REPO_NAME  := $(notdir $(CURDIR))
HOME_DIR   ?= $(HOME)
ENV_DIR    := $(HOME_DIR)/conda_envs/$(REPO_NAME)
CONDA_BASE ?= $(HOME_DIR)/miniconda3
CONDA_SH   := $(CONDA_BASE)/etc/profile.d/conda.sh

SHELL := /bin/bash

JUPY_PORT ?= 8888
TMUX_SESS ?= jupy
JUPY_LOG  ?= .jupyter_$(TMUX_SESS).log

PY_CLEAN = env -u LD_LIBRARY_PATH

.PHONY: help
help:
	@echo "Targets:"
	@echo "  make env"
	@echo "  make rebuild"
	@echo "  make activate"
	@echo "  make update"
	@echo "  make verify"
	@echo "  make lock"
	@echo "  make kernel"
	@echo "  make cpu"
	@echo "  make gpu"
	@echo "  make jup"

.PHONY: env
env:
	@echo "[INFO] Creating conda env at $(ENV_DIR)"
	@source "$(CONDA_SH)" && \
	mkdir -p "$(HOME_DIR)/conda_envs" && \
	conda env create --prefix "$(ENV_DIR)" -f environment.yml

.PHONY: rebuild
rebuild:
	@echo "[INFO] Rebuilding conda env at $(ENV_DIR)"
	@source "$(CONDA_SH)" && \
	conda deactivate >/dev/null 2>&1 || true && \
	conda env remove --prefix "$(ENV_DIR)" -y >/dev/null 2>&1 || true && \
	rm -rf "$(ENV_DIR)" && \
	mkdir -p "$(HOME_DIR)/conda_envs" && \
	conda env create --prefix "$(ENV_DIR)" -f environment.yml

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
	conda env update --prefix "$(ENV_DIR)" -f environment.yml --prune

.PHONY: verify
verify:
	@echo "[INFO] Verifying runtime and core imports"
	@source "$(CONDA_SH)" && \
	conda activate "$(ENV_DIR)" && \
	$(PY_CLEAN) python -c "import ssl,ctypes,pandas,torch; import torch.nn.functional as F; print('ssl OK'); print('ctypes OK'); print('pandas', pandas.__version__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('functional OK', F.relu(torch.tensor([-1.0,2.0])))"

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

.PHONY: cpu
cpu:
	salloc -t 00:10:00 \
	--ntasks=1 --nodes=1 -c 1 --mem=8G \
	--partition=coe-class-grn --qos=coe-class-grn --account=cs6953

.PHONY: gpu
gpu:
	salloc --time=00:10:00 \
	--ntasks=1 --gres=gpu:1 --mem=16G --nodes=1 \
	--partition=dlair-gpu-np --qos=cs6953-gpu-np --account=cs6953-gpu-np

.PHONY: jup
jup:
	srun --pty bash -lc '\
	  source "$(CONDA_SH)" && \
	  conda activate "$(ENV_DIR)" && \
	  env -u LD_LIBRARY_PATH jupyter lab --no-browser --ip=0.0.0.0 --port=$(JUPY_PORT) \
	'
