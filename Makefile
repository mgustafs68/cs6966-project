# ---- config ----
REPO_NAME := $(notdir $(CURDIR))
SCRATCH   ?= /scratch/general/vast/$(USER)
ENV_DIR   := $(SCRATCH)/conda_envs/$(REPO_NAME)
CONDA_BASE ?= $(SCRATCH)/miniconda3
CONDA_SH   := $(CONDA_BASE)/etc/profile.d/conda.sh

SHELL := /bin/bash

JUPY_PORT ?= 8888
TMUX_SESS ?= jupy
JUPY_LOG  ?= .jupyter_$(TMUX_SESS).log



.PHONY: env
env:
	@echo "[INFO] Creating conda env at $(ENV_DIR)"
	@source "$(CONDA_SH)" && \
	mkdir -p "$(SCRATCH)/conda_envs" && \
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

.PHONY: lock
lock:
	@echo "[INFO] Exporting locked environment"
	@source "$(CONDA_SH)" && \
	conda env export --prefix "$(ENV_DIR)" > environment.lock.yml

.PHONY: kernel
kernel:
	@echo "[INFO] Registering Jupyter kernel"
	@source "$(CONDA_SH)" && \
	conda activate "$(ENV_DIR)" && \
	python -m ipykernel install --user \
	  --name "$(REPO_NAME)" \
	  --display-name "CHPC: $(REPO_NAME)"

.PHONY: destroy
destroy:
	@echo "[INFO] Removing conda env at $(ENV_DIR)"
	@source "$(CONDA_SH)" && \
	conda deactivate || true && \
	conda env remove --prefix "$(ENV_DIR)" -y || true && \
	rm -rf "$(ENV_DIR)"

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
	  jupyter lab --no-browser --ip=0.0.0.0 --port=$(JUPY_PORT) \
	'