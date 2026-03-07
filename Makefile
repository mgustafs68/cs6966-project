# ---- config ----
REPO_NAME := $(notdir $(CURDIR))
ENV_DIR   := /scratch/general/vast/$(USER)/conda_envs/$(REPO_NAME)
MINIFORGE := miniforge3/25.11.0

SHELL := /bin/bash

JUPY_PORT ?= 8888
TMUX_SESS ?= jupy
JUPY_LOG  ?= .jupyter_$(TMUX_SESS).log


.PHONY: env
env:
	@echo "[INFO] Creating conda env at $(ENV_DIR)"
	module load $(MINIFORGE) && \
	mkdir -p $(HOME)/conda_envs && \
	conda env create --prefix $(ENV_DIR) -f environment.yml

.PHONY: activate
activate:
	@echo "[INFO] Activating conda env at $(ENV_DIR)"
	module load $(MINIFORGE) && \
	source $$(conda info --base)/etc/profile.d/conda.sh && \
	conda activate $(ENV_DIR) && \
	exec $$SHELL

.PHONY: update
update:
	@echo "[INFO] Updating conda env at $(ENV_DIR)"
	module load $(MINIFORGE) && \
	source $$(conda info --base)/etc/profile.d/conda.sh && \
	conda env update --prefix $(ENV_DIR) -f environment.yml --prune

.PHONY: lock
lock:
	@echo "[INFO] Exporting locked environment"
	module load $(MINIFORGE) && \
	source $$(conda info --base)/etc/profile.d/conda.sh && \
	conda env export --prefix $(ENV_DIR) > environment.lock.yml

.PHONY: kernel
kernel:
	@echo "[INFO] Registering Jupyter kernel"
	module load $(MINIFORGE) && \
	source $$(conda info --base)/etc/profile.d/conda.sh && \
	conda activate $(ENV_DIR) && \
	python -m ipykernel install --user \
	  --name $(REPO_NAME) \
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
	  module load $(MINIFORGE) && \
	  source $$(conda info --base)/etc/profile.d/conda.sh && \
	  conda activate $(ENV_DIR) && \
	  jupyter lab --no-browser --ip=0.0.0.0 --port=$(JUPY_PORT) \
	'

.PHONY: jup
juptmux:
	srun --pty bash -lc '\
	  module load $(MINIFORGE) && \
	  source $$(conda info --base)/etc/profile.d/conda.sh && \
	  conda activate $(ENV_DIR) && \
	  tmux kill-session -t jupy 2>/dev/null || true && \
	  tmux new-session -d -s jupy && \
	  tmux send-keys -t jupy "jupyter lab --no-browser --ip=0.0.0.0 --port=$(JUPY_PORT)" C-m && \
	  tmux split-window -h -t jupy && \
	  tmux send-keys -t jupy "module load $(MINIFORGE)" C-m && \
	  tmux send-keys -t jupy "source $$(conda info --base)/etc/profile.d/conda.sh" C-m && \
	  tmux send-keys -t jupy "conda activate $(ENV_DIR)" C-m && \
	  tmux select-pane -t 1 && \
	  tmux attach -t jupy \
	'

