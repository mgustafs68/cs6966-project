# cs6966-project
Project for CS 6966 in Spring 2026 Semester

# CHPC Setup
 0. Install VScode, [Remote Explorer](https://marketplace.visualstudio.com/items?itemName=ms-vscode.remote-explorer), [Remote-SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh), create a `config` text file in your `~/.ssh/` folder, ask AI what to put in there to connect to CHPC (depends on your OS). Then the option to connect will appear in the Remote Explorer extension in VSCode.
 1. In CHPC create an ssh key and add it to your github account so you can clone this repo (ask AI on how to do it). Clone this repo in your CHPC home. cd into the repo.
 2. `make env` creates the environment
 3. `make update` installs dependencies
 4. `make kernel` registers the Conda Python kernel to setup Jupyter
 5. Install [Vscode Jupyter Extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

# Use
To run a Jupyter server in CHPC compute node:
1. `make cpu` or `make gpu` to start a session on a compute node (CPU or GPU depending on your needs). You can tweak the duration and resources by changing the commands in the `Makefile`.
2. `make jup` will run a Jupyter server in the compute node. It will give you a URL you can paste in the VSCode extension to connect to it (use the non-IP-looking one). You will lose access to the terminal in the compute node since Jupyter will be running there.
3. `make juptmux` will create 2 tmux terminal instances, left one is Jupyter, right one is a terminal you can use to run additional commands inside the compute node. Advantage: you still have a terminal to run stuff inside the compute node. Disadvantage: tmux is a bit clunky to use.

# Useful commands
 1. `make activate` activates the venv
 2. To install dependencies you can add them to `environment.yml` and then run `make update`. You can run your own Conda commands but know the rest of the team won't be able to reproduce your environment if you don't update the `environment.yml` correctly.
 3. To generate a lock file with the environment: `make lock`, we will only probably need this at the end of the project when we hands things over.
 4. To use a CHPC CPU node run `make cpu`, to get a GPU node, `make gpu`.
 5. To run a Jupyter server in whatever node you are running, `make jup`.