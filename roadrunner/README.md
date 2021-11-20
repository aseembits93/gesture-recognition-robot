# Roadrunner

An amalgamation of Apex and CassieNet. New Cassie RL repo ~~that just works~~

This is a current "clean" *working* branch of roadrunner. This serves as a sort of safe checkpoint where we know things are working that people can start from or branch off. Please keep this branch clean, so if you'd like to make changes/use it for research please make a new personal branch for now. After changes and fixes get added to this branch we can come back and all rebranch off from the same place. 

This branch is essentially the [Feat/heeldown](https://github.com/osudrl/roadrunner/commit/c3fddabad2351e1a8040c7eca7d0d2fba1214dca) commit. It does not have working rpo.py or any of the planning stuff. Note that while these files are included they remain UNTESTED and for all intents an purposes should be considered broken for now. 

## Usage instructions 

Complete documentation, explanataion, and usage guide to come.

The main running file is `main.py`. This is the main file used for starting training and evaluating policies from the command line. Look at the "ppo" and "eval" modes for command line options. For ease of use, an example ppo training file `run_ppo.py` is included, which should train a successful policy with the already hard coded parameters (just run `python run_ppo.py`). 

A working conda environment is also provided and can be created from the included `spec-file.txt` (check the conda [guide](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#building-identical-conda-environments) for specific instructions). [Ray](https://docs.ray.io/en/latest/installation.html) needs to installed using pip, so just run `pip install ray`

Do note that this conda env currently has torch version 1.3.1 which is quite out of date. Things should still work with newer versions of torch, but this is technically untested. 

## Evaluation of the Pretrained Policies
run `python3 main.py eval --path ./pretrained_models/xxx` to visualize the pretrained policies. We have 2 base policies under `./pretrained_models/`, `speed` is the basic ICRA reward with objectives on tracking body velocities. `stairs` is trained with simulated stairs and the vis of stairs will be automatically turned on. 

Evaluating policies with `--interactive` will give you keyboard control, and you can modify any keys inside each training envs under `./envs/templates/`.

## Docs

Install sphinx and run `make html` at top level. Outputs are in docs/build/html

To rerun autodoc run `sphinx-apidoc -f -o docs/source/ .` in the root dir
