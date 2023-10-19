1. Clone the respository.
2. Recommended: Create a new conda environment and activate it.
3. `pip install -e .`
4. Set up WandB if necessary.
5. Launch the desired sweep with `wandb sweep sweeps/sweep_name.yaml`.
This outputs a URL for viewing the sweep (`https://wandb.ai/...`), and a command for running sweep agents (`wandb agent ...`).
6. Run the `wandb agent ...` command from any machine on which this package is installed.
For multiprocessing, run this command multiple times from the same machine. Each agent spawns one process.