1. `git clone https://github.com/orrp/theory-of-umt.git`
2. Recommended: [Create a new conda environment and activate it](https://conda.io/docs/test-drive.html).
Tested with Python 3.11.
3. `cd theory-of-umt`
3. `pip install -e .`
4. `wandb login` (create an account first, if necessary).
5. Launch an experiment: `wandb sweep sweeps/sweep_name.yaml`.
This outputs a URL for viewing the sweep (`https://wandb.ai/...`), and a command for running sweep agents (`wandb agent ...`).
6. Run `wandb agent ...`.
For multiprocessing, run this command multiple times (one for each process).