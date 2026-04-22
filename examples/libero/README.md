# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

## Running Inference using StreamingVLA models

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python examples/libero/streamingvla.py

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python examples/libero/streamingvla.py
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py policy:checkpoint --policy.config=streamingvla_pi05_libero --policy.dir=/path/to/your/checkpoint
```

## Reproducing PI0.5 Results
Terminal window 1:

```bash
source examples/libero/.venv/bin/activate
python examples/libero/pi05.py

```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=/path/to/pi05_libero/checkpoint
```