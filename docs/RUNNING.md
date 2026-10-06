# Running the comparison

## Installation and checks

```sh
uv sync --frozen --python 3.12
uv run ruff check . && uv run ruff format --check . && uv run pytest -q
```

The lockfile installs CPU Torch; no GPU is used. Box2D's isolated build gets SWIG from its pinned PyPI wheel through uv's extra build dependency. A C++ compiler is still required. The measured machine and compiler are in [results/machine.json](../results/machine.json). The pytest suite includes a small Box2D comparison; it does not train policies. CI runs the same three check commands separately.

## Development

Each entry in [configs/development.json](../configs/development.json) states task, method, seed range, variant and time limit. Chunks are bounded so they can share the machine:

```sh
/home/alp/Projects/profile-program/bin/pp-run heavy .venv/bin/python sweep.py --plan configs/development.json --chunk 0
```

Development timing is affected by other heavy work. It is kept as experiment history, not mixed into the main result. `docs/LOG.md` reports effects and unsuccessful attempts.

## Final measurements

Each final invocation uses `pp-run bench`, a single-thread environment, and nice 19. `configs/final.json` is the exact seed plan. Run each chunk separately; its five-seed Acrobot batches are bounded by the task's fixed 300-second per-seed budget. The shared wrapper enforces benchmark exclusivity and a gap between windows. Dependency installation takes place before measurement.

```sh
/home/alp/Projects/profile-program/bin/pp-run bench env PP_MODE=bench nice -n 19 .venv/bin/python sweep.py --plan configs/final.json --chunk 0 --output results/single-chunk
```

`measure.py` drives all chunks sequentially, opening a separate wrapper window for each. `--not-before` accepts a timezone-qualified ISO timestamp for shared-machine scheduling; waiting is outside the seed clocks. Use `--output` for a new replication directory: committed raw results are never overwritten. Batch command logs live in `results/batches/`, with a separate prefix for a replication.

```sh
.venv/bin/python measure.py --plan configs/final.json --output results/replication
```

The deliberately explicit `PP_MODE=bench` marker is not itself an exclusivity mechanism: `pp-run bench` is required. Every worker starts in a new process. The launcher refuses to overwrite an existing seed result; select a new output directory for a replication rather than deleting original data. For one method/task independently:

```sh
/home/alp/Projects/profile-program/bin/pp-run bench env PP_MODE=bench nice -n 19 .venv/bin/python run.py --phase final --task CartPole-v1 --method cem --seeds 0:5 --output results/one-method
```

Every raw JSON contains the worker command, launcher arguments, software versions, thread settings, full evaluation-return arrays, elapsed clock values, stop reason and unsuccessful seeds. The absolute monotonic start timestamp in a command is a trace value, not a reusable CLI argument; `run.py` generates it anew.

The wrapper path is specific to the measured shared laptop and can be supplied with `measure.py --runner /path/to/pp-run`. On another host use the same exclusive-run wrapper or an equivalent dedicated-machine setup and record that change. Fresh processes do not imply cold disk caches: package files can remain in the OS page cache across seeds. CPU frequency and ordinary OS background activity are not pinned.

## Tables

```sh
uv run python summarize.py --input results/final --output results/summary.json
```

The script prints Markdown and saves machine-readable statistics including the files that contributed to every row. Its median and interquartile range are conditional on success. The Kaplan–Meier output uses actual stop times, including training-step horizon stops, as censoring; this is descriptive because early horizon exhaustion need not be independent of eventual solve time. A method that never passes has no median solve time, not a zero or its timeout value.
