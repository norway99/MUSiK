"""Run the simulations for one or more cardiac-phase experiments.

    python 04_run.py --frames 0 1 2 [--node 0] [--workers 3]

Under a Slurm array, leave out `--frames` and each task runs the phase equal to
`SLURM_ARRAY_TASK_ID`. Within a phase, `--node` splits the work further using MUSiK's
own `nodes` partitioning. Finished indices are skipped, so jobs can be resubmitted.
"""

import argparse
import os

from common import Paths, load_config

from musik import experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--frames", type=int, nargs="*", default=None)
    parser.add_argument("--node", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    paths = Paths(config)
    if args.frames:
        frames = args.frames
    elif "SLURM_ARRAY_TASK_ID" in os.environ:
        frames = [int(os.environ["SLURM_ARRAY_TASK_ID"])]
    else:
        frames = range(config["cardiac"]["num_frames"])

    for frame in frames:
        path = paths.experiment(frame)
        if not path.exists():
            print(f"phase {frame:02d}: no experiment at {path}, skipping")
            continue
        loaded = experiment.Experiment.load(str(path))
        print(f"phase {frame:02d}: {len(loaded.indices_to_run())} of {len(loaded)} simulations remaining")
        loaded.run(node=args.node, workers=args.workers)


if __name__ == "__main__":
    main()
