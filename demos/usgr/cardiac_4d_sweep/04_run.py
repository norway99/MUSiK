"""Run the simulations for one or more cardiac-phase experiments.

    python 04_run.py --frames 0 1 2 [--node 0] [--prep-workers 1] [--sim-workers 3] [--gpus 0 1]

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
    parser.add_argument("--prep-workers", type=int, default=None, help="default: config")
    parser.add_argument("--sim-workers", type=int, default=None, help="simulations per GPU (default: config)")
    parser.add_argument("--gpus", type=int, nargs="*", default=None, help="device indices (default: config)")
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
        # Settings come from the config, not the saved experiment, so they can change between runs.
        sim = config["simulation"]
        loaded.run(
            node=args.node,
            prep_workers=args.prep_workers or sim["prep_workers"],
            sim_workers=args.sim_workers or sim["sim_workers"],
            gpus=args.gpus or sim["gpus"],
        )


if __name__ == "__main__":
    main()
