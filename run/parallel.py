import sys
import os
import argparse

parent = os.path.dirname(os.path.realpath('.'))
sys.path.append(parent)

from core import *
from utils import geometry

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='path to experiment directory')
    parser.add_argument('-n', '--node', type=int, help='node index to run')
    parser.add_argument('-s', '--nodes', type=int, help='total number of nodes')
    parser.add_argument('-g', '--gpu', type=bool, default=1, help='whether to use GPU')
    parser.add_argument('-r', '--repeat', type=bool, default=1, help='repeat and overwrite?')
    parser.add_argument('-w', '--workers', type=int, default=2, help='number of workers per node')
    
    args = parser.parse_args()
    
    test_experiment = experiment.Experiment.load(args.path)
    
    test_experiment.nodes = args.nodes
    test_experiment.gpu = args.gpu
    slurm_cpus = os.getenv('SLURM_CPUS_PER_TASK') # Check to see if we are in a slurm computing environment to avoid oversubscription
    if slurm_cpus is not None:
        print(f"Slurm environment detected. Found {slurm_cpus} cpus available")
        test_experiment.workers = int(slurm_cpus)
        test_experiment.repeat = -1
        print(f"Setting repeat to -1 to avoid asynchronous index allocation")
    else:
        test_experiment.workers = args.workers
        test_experiment.repeat = args.repeat
                
    test_experiment.run(args.node, repeat=test_experiment.repeat)
    
if __name__ == "__main__":
    sys.exit(main())
