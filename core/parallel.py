import sys
import argparse

sys.path.append('../utils')
sys.path.append('../core')
sys.path.append('../')
from core import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='path to experiment directory')
    parser.add_argument('-n', '--node', type=int, help='node index to run')
    parser.add_argument('-s', '--nodes', type=int, help='total number of nodes')
    parser.add_argument('-g', '--gpu', type=bool, default=1, help='whether to use GPU')
    parser.add_argument('-r', '--results', type=bool, default=0, help='look for results?')
    parser.add_argument('-w', '--workers', type=int, default=2, help='number of workers per node')
    
    args = parser.parse_args()
    
    test_experiment = experiment.Experiment.load(args.path)
    
    test_experiment.nodes = args.nodes
    test_experiment.gpu = args.gpu
    test_experiment.workers = args.workers
    if args.results:
        test_experiment.indices = test_experiment.indices_to_run()
    test_experiment.run(args.node)
    
if __name__ == "__main__":
    sys.exit(main())