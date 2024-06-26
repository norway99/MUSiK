import sys
import argparse

sys.path.append('./utils')
sys.path.append('./core')
sys.path.append('./')

sys.path.append('../utils')
sys.path.append('../core')
sys.path.append('../')

import utils
import geometry
from experiment import Experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='path to experiment directory')
    parser.add_argument('-n', '--node', type=int, help='node index to run')
    parser.add_argument('-s', '--nodes', type=int, help='total number of nodes')
    parser.add_argument('-g', '--gpu', type=bool, default=1, help='whether to use GPU')
    parser.add_argument('-r', '--repeat', type=bool, default=1, help='repeat and overwrite?')
    parser.add_argument('-w', '--workers', type=int, default=2, help='number of workers per node')
    
    args = parser.parse_args()
    
    experiment = Experiment.load(args.path)
    
    experiment.nodes = args.nodes
    experiment.gpu = args.gpu
    experiment.workers = args.workers
    if not args.repeat:
        experiment.indices = experiment.indices_to_run()
    experiment.run(args.node)
    
if __name__ == "__main__":
    sys.exit(main())
