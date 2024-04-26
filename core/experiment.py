import numpy as np
import scipy
import os
import glob
import re
import multiprocessing
import functools
import time
import tqdm
import sys
sys.path.append('../utils')

import utils
import geometry
from .phantom import Phantom
from .transducer_set import TransducerSet
from .simulation import Simulation, SimProperties
from .sensor import Sensor


class Results:
    
    def __init__(self,
                 results_path = None,
                 ):
        self.result_paths = sorted(glob.glob(results_path+'/signal_*.np?'))
        self.other_signal_paths = sorted(glob.glob(results_path+'/key_signal*.np?'))
        self.length = len(self.result_paths)
        if self.length == 0:
            self.result_shape = None
        else:
            first = utils.load_array(self.result_paths[0])
            self.result_shape = first.shape
        
                
    def __len__(self):
        return self.length
    
    
    def __getitem__(self, index):
        search_index = self.indices().index(index)
        data = utils.load_array(self.result_paths[search_index])
        if len(self.other_signal_paths):
            other_data = utils.load_array(self.other_signal_paths[search_index])
            return data[0], data[1:], other_data
        return data[0], data[1:]
    
    
    def indices(self,):
        indices = []
        for path in self.result_paths:
            indices.append(int(re.findall(r'\d+', os.path.basename(path))[0]))
        return indices



class Experiment:
    
    def __init__(self, 
                 simulation_path = None,
                 sim_properties  = None,
                 phantom         = None,
                 transducer_set  = None,
                 sensor          = None,
                 nodes           = None,
                 results         = None,
                 indices         = None,
                 gpu             = True,
                 workers         = 2,
                 additional_keys = [],
                 ):
        if simulation_path is None:
            simulation_path = os.path.join(os.getcwd(), 'experiment')
            os.makedirs(simulation_path, exist_ok=True)
            self.simulation_path = simulation_path
        else:
            self.simulation_path = simulation_path
            
        self.sim_properties = sim_properties
        self.phantom = phantom
        self.transducer_set = transducer_set
        self.sensor = sensor
        self.nodes = nodes
        self.gpu = gpu
        if workers is not None and workers > 3:
            print('workers specifies the number of simultaneous simulations running on a single gpu node, setting workers higher than 3 will likely use more RAM without decreasing overall runtime')
        self.workers = workers

        os.makedirs(os.path.join(simulation_path, f'results'), exist_ok=True)
        self.add_results()
        self.indices = self.indices_to_run(indices)
        self.additional_keys = self.check_added_keys(additional_keys)
        if sim.sensor is not None and sim.sensor.aperture_type == "pressure_field" and "p_max" not in self.additional_keys:
            self.additional_keys.append("p_max")
        print(self.additional_keys)
        
        
    def __len__(self):
        if self.transducer_set is None:
            return 0
        return sum([transducer.get_num_rays() for transducer in self.transducer_set.transducers])


    # get the simulation indices of any simulations that do not have results
    def indices_to_run(self, indices=None, repeat=False):
        if indices is None:
            indices = list(range(len(self)))
        if np.isscalar(indices):
            indices = [indices]
        if (self.results is None or len(self.results) == 0) or repeat:
            return indices
        else:
            return sorted(list(set(indices) - set(self.results.indices())))        
    
    
    # save experiment
    def save(self, filepath=None):
        if filepath is None:
            filepath = self.simulation_path
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        self.sim_properties.save(os.path.join(filepath, f'sim_properties.json'))
        self.phantom.save(os.path.join(filepath, f'phantom'))
        self.transducer_set.save(os.path.join(filepath, f'transducer_set.json'))
        self.sensor.save(os.path.join(filepath, f'sensor.json'))
        dictionary = {
            'indices': self.indices,
            'nodes': self.nodes,
            'gpu': self.gpu,
            'workers': self.workers,
            'additional_keys': self.additional_keys,
        }
        utils.dict_to_json(dictionary, os.path.join(filepath, f'experiment.json'))
        
        
    # load experiment
    @classmethod
    def load(cls, filepath):
        experiment = cls(simulation_path=filepath)
        experiment.sim_properties = SimProperties.load(os.path.join(filepath, f'sim_properties.json'))
        experiment.phantom = Phantom.load(os.path.join(filepath, f'phantom'))
        experiment.transducer_set = TransducerSet.load(os.path.join(filepath, f'transducer_set.json'), c0=experiment.phantom.baseline[0])        
        experiment.sensor = Sensor.load(os.path.join(filepath, f'sensor.json'), experiment.transducer_set)        
        experiment.results = Results(os.path.join(filepath, f'results'))
        
        experiment_dict = utils.json_to_dict(os.path.join(filepath, f'experiment.json'))
        experiment.nodes = experiment_dict['nodes']
        experiment.gpu = experiment_dict['gpu']
        experiment.workers = experiment_dict['workers']
        experiment.additional_keys = experiment_dict['additional_keys']
        experiment.add_results()
        if experiment_dict['indices'] is None:
            experiment.indices = experiment.indices_to_run()
        else:
            experiment.indices = experiment_dict['indices']
            experiment.indices = experiment.indices_to_run(experiment.indices)
        if len(experiment.results) < len(experiment):
            print(f'Number of simulation results ({len(experiment.results)}) is less than the expected number of simulation results ({len(experiment)}), are you sure the simulation finished running?')
        elif len(experiment.results) > len(experiment):
            print(f'Number of simulation results ({len(experiment.results)}) is greater than the expected number of simulation results ({len(experiment)}), did the experiment parameters change since running?')
        return experiment
    
    
    # subdivide
    def subdivide(self, indices = None):
        if indices is None:
            indices = self.indices
        if len(indices) == 0:
            return None
        return np.array_split(np.array(indices), self.nodes)
    
    
    # run simulations by node
    def run(self, node=None, dry=False, repeat=False):
        assert os.path.exists(self.simulation_path), 'Attempting to run simulations but an experiment directory does not exist. Please save the experiment (my_experiment.save()) before running simulations.'
        
        if dry:
            indices = self.indices_to_run(repeat=True)
        else:
            indices = self.indices_to_run(repeat=repeat)
        
        if node is None:
            if self.nodes is None:
                self.nodes = 1
            if dry:
                self.run(0, dry=dry, repeat=repeat)
            else:
                for node in range(self.nodes):
                    self.run(node, dry=dry, repeat=repeat)
        else:
            if dry:
                # if self.workers is None:
                #     print('dry run of simulation')
                #     for index in tqdm.tqdm(indices):
                #         self.simulate(index, dry=dry)
                # else:
                print('dry run of simulation')
                index = 0
                for transducer in tqdm.tqdm(self.transducer_set.transducers):
                    self.simulate(index, dry=dry)
                    index += transducer.get_num_rays()
            else:
                if self.workers is None:
                    for index in indices:
                        self.simulate(index)
                else:
                    subdivisions = self.subdivide()
                    if subdivisions is None:
                        print('Found no more simulations to run.')
                    else:
                        print('running with {} workers\n'.format(self.workers))
                        queue = multiprocessing.Queue()
                        
                        if self.workers > 2:
                            simulations = np.array_split(subdivisions[node], self.workers - 1)
                            prep_procs = []
                            for i in range(self.workers - 1):
                                prep_procs.append(multiprocessing.Process(name=f'prep_{i}', target=self.prep_worker, args=(queue, simulations[i], dry)))
                                prep_procs[i].start()
                        else:
                            prep_procs = [multiprocessing.Process(name='prep', target=self.prep_worker, args=(queue, subdivisions[node], dry)),]
                            prep_procs[0].start()
                            
                        run_proc = multiprocessing.Process(name='run', target=self.run_worker, args=(queue, subdivisions[node],))
                        run_proc.start()
                        
                        for prep_proc in prep_procs:
                            prep_proc.join()
                        run_proc.join()
    
                
    def prep_worker(self, queue, indices, dry=False):
        count = 0
        while True:
            if queue.qsize() > 3:
                time.sleep(5)
                continue
            simulation = Simulation(self.sim_properties, 
                                    self.phantom, 
                                    self.transducer_set, 
                                    self.sensor, 
                                    simulation_path=self.simulation_path, 
                                    index=indices[count], 
                                    gpu=self.gpu, 
                                    dry=dry, 
                                    additional_keys=self.additional_keys)
            simulation.prep()
            queue.put(simulation)
            count += 1
            if count == len(indices):
                break
    
    
    def run_worker(self, queue, indices):
        count = 0
        while True:
            if queue.qsize() == 0:
                time.sleep(1)
                continue
            simulation = queue.get()
            simulation.run()
            count += 1
            if count == len(indices):
                break
        
                    
    def simulate(self, index, dry=False):
        simulation = Simulation(self.sim_properties, 
                                self.phantom, 
                                self.transducer_set, 
                                self.sensor, 
                                simulation_path=self.simulation_path, 
                                index=index, 
                                gpu=self.gpu, 
                                dry=dry, 
                                additional_keys=self.additional_keys)
        simulation.prep()
        simulation.run()
        
    
    def add_results(self,):
        self.results = Results(os.path.join(self.simulation_path,'results'))
        
        
    def plot_ray_path(self, index, ax=None, save=False, save_path=None, cmap='viridis'):
        simulation = Simulation(self.sim_properties, 
                                self.phantom, 
                                self.transducer_set, 
                                self.sensor, 
                                simulation_path=self.simulation_path, 
                                index=index, 
                                gpu=self.gpu)
        simulation.plot_medium_path(index, ax=ax, save=save, save_path=save_path, cmap=cmap)


    def check_added_keys(self, additional_keys):
        valid_keys = []
        allowed_keys = [
            "p",
            "p_max",
            "p_min",
            "p_rms",
            "p_max_all",
            "p_min_all",
            "p_final",
            "u",
            "u_max",
            "u_min",
            "u_rms",
            "u_max_all",
            "u_min_all",
            "u_final",
            "u_non_staggered",
            "I",
            "I_avg",
        ]
        for key in additional_keys:
            if key in allowed_keys:
                valid_keys.append(key)
            else:
                print(f'warning, requested flag "{key}" is not a valid flag, ignoring')
        return list(set(valid_keys))
