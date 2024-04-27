import numpy as np
import sys
sys.path.append('../utils')
import geometry
import matplotlib.pyplot as plt
import utils

sys.path.append('../k-wave-python')
import kwave
import kwave.ktransducer
from scipy.signal import hilbert


class Transducer:
    def __init__(self,
                 label                      = None,
                 max_frequency              = 2e6,
                 source_strength            = 1e6,
                 cycles                     = 2,
                 elements                   = 32,
                 active_elements            = None,
                 
                 width                      = 1e-2,         # transducer total width
                 height                     = 1e-2,         # transducer total width
                 
                #  element_width = 1e-4,                    # changed due to dynamic grid sizing
                #  elevation = 0.01,                        # changed due to dynamic grid sizing
                #  kerf = 0,                                # removed
                 radius                     = float('inf'),
                 focus_azimuth              = float('inf'), # PW by default
                 focus_elevation            = float('inf'),
                 sensor_sampling_scheme     = 'centroid',
                 sweep                      = np.pi/3,
                 ray_num                    = 64,
                 imaging_ndims              = 2, # should be 2 or 3
                 transmit_apodization       = 'Tukey',
                 receive_apodization        = 'Rectangular',
                 harmonic                   = 1,
                 bandwidth                  = 100,
                 compression_fac            = None,
                 normalize                  = True,
                 ):

        # probably should check that the geometry values are all positive!
        
        self.label = label
        self.max_frequency = max_frequency
        self.source_strength = 1e6
        self.cycles = cycles
        self.elements = elements
        self.active_elements = np.arange(elements)
        self.width = width
        self.height = height
        # self.element_width = element_width
        # self.elevation = elevation
        # self.kerf = kerf
        # self.azimuth = self.elements*(self.element_width+self.kerf) - self.kerf         # same as self.width
        self.radius = radius
        self.focus_azimuth = focus_azimuth
        self.focus_elevation = focus_elevation
        
        self.transmit_apodization = transmit_apodization
        self.receive_apodization = receive_apodization
        self.sensor_sampling_scheme = sensor_sampling_scheme
        self.sensors_per_el = None
        self.sensor_coords = None
        self.type = None

        if imaging_ndims != 2 and imaging_ndims != 3:
            raise Exception("Imaging must take place in either 2D or 3D")

        if not np.isscalar(sweep) and not np.isscalar(ray_num):
            if (len(sweep) != len(ray_num)) or (len(sweep) != (imaging_ndims - 1)):
                raise Exception("Dimensions do not match")
            self.sweep = np.asarray(sweep)
            self.ray_num = np.asarray(ray_num)
            ray_num_exception = len([ray_per_dim for ray_per_dim in self.ray_num if ray_per_dim <= 0]) > 0
        else:
            if imaging_ndims > 2:
                raise Exception("Dimensions do not match")
            self.sweep = sweep
            self.ray_num = ray_num
            ray_num_exception = (ray_num <= 0)
        if ray_num_exception:
            raise Exception("The array or list of rays per dimension should not contain entries <= 0")
        self.steering_angles = None
        
        self.pulse = None
        self.not_transducer = None
        
        # pulse preprocessing parameters
        self.harmonic = harmonic
        self.bandwidth = bandwidth
        self.compression_fac = compression_fac
        self.normalize = normalize

    @classmethod
    def load(cls, transducer_dict):
        transducer = cls()
        
        ray_transforms = []
        for ray_dict in transducer_dict['ray_transforms']:
            ray_transforms.append(geometry.Transform.load(ray_dict))
        transducer.ray_transforms = ray_transforms
        transducer_dict.pop('ray_transforms')
        transducer_dict['active_elements'] = np.array(transducer_dict['active_elements'])
        transducer_dict['sensor_coords'] = np.array(transducer_dict['sensor_coords'])
        transducer_dict['steering_angles'] = np.array(transducer_dict['steering_angles'])
        transducer_dict['pulse'] = np.array(transducer_dict['pulse'])
        for key, value in transducer_dict.items():
            setattr(transducer, key, value)
        return transducer


    def save(self) -> dict:
        save_dict = self.__dict__.copy()
        rays = []
        for ray in self.ray_transforms:
            rays.append(ray.save())
        save_dict.pop('ray_transforms')
        save_dict.pop('not_transducer')
        save_dict['ray_transforms'] = rays
        return save_dict
           
    
    def make_ray_transforms(self, imaging_ndims, sweep, ray_num):

        if imaging_ndims == 2:
            if ray_num == 1:
                coeff = -1
            else:
                coeff = 2
            yaws = np.linspace(-sweep/coeff, sweep/coeff, ray_num)
            pitches = np.zeros(ray_num)
            rays = [geometry.Transform(rotation = (yaw, 0, 0)) for yaw in yaws]
        else:
            coeff = np.where(ray_num == 1, -1, 2)
            yaws = np.linspace(-sweep[0]/coeff[0], sweep[0]/coeff[0], ray_num[0])
            pitches = np.linspace(-sweep[1]/coeff[1], sweep[1]/coeff[1], ray_num[1])                              
            rays = [geometry.Transform(rotation = (yaw, pitch, 0)) for yaw in yaws for pitch in pitches]
        return rays, yaws, pitches
    

    def get_num_rays(self) -> int:
        return len(self.ray_transforms)   
                
    
    def make_sensor_coords(self, c0):

        if self.sensor_sampling_scheme == "centroid":
            numpts = 1
            sensor_z_coords = 0
        else:
            wavelength = c0/self.max_frequency
            numpts = int(self.height/wavelength * 2)
            sensor_z_coords = np.linspace(-self.height/2, self.height/2, num = numpts)
        sensor_y_coords = np.transpose(np.linspace(-self.width/2 + (self.width / self.elements)/2, self.width/2 - (self.width / self.elements)/2, num = self.elements))
        
        sensor_coords = np.zeros((self.elements, numpts, 3))
        for col_num in range(numpts): 
            sensor_coords[:, col_num, 1] = sensor_y_coords
        for row_num in range(self.elements):
            sensor_coords[row_num, :, 2] = sensor_z_coords
        sensor_coords = np.reshape(sensor_coords, (self.elements*numpts, 3))
        self.sensors_per_el = numpts
        self.sensor_coords = sensor_coords


    def get_num_elements(self):
        return self.elements
        
        
    def get_sensor_coords(self):
        return self.sensor_coords
    
    
    def plot_sensor_coords(self, ax=None, transform=None, save=False, save_path=None, scale=0.003, color='b'):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_xlim(-scale, scale)
            ax.set_ylim(-scale, scale)
            ax.set_zlim(-scale, scale)
            ax.grid(False)
        if transform is not None:
            transformed_coords = transform.apply_to_points(self.sensor_coords)
            ax.scatter(transformed_coords[:,0], transformed_coords[:,1], transformed_coords[:,2], s=0.1, color=color)
        else:
            ax.scatter(self.sensor_coords[:,0], self.sensor_coords[:,1], self.sensor_coords[:,2], c=range(self.sensor_coords.shape[0]), s=0.5, cmap='viridis')
        if ax is None:
            if save:
                plt.savefig(save_path)
            else:
                plt.show()
                
    def plot_fov(self, ax=None, transform=None, save=False, save_path=None, scale=0.003, length=0.025, color='b'):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_xlim(-scale, scale)
            ax.set_ylim(-scale, scale)
            ax.set_zlim(-scale, scale)
            ax.grid(False)
        ray_num = self.ray_num if np.isscalar(self.ray_num) else self.ray_num[0]
        extents = [self.ray_transforms[0], 
                   self.ray_transforms[ray_num-1],
                   self.ray_transforms[-1],
                   self.ray_transforms[-ray_num],]
        points = np.array([t.apply_to_point(np.array([1,0,0])) for t in extents])
        if transform is not None:
            transformed_coords = transform.apply_to_points(points) * length
            centroid = transform.apply_to_points(np.zeros_like(transformed_coords))
                        
            ax.quiver(centroid[:,0], centroid[:,1], centroid[:,2], 
                      transformed_coords[:,0], transformed_coords[:,1], transformed_coords[:,2], 
                      length=1, linewidth=0.5, arrow_length_ratio=0, normalize=False, color=color)
            ax.quiver(centroid[:,0] + np.roll(transformed_coords[:,0],1), 
                      centroid[:,1] + np.roll(transformed_coords[:,1],1), 
                      centroid[:,2] + np.roll(transformed_coords[:,2],1), 
                      transformed_coords[:,0] - np.roll(transformed_coords[:,0],1), 
                      transformed_coords[:,1] - np.roll(transformed_coords[:,1],1), 
                      transformed_coords[:,2] - np.roll(transformed_coords[:,2],1), 
                      length=1, linewidth=0.5, arrow_length_ratio=0, normalize=False, color=color)
        else:
            points = points * length
            centroid = np.zeros_like(points)
            ax.quiver(centroid[:,0], centroid[:,1], centroid[:,2], 
                      points[:,0], points[:,1], points[:,2], 
                      length=1, linewidth=0.5, arrow_length_ratio=0, normalize=False)
            ax.quiver(np.roll(points[:,0],1), np.roll(points[:,1],1), np.roll(points[:,2],1), 
                      points[:,0]-np.roll(points[:,0],1), points[:,1]-np.roll(points[:,1],1), points[:,2]-np.roll(points[:,2],1), 
                      length=1, linewidth=0.5, arrow_length_ratio=0, normalize=False)
        if ax is None:
            if save:
                plt.savefig(save_path)
            else:
                plt.show()


    def get_sensors_per_el(self):
        return self.sensors_per_el


    def get_label(self):
        return self.label
    
    
    def get_freq(self):
        return self.max_frequency
    
    
    def make_pulse(self, dt, c0, rho0):
        signal = kwave.utils.signals.tone_burst(1/dt, self.max_frequency, self.cycles)
        signal = self.source_strength / (c0 * rho0) * signal
        self.pulse = signal


    def get_pulse(self):
        return self.pulse


    def __discretize(self, kgrid):
        # element_width = self.element_width / kgrid.dy
        # if element_width < 1:
        #     element_width = 1
        element_width = 1
        # kerf = self.kerf / kgrid.dy
        kerf = 0
        azimuth = self.width / kgrid.dy
        elevation = self.height / kgrid.dz    
        
        return element_width, kerf, azimuth, elevation
        
        
    def make_notatransducer(self, kgrid, c0, s_angle, pml) -> kwave.ktransducer.NotATransducer: #  gets called immediately before sim is run
        element_width, kerf, azimuth, elevation = self.__discretize(kgrid)
        num_elements = self.width // kgrid.dy
        
        position = [1, ((kgrid.Ny + pml[1] - azimuth - 1))/2, ((kgrid.Nz + pml[2] - elevation - 1))/2]
        # position in terms of transducer-centric coordinates is always 0 0 0, this is the corner of the transducer
                    
        # my_transducer = kwave.ktransducer.kWaveTransducerSimple(
        #         kgrid, self.elements, element_width, elevation, kerf,
        #         position, self.radius)
        
        my_transducer = kwave.ktransducer.kWaveTransducerSimple(
                kgrid, num_elements, element_width, elevation, kerf,
                position, self.radius)
        
        # print(f'self.elements, element_width, elevation, kerf: {num_elements, element_width, elevation, kerf}')
        
        not_transducer = kwave.ktransducer.NotATransducer(transducer = my_transducer,
                                            kgrid = kgrid,
                                            # active_elements = self.active_elements, 
                                            active_elements = None, 
                                            focus_distance = self.focus_azimuth,
                                            elevation_focus_distance = self.focus_elevation,
                                            receive_apodization = self.receive_apodization,
                                            transmit_apodization = self.transmit_apodization, 
                                            sound_speed = c0,
                                            input_signal = self.pulse, 
                                            steering_angle_max = np.max(self.steering_angles),
                                            steering_angle = s_angle)
        self.not_transducer = not_transducer
        return self.not_transducer


    def window(self, scan_lines) -> np.ndarray:
        l = self.get_pulse().shape[-1] * 4
        scan_lines[...,:l] = 0
        # scan_lines[...,-l:] = 0
        return scan_lines
        
        
    def gain_compensation(self, scan_lines, t_array, sim_properties) -> np.ndarray:
        l = self.get_pulse().shape[-1]
        t0 = l * (t_array[-1] / t_array.shape[0]) / 2
        r = 1540 * (t_array / 2 - t0)
        alpha_db_cm = sim_properties.alpha_coeff * self.get_freq() ** sim_properties.alpha_power
        alpha_np_m = alpha_db_cm / 8.686 / 100 / 1e6
        tgc = np.exp(alpha_np_m * 2 * r)
        rf = np.multiply(tgc, scan_lines)
        return rf


    def envelope_detection(self, scan_lines) -> np.ndarray: 
        env = np.abs(hilbert(scan_lines, axis=-1))
        return env
    
    
    def preprocess(self, scan_lines, t_array, sim_properties) -> np.ndarray:
        scan_lines = self.window(scan_lines)
        scan_lines = self.gain_compensation(scan_lines, t_array, sim_properties)
        scan_lines = kwave.utils.filters.gaussian_filter(scan_lines, 1 / (t_array[-1] / t_array.shape[0]), self.harmonic * self.get_freq(), self.bandwidth)
        scan_lines = self.envelope_detection(scan_lines)
        scan_lines = self.window(scan_lines)
        if self.compression_fac is not None:
            scan_lines = kwave.reconstruction.tools.log_compression(scan_lines, self.compression_fac, self.normalize)
        return scan_lines
    
    
    # # implement this in the subclass
    # def make_scan_line(self, sensor_data):
    #     print('in wrong fn')
    #     pass
    
        
        
class Focused(Transducer):

    def __init__(self,
                 label                      = None,
                 max_frequency              = 2e6,
                 source_strength            = 1e6,
                 cycles                     = 2,
                 elements                   = 32,
                 active_elements            = None,
                 
                 width                      = 1e-2,         # transducer total width
                 height                     = 1e-2,         # transducer total width
                 
                 radius                     = float('inf'),
                 focus_azimuth              = 20e-3,
                 focus_elevation            = float('inf'),
                 sensor_sampling_scheme     = 'centroid',
                 sweep                      = np.pi/3,
                 ray_num                    = 64,
                 imaging_ndims              = 2, # should be 2 or 3
                 transmit_apodization       = 'Hanning', # 
                 receive_apodization        = 'Rectangular',
                 harmonic                   = 1,
                 bandwidth                  = 100,
                 compression_fac            = None,
                 normalize                  = True,
                 ):
        if focus_azimuth == float('inf'):
            print('Focused transducers must have a finite focal length. Consider instantiating a plane-wave transducer if you require infinite focal length.')
        super().__init__(label, max_frequency, source_strength, cycles, elements, active_elements,
                         width, height, radius, focus_azimuth, focus_elevation, sensor_sampling_scheme,
                         sweep, ray_num, imaging_ndims, transmit_apodization, receive_apodization, harmonic, bandwidth, compression_fac, normalize)
        self.ray_transforms = self.make_ray_transforms(imaging_ndims, self.sweep, self.ray_num)[0]
        self.steering_angles = np.zeros(self.get_num_rays())
        self.type = 'focused'
        
    
    def make_scan_line(self, sensor_data):
        
        # get the receive apodization
        if len(self.active_elements) > 1:
            if self.receive_apodization == 'Rectangular': self.receive_apodization = 'Rectangular'
            apodization, _ = kwave.utils.signals.get_win(len(self.active_elements), type_=self.receive_apodization)
        else:
            apodization = 1
        apodization = np.array(apodization)

        # get the current beamforming weights and reverse
        delays = -self.not_transducer.beamforming_delays

        # offset the received sensor_data by the beamforming delays and apply receive apodization
        for element_index in range(len(self.not_transducer.active_elements)):
            if delays[element_index] > 0:
                # shift element data forwards
                sensor_data[element_index, :-1] = apodization[element_index] * np.concatenate((sensor_data[element_index, 1 + delays[element_index]:], np.zeros(delays[element_index])))
            elif delays[element_index] < 0:
                # shift element data backwards
                sensor_data[element_index, :-1] = apodization[element_index] * np.concatenate((np.zeros(-delays[element_index]), sensor_data[element_index, :-delays[element_index]]))

        # form the a-line summing across the elements
        line = np.sum(sensor_data, axis=0)

        return line
    

    @classmethod
    def load(cls, transducer_dict):
        transducer = cls()
        
        ray_transforms = []
        for ray_dict in transducer_dict['ray_transforms']:
            ray_transforms.append(geometry.Transform.load(ray_dict))
        transducer.ray_transforms = ray_transforms
        transducer_dict.pop('ray_transforms')
        transducer_dict['active_elements'] = np.array(transducer_dict['active_elements'])
        transducer_dict['sensor_coords'] = np.array(transducer_dict['sensor_coords'])
        transducer_dict['steering_angles'] = np.array(transducer_dict['steering_angles'])
        transducer_dict['pulse'] = np.array(transducer_dict['pulse'])
        for key, value in transducer_dict.items():
            setattr(transducer, key, value)
        return transducer


class Planewave(Transducer):

    def __init__(self,
                 label = None,
                 max_frequency = 2e6,
                 source_strength = 1e6,
                 cycles = 2,
                 elements = 32,
                 active_elements = None,

                 width                      = 1e-2,         # transducer total width
                 height                     = 1e-2,         # transducer total width
                 
                 radius = float('inf'),
                 focus_azimuth = float('inf'),
                 focus_elevation = 20e-3,
                 sensor_sampling_scheme = 'centroid',
                 sweep = np.pi/3,
                 ray_num = 64,
                 imaging_ndims = 2, # should be 2 or 3
                 steering_angles = None,
                 transmit_apodization = 'Tukey',
                 receive_apodization = 'Rectangular'
                 ):    
        super().__init__(label, max_frequency, source_strength, cycles, elements, active_elements,
                         width, height, radius, focus_azimuth, focus_elevation, sensor_sampling_scheme,
                         sweep, ray_num, imaging_ndims, transmit_apodization, receive_apodization)

        self.set_steering_angles(imaging_ndims, sweep, self.ray_num)
        if steering_angles is not None:
            if type(steering_angles) != list:
                raise Exception("Please supply steering angles as a list")
            else:
                self.steering_angles = np.asarray(steering_angles)
        self.type = 'planewave'


    def set_steering_angles(self, imaging_ndims, sweep, ray_num):

        ray_transforms, yaws = self.make_ray_transforms(imaging_ndims, sweep, ray_num)[:2] 
        if imaging_ndims == 2:
            self.steering_angles = yaws
            self.ray_transforms = [geometry.Transform(rotation = (0, 0, 0)) for yaw in yaws]
        else:
            self.ray_transforms = ray_transforms
            self.steering_angles = np.zeros((1, self.get_num_rays()))
            
    
    @classmethod
    def load(cls, transducer_dict):
        transducer = cls()
        
        ray_transforms = []
        for ray_dict in transducer_dict['ray_transforms']:
            ray_transforms.append(geometry.Transform.load(ray_dict))
        transducer.ray_transforms = ray_transforms
        transducer_dict.pop('ray_transforms')
        transducer_dict['active_elements'] = np.array(transducer_dict['active_elements'])
        transducer_dict['sensor_coords'] = np.array(transducer_dict['sensor_coords'])
        transducer_dict['steering_angles'] = np.array(transducer_dict['steering_angles'])
        transducer_dict['pulse'] = np.array(transducer_dict['pulse'])
        for key, value in transducer_dict.items():
            setattr(transducer, key, value)
        return transducer
           
                                       


    
