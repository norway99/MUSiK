import numpy as np
from utils import geometry
import matplotlib.pyplot as plt

import kwave
import kwave.ktransducer
from scipy.signal import hilbert


class Transducer:
    """
    A class to represent a transducer object. This class is used to create a transducer object that can be used in a simulation.

    Attributes:
        label (str): The label of the transducer.
        max_frequency (float): The maximum frequency of the transducer in Hz.
        source_strength (float): The source strength of the transducer.
        cycles (int): The number of cycles of the transducer.
        elements (int): The number of elements in the transducer.
        active_elements (ndarray): An array of active element indices.
        width (float): The total width of the transducer.
        height (float): The total height of the transducer.
        radius (float): The radius of the transducer.
        focus_azimuth (float): The azimuth focus of the transducer.
        focus_elevation (float): The elevation focus of the transducer.
        sensor_sampling_scheme (str): The sensor sampling scheme.
        sweep (float or ndarray): The sweep angle of the transducer.
        ray_num (int or ndarray): The number of rays per dimension.
        imaging_ndims (int): The number of dimensions for imaging.
        transmit_apodization (str): The transmit apodization method.
        receive_apodization (str): The receive apodization method.
        harmonic (int): The harmonic of the transducer.
        bandwidth (float): The bandwidth of the transducer.
        compression_fac (float): The compression factor of the transducer.
        normalize (bool): Whether to normalize the transducer.

    Methods:
        load(transducer_dict): Loads a transducer object from a dictionary.
        save(): Saves the transducer object as a dictionary.
        make_ray_transforms(imaging_ndims, sweep, ray_num): Creates ray transforms based on the imaging dimensions, sweep angle, and number of rays per dimension.
        get_num_rays(): Returns the number of rays in the transducer.
        make_sensor_coords(c0): Creates sensor coordinates based on the speed of sound.
        get_num_elements(): Returns the number of elements in the transducer.
        get_sensor_coords(): Returns the sensor coordinates of the transducer.
        plot_sensor_coords(ax=None, transform=None, save=False, save_path=None, scale=0.003, color='b'): Plots the sensor coordinates of the transducer.
        plot_fov(ax=None, transform=None, save=False, save_path=None, scale=0.003, length=0.025, color='b'): Plots the field of view of the transducer.
    """

    def __init__(
        self,
        label=None,
        max_frequency=2e6,
        source_strength=1e6,
        cycles=2,
        elements=32,
        active_elements=None,
        width=1e-2,  # transducer total width
        height=1e-2,  # transducer total width
        radius=float("inf"),
        focus_azimuth=float("inf"),  # PW by default
        focus_elevation=float("inf"),
        sensor_sampling_scheme="centroid",
        sweep=np.pi / 3,
        ray_num=64,
        imaging_ndims=2,  # should be 2 or 3
        transmit_apodization="Tukey",
        receive_apodization="Rectangular",
        harmonic=1,
        bandwidth=100,
        compression_fac=None,
        normalize=True,
        balance_3D=False,  # balance_3D attempts to maintain symmetry in azimuthal and elevational axes
        transmit=True,  # if false, use as sensor only
    ):
        """
        Args:
            label (str, optional): The label of the transducer. Defaults to None.
            max_frequency (float, optional): The maximum frequency of the transducer in Hz. Defaults to 2e6.
            source_strength (float, optional): The source strength of the transducer. Defaults to 1e6.
            cycles (int, optional): The number of cycles of the transducer. Defaults to 2.
            elements (int, optional): The number of elements in the transducer. Defaults to 32.
            active_elements (ndarray, optional): An array of active element indices. Defaults to None.
            width (float, optional): The total width of the transducer. Defaults to 1e-2.
            height (float, optional): The total height of the transducer. Defaults to 1e-2.
            radius (float, optional): The radius of the transducer. Defaults to float('inf').
            focus_azimuth (float, optional): The azimuth focus of the transducer. Defaults to float('inf').
            focus_elevation (float, optional): The elevation focus of the transducer. Defaults to float('inf').
            sensor_sampling_scheme (str, optional): The sensor sampling scheme. Defaults to 'centroid'.
            sweep (float or ndarray, optional): The sweep angle of the transducer. Defaults to np.pi/3.
            ray_num (int or ndarray, optional): The number of rays per dimension. Defaults to 64.
            imaging_ndims (int, optional): The number of dimensions for imaging. Defaults to 2.
            transmit_apodization (str, optional): The transmit apodization method. Defaults to 'Tukey'.
            receive_apodization (str, optional): The receive apodization method. Defaults to 'Rectangular'.
            harmonic (int, optional): The harmonic of the transducer. Defaults to 1.
            bandwidth (float, optional): The bandwidth of the transducer. Defaults to 100.
            compression_fac (float, optional): The compression factor of the transducer. Defaults to None.
            normalize (bool, optional): Whether to normalize the transducer. Defaults to True.
        """

        self.label = label
        self.max_frequency = max_frequency
        self.source_strength = 1e6
        self.cycles = cycles
        self.elements = elements
        self.active_elements = np.arange(elements)
        self.width = width
        self.height = height
        self.radius = radius
        self.focus_azimuth = focus_azimuth
        self.focus_elevation = focus_elevation

        self.transmit_apodization = transmit_apodization
        self.receive_apodization = receive_apodization
        self.sensor_sampling_scheme = sensor_sampling_scheme
        self.sensors_per_el = None
        self.sensor_coords = None
        self.type = None
        self.balance_3D = balance_3D
        self.transmit = transmit

        if imaging_ndims != 2 and imaging_ndims != 3:
            raise Exception("Imaging must take place in either 2D or 3D")

        if not np.isscalar(sweep) and not np.isscalar(ray_num):
            if (len(sweep) != len(ray_num)) or (len(sweep) != (imaging_ndims - 1)):
                raise Exception("Dimensions do not match")
            self.sweep = np.asarray(sweep)
            self.ray_num = np.asarray(ray_num)
            ray_num_exception = (
                len([ray_per_dim for ray_per_dim in self.ray_num if ray_per_dim <= 0])
                > 0
            )
        else:
            if imaging_ndims > 2:
                raise Exception("Dimensions do not match")
            self.sweep = sweep
            self.ray_num = ray_num
            ray_num_exception = ray_num <= 0
        if ray_num_exception:
            raise Exception(
                "The array or list of rays per dimension should not contain entries <= 0"
            )
        self.steering_angles = None

        self.pulse = None
        self.not_transducer = None

        self.harmonic = harmonic
        self.bandwidth = bandwidth
        self.compression_fac = compression_fac
        self.normalize = normalize

    @classmethod
    def load(cls, transducer_dict):
        transducer = cls()

        ray_transforms = []
        for ray_dict in transducer_dict["ray_transforms"]:
            ray_transforms.append(geometry.Transform.load(ray_dict))
        transducer.ray_transforms = ray_transforms
        transducer_dict.pop("ray_transforms")
        transducer_dict["active_elements"] = np.array(
            transducer_dict["active_elements"]
        )
        transducer_dict["sensor_coords"] = np.array(transducer_dict["sensor_coords"])
        transducer_dict["steering_angles"] = np.array(
            transducer_dict["steering_angles"]
        )
        transducer_dict["pulse"] = np.array(transducer_dict["pulse"])
        for key, value in transducer_dict.items():
            setattr(transducer, key, value)
        return transducer

    def save(self) -> dict:
        save_dict = self.__dict__.copy()
        rays = []
        for ray in self.ray_transforms:
            rays.append(ray.save())
        save_dict.pop("ray_transforms")
        save_dict.pop("not_transducer")
        save_dict["ray_transforms"] = rays
        return save_dict

    def make_ray_transforms(self, imaging_ndims, sweep, ray_num):
        """
        Creates ray transforms based on the imaging dimensions, sweep angle, and number of rays per dimension.

        Args:
            imaging_ndims (int): The number of dimensions for imaging.
            sweep (float or ndarray): The sweep angle of the transducer.
            ray_num (int or ndarray): The number of rays per dimension.

        Returns:
            list: A list of ray transforms.
            ndarray: The yaw angles.
            ndarray: The pitch angles.
        """
        if imaging_ndims == 2:
            if ray_num == 1:
                coeff = -1
            else:
                coeff = 2
            yaws = np.linspace(-sweep / coeff, sweep / coeff, ray_num)
            pitches = np.zeros(ray_num)
            rays = [geometry.Transform(rotation=(yaw, 0, 0)) for yaw in yaws]
        else:
            if self.balance_3D:
                coeff = np.where(ray_num == 1, -1, 2)
                # xs = np.linspace(-sweep[0]/coeff[0], sweep[0]/coeff[0], ray_num[0])
                # ys = np.linspace(-sweep[1]/coeff[1], sweep[1]/coeff[1], ray_num[1])
                xs = np.arctan(
                    np.linspace(-sweep[0] / coeff[0], sweep[0] / coeff[0], ray_num[0])
                )
                ys = np.arctan(
                    np.linspace(-sweep[1] / coeff[1], sweep[0] / coeff[1], ray_num[1])
                )

                rays = [
                    geometry.Transform(
                        rotation=(np.sqrt(x**2 + y**2), 0, np.arctan2(y, x)),
                        intrinsic=False,
                    )
                    for y in ys
                    for x in xs
                ]
                yaws = np.sqrt(xs**2 + ys**2)
                pitches = np.sqrt(xs**2 + ys**2)
            else:
                coeff = np.where(ray_num == 1, -1, 2)
                yaws = np.linspace(
                    -sweep[0] / coeff[0], sweep[0] / coeff[0], ray_num[0]
                )
                pitches = np.linspace(
                    -sweep[1] / coeff[1], sweep[1] / coeff[1], ray_num[1]
                )
                rays = [
                    geometry.Transform(rotation=(yaw, pitch, 0))
                    for yaw in yaws
                    for pitch in pitches
                ]
        return rays, yaws, pitches

    def get_num_rays(self) -> int:
        """
        Returns the number of rays in the transducer.

        Returns:
            int: The number of rays in the transducer.
        """
        return len(self.ray_transforms)

    # Needs editing for number of elements
    def make_sensor_coords(self, c0):
        """
        Creates sensor coordinates based on the speed of sound.

        Args:
            c0 (float): The speed of sound.
        """
        if self.sensor_sampling_scheme == "centroid":
            numpts = 1
            sensor_z_coords = 0
        else:
            wavelength = c0 / self.max_frequency
            numpts = max(int(self.height / wavelength * 2), 1)
            if numpts == 1:
                sensor_z_coords = np.array([0])
            else:
                sensor_z_coords = np.linspace(
                    -self.height / 2, self.height / 2, num=numpts, endpoint=True
                )
        sensor_y_coords = np.transpose(
            np.linspace(
                -self.width / 2 + (self.width / self.elements) / 2,
                self.width / 2 - (self.width / self.elements) / 2,
                num=self.elements,
            )
        )

        sensor_coords = np.zeros((self.elements, numpts, 3))
        for col_num in range(numpts):
            sensor_coords[:, col_num, 1] = sensor_y_coords
        for row_num in range(self.elements):
            sensor_coords[row_num, :, 2] = sensor_z_coords
        sensor_coords = np.reshape(sensor_coords, (self.elements * numpts, 3))
        self.sensors_per_el = numpts
        self.sensor_coords = sensor_coords

    def get_num_elements(self):
        """
        Returns the number of elements in the transducer.

        Returns:
            int: The number of elements in the transducer.
        """
        return self.elements

    def get_sensor_coords(self):
        """
        Returns the sensor coordinates of the transducer.

        Returns:
            ndarray: The sensor coordinates of the transducer.
        """
        return self.sensor_coords

    def plot_sensor_coords(
        self,
        ax=None,
        transform=None,
        save=False,
        save_path=None,
        scale=0.003,
        color="b",
    ):
        """
        Plots the sensor coordinates of the transducer.

        Args:
            ax (Axes3D, optional): The 3D axes to plot on. If None, a new figure and axes will be created. Defaults to None.
            transform (Transform, optional): The transform to apply to the sensor coordinates. Defaults to None.
            save (bool, optional): Whether to save the plot. Defaults to False.
            save_path (str, optional): The path to save the plot. Required if save is True. Defaults to None.
            scale (float, optional): The scale of the plot. Defaults to 0.003.
            color (str, optional): The color of the plot. Defaults to 'b'.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            ax.set_xlim(-scale, scale)
            ax.set_ylim(-scale, scale)
            ax.set_zlim(-scale, scale)
            ax.grid(False)
        if transform is not None:
            transformed_coords = transform.apply_to_points(self.sensor_coords)
            ax.scatter(
                transformed_coords[:, 0],
                transformed_coords[:, 1],
                transformed_coords[:, 2],
                s=0.1,
                color=color,
            )
        else:
            ax.scatter(
                self.sensor_coords[:, 0],
                self.sensor_coords[:, 1],
                self.sensor_coords[:, 2],
                c=range(self.sensor_coords.shape[0]),
                s=0.5,
                cmap="viridis",
            )
        if ax is None:
            if save:
                plt.savefig(save_path)
            else:
                plt.show()

    def plot_fov(
        self,
        ax=None,
        transform=None,
        save=False,
        save_path=None,
        scale=0.003,
        length=0.025,
        color="b",
    ):
        """
        Plots the field of view of the transducer.

        Args:
            ax (Axes3D, optional): The 3D axes to plot on. If None, a new figure and axes will be created. Defaults to None.
            transform (Transform, optional): The transform to apply to the field of view. Defaults to None.
            save (bool, optional): Whether to save the plot. Defaults to False.
            save_path (str, optional): The path to save the plot. Required if save is True. Defaults to None.
            scale (float, optional): The scale of the plot. Defaults to 0.003.
            length (float, optional): The length of the field of view. Defaults to 0.025.
            color (str, optional): The color of the plot. Defaults to 'b'.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            ax.set_xlim(-scale, scale)
            ax.set_ylim(-scale, scale)
            ax.set_zlim(-scale, scale)
            ax.grid(False)
        ray_num = self.ray_num if np.isscalar(self.ray_num) else self.ray_num[0]
        extents = [
            self.ray_transforms[0],
            self.ray_transforms[ray_num - 1],
            self.ray_transforms[-1],
            self.ray_transforms[-ray_num],
        ]
        points = np.array([t.apply_to_point(np.array([1, 0, 0])) for t in extents])
        if transform is not None:
            transformed_coords = transform.apply_to_points(points) * length
            centroid = transform.apply_to_points(np.zeros_like(transformed_coords))

            ax.scatter(centroid[:, 0], centroid[:, 1], centroid[:, 2], color=color)
            if self.transmit:
                ax.quiver(
                    centroid[:, 0],
                    centroid[:, 1],
                    centroid[:, 2],
                    transformed_coords[:, 0],
                    transformed_coords[:, 1],
                    transformed_coords[:, 2],
                    length=1,
                    linewidth=0.5,
                    arrow_length_ratio=0,
                    normalize=False,
                    color=color,
                )
                ax.quiver(
                    centroid[:, 0] + np.roll(transformed_coords[:, 0], 1),
                    centroid[:, 1] + np.roll(transformed_coords[:, 1], 1),
                    centroid[:, 2] + np.roll(transformed_coords[:, 2], 1),
                    transformed_coords[:, 0] - np.roll(transformed_coords[:, 0], 1),
                    transformed_coords[:, 1] - np.roll(transformed_coords[:, 1], 1),
                    transformed_coords[:, 2] - np.roll(transformed_coords[:, 2], 1),
                    length=1,
                    linewidth=0.5,
                    arrow_length_ratio=0,
                    normalize=False,
                    color=color,
                )
        else:
            points = points * length
            centroid = np.zeros_like(points)
            ax.quiver(
                centroid[:, 0],
                centroid[:, 1],
                centroid[:, 2],
                points[:, 0],
                points[:, 1],
                points[:, 2],
                length=1,
                linewidth=0.5,
                arrow_length_ratio=0,
                normalize=False,
            )
            ax.quiver(
                np.roll(points[:, 0], 1),
                np.roll(points[:, 1], 1),
                np.roll(points[:, 2], 1),
                points[:, 0] - np.roll(points[:, 0], 1),
                points[:, 1] - np.roll(points[:, 1], 1),
                points[:, 2] - np.roll(points[:, 2], 1),
                length=1,
                linewidth=0.5,
                arrow_length_ratio=0,
                normalize=False,
            )
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
        signal = kwave.utils.signals.tone_burst(1 / dt, self.max_frequency, self.cycles)
        signal = self.source_strength / (c0 * rho0) * signal
        self.pulse = signal

    def get_pulse(self):
        return self.pulse

    def __discretize(self, kgrid):
        num_elements = self.width // kgrid.dy
        element_width = 1
        kerf = 0
        azimuth = self.width / kgrid.dy
        elevation = self.height / kgrid.dz
        return num_elements, element_width, kerf, azimuth, elevation

    def make_notatransducer(
        self, kgrid, c0, s_angle, pml
    ) -> kwave.ktransducer.NotATransducer:  #  gets called immediately before sim is run
        num_elements, element_width, kerf, azimuth, elevation = self.__discretize(kgrid)

        position = [
            1,
            (kgrid.Ny + pml[1] - azimuth - 1) / 2,
            (kgrid.Nz + pml[2] - elevation - 1) / 2,
        ]

        my_transducer = kwave.ktransducer.kWaveTransducerSimple(
            kgrid, num_elements, element_width, elevation, kerf, position, self.radius
        )

        not_transducer = kwave.ktransducer.NotATransducer(
            transducer=my_transducer,
            kgrid=kgrid,
            active_elements=None,
            focus_distance=self.focus_azimuth,
            elevation_focus_distance=self.focus_elevation,
            receive_apodization=self.receive_apodization,
            transmit_apodization=self.transmit_apodization,
            sound_speed=c0,
            input_signal=self.pulse,
            steering_angle_max=180 / np.pi * np.max(self.steering_angles),
            steering_angle=s_angle * 180 / np.pi,
        )
        self.not_transducer = not_transducer
        return self.not_transducer

    def window(self, scan_lines, window_factor=4) -> np.ndarray:
        l = self.get_pulse().shape[-1] * window_factor
        scan_lines[..., :l] = 0
        return scan_lines

    def gain_compensation(
        self, scan_lines, t_array, sim_properties, attenuation_factor=1
    ) -> np.ndarray:
        l = self.get_pulse().shape[-1]
        t0 = l * (t_array[-1] / t_array.shape[0]) / 2
        r = 1540 * (t_array / 2 - t0)
        alpha_db_cm = (
            sim_properties.alpha_coeff * self.get_freq() ** sim_properties.alpha_power
        )
        alpha_np_m = alpha_db_cm / 8.686 / 100 / 1e6
        tgc = np.exp(alpha_np_m * 2 * r * attenuation_factor)
        rf = np.multiply(tgc, scan_lines)
        return rf

    def envelope_detection(self, scan_lines) -> np.ndarray:
        env = np.abs(hilbert(scan_lines, axis=-1))
        return env

    def preprocess(
        self, scan_lines, t_array, sim_properties, window_factor=4, attenuation_factor=1
    ) -> np.ndarray:
        scan_lines = self.window(scan_lines, window_factor)
        return scan_lines


class Focused(Transducer):
    def __init__(
        self,
        label=None,
        max_frequency=2e6,
        source_strength=1e6,
        cycles=2,
        elements=32,
        active_elements=None,
        width=1e-2,  # transducer total width
        height=1e-2,  # transducer total width
        radius=float("inf"),
        focus_azimuth=float("inf"),
        focus_elevation=float("inf"),
        sensor_sampling_scheme="centroid",
        sweep=np.pi / 3,
        ray_num=64,
        imaging_ndims=2,  # should be 2 or 3
        transmit_apodization="Hanning",  #
        receive_apodization="Rectangular",
        harmonic=1,
        bandwidth=100,
        compression_fac=None,
        normalize=True,
        balance_3D=False,
        transmit=True,
    ):
        # if focus_azimuth == float('inf'):
        #     print('Focused transducers must have a finite focal length. Consider instantiating a plane-wave transducer if you require infinite focal length.')
        super().__init__(
            label,
            max_frequency,
            source_strength,
            cycles,
            elements,
            active_elements,
            width,
            height,
            radius,
            focus_azimuth,
            focus_elevation,
            sensor_sampling_scheme,
            sweep,
            ray_num,
            imaging_ndims,
            transmit_apodization,
            receive_apodization,
            harmonic,
            bandwidth,
            compression_fac,
            normalize,
            balance_3D,
            transmit,
        )
        self.ray_transforms = self.make_ray_transforms(
            imaging_ndims, self.sweep, self.ray_num
        )[0]
        self.steering_angles = np.zeros(self.get_num_rays())
        self.type = "focused"

    def make_scan_line(self, sensor_data, transmit_as_receive):
        # get the receive apodization
        if transmit_as_receive:
            num_element_signals = len(self.not_transducer.active_elements)
            delays = -self.not_transducer.beamforming_delays
        else:
            num_element_signals = len(self.active_elements)
            delays = np.zeros(num_element_signals)
            # print('beamforming for custom focused transducer not yet implemented - will not apply time delays for receive signal')

        if len(self.active_elements) > 1:
            if self.receive_apodization == "Rectangular":
                self.receive_apodization = "Rectangular"
            apodization, _ = kwave.utils.signals.get_win(
                num_element_signals, type_=self.receive_apodization
            )
        else:
            apodization = 1
        apodization = np.array(apodization)

        # offset the received sensor_data by the beamforming delays and apply receive apodization
        for element_index in range(num_element_signals):
            if delays[element_index] > 0:
                # shift element data forwards
                sensor_data[element_index, :-1] = apodization[
                    element_index
                ] * np.concatenate(
                    (
                        sensor_data[element_index, 1 + delays[element_index] :],
                        np.zeros(delays[element_index]),
                    )
                )
            elif delays[element_index] < 0:
                # shift element data backwards
                sensor_data[element_index, :-1] = apodization[
                    element_index
                ] * np.concatenate(
                    (
                        np.zeros(-delays[element_index]),
                        sensor_data[element_index, : -delays[element_index]],
                    )
                )

        # form the a-line summing across the elements
        line = np.sum(sensor_data, axis=0)

        return line

    @classmethod
    def load(cls, transducer_dict):
        transducer = cls()

        ray_transforms = []
        for ray_dict in transducer_dict["ray_transforms"]:
            ray_transforms.append(geometry.Transform.load(ray_dict))
        transducer.ray_transforms = ray_transforms
        transducer_dict.pop("ray_transforms")
        transducer_dict["active_elements"] = np.array(
            transducer_dict["active_elements"]
        )
        transducer_dict["sensor_coords"] = np.array(transducer_dict["sensor_coords"])
        transducer_dict["steering_angles"] = np.array(
            transducer_dict["steering_angles"]
        )
        transducer_dict["pulse"] = np.array(transducer_dict["pulse"])
        for key, value in transducer_dict.items():
            setattr(transducer, key, value)
        return transducer

    def preprocess(
        self,
        scan_lines,
        t_array,
        sim_properties,
        window_factor=4,
        attenuation_factor=1,
        saft=False,
        demodulate=True,
    ) -> np.ndarray:
        scan_lines = self.window(scan_lines, window_factor)
        scan_lines = self.gain_compensation(
            scan_lines, t_array, sim_properties, attenuation_factor
        )
        scan_lines = kwave.utils.filters.gaussian_filter(
            scan_lines,
            1 / (t_array[-1] / t_array.shape[0]),
            self.harmonic * self.get_freq(),
            self.bandwidth,
        )
        if not saft:
            scan_lines = self.envelope_detection(scan_lines)
        scan_lines = self.window(scan_lines, window_factor)
        if self.compression_fac is not None:
            scan_lines = kwave.reconstruction.tools.log_compression(
                scan_lines, self.compression_fac, self.normalize
            )
        return scan_lines


class Planewave(Transducer):
    def __init__(
        self,
        label=None,
        max_frequency=2e6,
        source_strength=1e6,
        cycles=2,
        elements=32,
        active_elements=None,
        width=1e-2,  # transducer total width
        height=1e-2,  # transducer total width
        radius=float("inf"),
        focus_azimuth=float("inf"),
        focus_elevation=20e-3,
        sensor_sampling_scheme="centroid",
        sweep=np.pi / 3,
        ray_num=64,
        imaging_ndims=2,  # should be 2 or 3
        steering_angles=None,
        transmit_apodization="Rectangular",
        receive_apodization="Rectangular",
        transmit=True,
    ):
        super().__init__(
            label,
            max_frequency,
            source_strength,
            cycles,
            elements,
            active_elements,
            width,
            height,
            radius,
            focus_azimuth,
            focus_elevation,
            sensor_sampling_scheme,
            sweep,
            ray_num,
            imaging_ndims,
            transmit_apodization,
            receive_apodization,
            transmit,
        )

        self.set_steering_angles(imaging_ndims, sweep, self.ray_num)
        if steering_angles is not None:
            if type(steering_angles) != list:
                raise Exception("Please supply steering angles as a list")
            else:
                self.steering_angles = np.asarray(steering_angles)
        self.type = "planewave"

    def set_steering_angles(self, imaging_ndims, sweep, ray_num):
        ray_transforms, yaws = self.make_ray_transforms(imaging_ndims, sweep, ray_num)[
            :2
        ]
        if imaging_ndims == 2:
            self.steering_angles = yaws
            self.ray_transforms = [
                geometry.Transform(rotation=(0, 0, 0)) for yaw in yaws
            ]
        else:
            self.ray_transforms = ray_transforms
            self.steering_angles = np.zeros((1, self.get_num_rays()))

    @classmethod
    def load(cls, transducer_dict):
        transducer = cls()

        ray_transforms = []
        for ray_dict in transducer_dict["ray_transforms"]:
            ray_transforms.append(geometry.Transform.load(ray_dict))
        transducer.ray_transforms = ray_transforms
        transducer_dict.pop("ray_transforms")
        transducer_dict["active_elements"] = np.array(
            transducer_dict["active_elements"]
        )
        transducer_dict["sensor_coords"] = np.array(transducer_dict["sensor_coords"])
        transducer_dict["steering_angles"] = np.array(
            transducer_dict["steering_angles"]
        )
        transducer_dict["pulse"] = np.array(transducer_dict["pulse"])
        for key, value in transducer_dict.items():
            setattr(transducer, key, value)
        return transducer

    # def preprocess(self, scan_lines, t_array, sim_properties, window_factor=8,) -> np.ndarray:
    #     scan_lines = self.window(scan_lines, window_factor)
    #     scan_lines = self.gain_compensation(scan_lines, t_array, sim_properties)
    #     if self.compression_fac is not None:
    #         scan_lines = kwave.reconstruction.tools.log_compression(scan_lines, self.compression_fac, self.normalize)

    #     return scan_lines

    def preprocess(
        self,
        scan_lines,
        t_array,
        sim_properties,
        window_factor=4,
        attenuation_factor=1,
        saft=False,
        demodulate=False,
        gain_compensate=False,
    ) -> np.ndarray:
        scan_lines = self.window(scan_lines, window_factor)
        if demodulate:
            scan_lines = self.envelope_detection(scan_lines)
        if gain_compensate or attenuation_factor != 1:
            scan_lines = self.gain_compensation(
                scan_lines, t_array, sim_properties, attenuation_factor
            )
            scan_lines = self.window(scan_lines, window_factor)
        return scan_lines
