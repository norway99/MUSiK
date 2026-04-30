import numpy as np
from .utils import utils
from dataclasses import dataclass, field
from typing import List, Optional, Union, Tuple, Dict, Any, Type
from .transducer import Focused


@dataclass
class Sensor:
    """
    Represents an ultrasound sensor with spatial coordinates and transducer information.
    
    This class handles sensor configurations for various aperture types, including:
    - transmit_as_receive: Uses the same elements for transmitting and receiving
    - extended_aperture: Uses a synthetic/extended aperture configuration
    - pressure_field: Records pressure across a field
    - microphone: Single-point receiver at specified coordinates
    
    Sensor points are represented in global coordinate space for this class.
    """
    sensor_type: Optional[str] = None
    transducer_set: Optional[Any] = None
    sensor_coords: Optional[np.ndarray] = None
    aperture_type: Optional[str] = None  # transmit_as_receive, extended_aperture, pressure_field, microphone
    element_lookup: Optional[np.ndarray] = field(default=None, init=False)
    sensors_per_el: Optional[np.ndarray] = field(default=None, init=False)
    
    def __post_init__(self):
        """Initialize derived attributes after dataclass initialization."""
        if self.aperture_type == "microphone":
            if self.sensor_coords is None:
                raise Exception(
                    "Please supply sensor coordinates to use a microphone-style sensor mask"
                )
        elif self.aperture_type is None or self.aperture_type == "pressure_field":
            self.sensor_coords = None
            self.element_lookup = np.array([])
            self.sensors_per_el = np.array([])
        else:
            if self.transducer_set is None:
                raise Exception("Please supply a transducer set")
            all_sensor_coords = []
            all_element_lookup = []
            element_shift = 0
            sensors_per_el = []
            for key in range(len(self.transducer_set)):
                t = self.transducer_set.transducers[key]
                p = self.transducer_set.poses[key]
                assert p is not None, (
                    f"Pose for transducer {key} has not been assigned."
                )
                sensor_coords = t.get_sensor_coords()
                t_element_lookup = np.arange(sensor_coords.shape[0], dtype=np.int64)
                t_sensors_per_el = t.get_sensors_per_el()
                t_element_lookup = t_element_lookup // t_sensors_per_el
                t_element_lookup = np.add(t_element_lookup, element_shift)
                all_element_lookup.append(t_element_lookup)
                global_sensor_coords = p.apply_to_points(sensor_coords, inverse=False)
                all_sensor_coords.append(global_sensor_coords)
                sensors_per_el.append(np.ones(t.get_num_elements()) * t_sensors_per_el)
                element_shift = t_element_lookup[-1] + 1
            self.element_lookup = np.concatenate(all_element_lookup, axis=0)
            self.sensor_coords = np.concatenate(all_sensor_coords, axis=0)
            self.sensors_per_el = np.concatenate(sensors_per_el, axis=0)

    @classmethod
    def load(cls, filepath: str, transducer_set: Optional[Any] = None) -> 'Sensor':
        """
        Load a saved Sensor from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            transducer_set: Optional transducer set to use
            
        Returns:
            Loaded Sensor object
        """
        sensor = cls()
        my_dict = utils.json_to_dict(filepath)
        for key, value in my_dict.items():
            setattr(sensor, key, value)
        if transducer_set is not None:
            sensor.transducer_set = transducer_set
        else:
            print("loading sensor without transducer set, mind the aperture type!")
        if sensor.sensor_coords is not None:
            sensor.sensor_coords = np.array(sensor.sensor_coords)
        if sensor.element_lookup is not None:
            sensor.element_lookup = np.array(sensor.element_lookup)
        if sensor.sensors_per_el is not None:
            sensor.sensors_per_el = np.array(sensor.sensors_per_el)
        return sensor

    def save(self, savefile: str) -> None:
        """
        Save the Sensor to a JSON file.
        
        Args:
            savefile: Path to save the JSON file
        """
        save_dict = {}
        save_dict["aperture_type"] = self.aperture_type
        save_dict["sensor_coords"] = self.to_list()
        save_dict["element_lookup"] = np.ndarray.tolist(self.element_lookup)
        save_dict["sensors_per_el"] = np.ndarray.tolist(self.sensors_per_el)
        utils.dict_to_json(save_dict, savefile)

    def to_list(self, my_list: Optional[np.ndarray] = None) -> Optional[List[Tuple]]:
        """
        Convert sensor coordinates to a list of tuples.
        
        Args:
            my_list: Optional list to convert instead of using sensor_coords
            
        Returns:
            List of coordinate tuples or None if no coordinates
        """
        if my_list is None:
            if self.sensor_coords is None:
                return None
            my_list = self.sensor_coords
        sensor_coord_list = []
        for coord in my_list:
            sensor_coord_list.append(tuple(coord))
        return sensor_coord_list

    def make_sensor_mask(
        self, 
        sim_transducer: Any, 
        not_transducer: Any, 
        grid_voxel_size: Union[float, np.ndarray],
        transmit_transform: Optional[Any] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create a binary mask representing sensor locations.
        
        Takes in a list of sensor coords (global coordinate system), transforms 
        to match reference of transmit transducer, and discretizes.
        
        Args:
            sim_transducer: Transducer used for simulation
            not_transducer: Transducer for sensor mask
            grid_voxel_size: Size of voxels in simulation grid
            transmit_transform: Optional transform to apply
            
        Returns:
            Tuple of (sensor_mask, discretized_sensor_coords)
        """
        if self.aperture_type == "transmit_as_receive":
            sensor_mask = not_transducer.indexed_mask
            sensor_mask = np.where(sensor_mask > 0, 1, sensor_mask)
            discretized_sensor_coords = None
        else:
            sensor_mask = np.zeros(not_transducer.indexed_mask.shape)
            if self.aperture_type == "pressure_field":
                sensor_mask[:, :, sensor_mask.shape[2] // 2] = 1
                discretized_sensor_coords = None
            else:
                if transmit_transform is None:
                    raise Exception(
                        "Please supply a transmit transducer affine transformation"
                    )
                transformed_sensor_coords = transmit_transform.apply_to_points(
                    self.sensor_coords, inverse=True
                )
                if isinstance(sim_transducer, Focused):
                    label = sim_transducer.get_label()
                    index = self.transducer_set.find_transducer(label)
                    if index is None:
                        raise Exception(
                            "To use n focused transducers with a synthetic/extended aperture, please instantiate with n unique labels."
                        )
                    start_coord = 0
                    for t in self.transducer_set.transducers[:index]:
                        start_coord += t.get_num_elements() * t.get_sensors_per_el()
                    transformed_sensor_coords[
                        start_coord : (
                            sim_transducer.get_num_elements()
                            * sim_transducer.get_sensors_per_el()
                        ),
                        :,
                    ] = sim_transducer.sensor_coords
                transformed_sensor_coords = np.divide(
                    transformed_sensor_coords, grid_voxel_size
                )
                mask_centroid = np.array(sensor_mask.shape) / 2
                mask_centroid[0] = 0
                recenter_matrix = np.broadcast_to(
                    mask_centroid, transformed_sensor_coords.shape
                )
                transformed_sensor_coords = transformed_sensor_coords + recenter_matrix
                discretized_sensor_coords = np.ndarray.astype(
                    np.round(transformed_sensor_coords), int
                )
                for coord in discretized_sensor_coords:
                    if np.prod(np.where(coord >= 0, 1, 0)) == 0:
                        continue
                    if np.prod(np.where(coord < sensor_mask.shape, 1, 0)) == 0:
                        continue
                    sensor_mask[coord[0], coord[1], coord[2]] = 1
        return sensor_mask, discretized_sensor_coords

    def voxel_to_element(
        self,
        sim_properties: Any,
        transmit: Any,
        discretized_sensor_coords: Optional[np.ndarray],
        sensor_data: Dict[str, Any],
        additional_keys: List[str],
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[Any]]]:
        """
        Convert voxel-level sensor data to element-level data.
        
        Takes sensor data and aggregates voxel-level data into element-level data.
        
        Args:
            sim_properties: Properties of the simulation
            transmit: Transmit configuration
            discretized_sensor_coords: Discretized sensor coordinates
            sensor_data: Dictionary of sensor data
            additional_keys: Additional data keys to process
            
        Returns:
            Signals or tuple of (signals, other_signals)
        """
        computational_grid_size = np.array(sim_properties.matrix_size) - 2 * np.array(
            sim_properties.PML_size
        )
        data = sensor_data["p"].T
        if self.aperture_type == "transmit_as_receive":
            signals = transmit.not_transducer.combine_sensor_data(data)
            other_signals = []
            for other_key in additional_keys:
                other_signals.append(sensor_data[other_key])
        else:
            # omissions
            condensed_discrete_coords = discretized_sensor_coords[
                np.where(
                    [
                        np.logical_and(
                            np.prod(np.where(discretized_sensor_coords[i] >= 0, 1, 0))
                            != 0,
                            np.prod(
                                np.where(
                                    discretized_sensor_coords[i]
                                    < computational_grid_size,
                                    1,
                                    0,
                                )
                            )
                            != 0,
                        )
                        for i in range(len(discretized_sensor_coords))
                    ],
                )
            ]

            # hashing
            hash_list = np.sum(
                condensed_discrete_coords
                * np.array(
                    [
                        1,
                        computational_grid_size[0],
                        computational_grid_size[0] * computational_grid_size[1],
                    ]
                ),
                axis=1,
            )
            # collisions
            hash_list = np.unique(hash_list)
            # sorting
            hash_list = np.array(sorted(hash_list))

            sensor_point_signals = np.array(
                [
                    self.hash_fn(coord, hash_list, data, computational_grid_size)
                    for coord in discretized_sensor_coords
                ]
            )

            if self.aperture_type == "microphone":
                return sensor_point_signals

            signals = np.zeros([len(self.sensors_per_el), data.shape[1]])
            count = 0
            for i, points in enumerate(list(self.sensors_per_el)):
                signals[i] = np.mean(
                    sensor_point_signals[int(count) : int(count + points)], axis=0
                )
                count += points

            other_signals = []
            for other_key in additional_keys:
                other_signals.append(sensor_data[other_key])

        return signals, other_signals

    def hash_fn(self, 
                coord: np.ndarray, 
                hash_list: np.ndarray, 
                data: np.ndarray, 
                computational_grid_size: np.ndarray
    ) -> np.ndarray:
        """
        Hashing function to retrieve data for a given coordinate.
        
        Args:
            coord: Coordinate to hash
            hash_list: List of hash values
            data: Data array
            computational_grid_size: Size of computational grid
            
        Returns:
            Data at the hashed coordinate
        """
        if np.logical_or(
            np.prod(np.where(coord >= 0, 1, 0)) == 0,
            np.prod(np.where(coord < computational_grid_size, 1, 0)) == 0,
        ):
            return np.zeros(data[0].shape)
        hash_val = np.sum(
            coord
            * np.array(
                [
                    1,
                    computational_grid_size[0],
                    computational_grid_size[0] * computational_grid_size[1],
                ]
            )
        )
        index = np.where(hash_list == hash_val)[0][0]
        return data[index]

    def sort_pressure_field(self, 
                          sensor_data: Dict[str, Any], 
                          additional_keys: List[str], 
                          grid_shape: Tuple[int, int], 
                          PML_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Sort and reshape pressure field data.
        
        Args:
            sensor_data: Dictionary of sensor data
            additional_keys: Additional keys to process
            grid_shape: Shape of the grid
            PML_size: Size of PML layers
            
        Returns:
            Tuple of (signals, other_signals)
        """
        signals = (
            sensor_data["p"]
            .T.reshape(grid_shape[1], grid_shape[0], -1)
            .transpose(1, 0, 2)
        )
        other_signals = []
        for other_key in additional_keys:
            other_signals.append(
                sensor_data[other_key]
                .T.reshape(
                    grid_shape[1] + PML_size[1] * 2, grid_shape[0] + PML_size[0] * 2
                )
                .transpose(1, 0)
            )
        return signals, other_signals
