import numpy as np

def read_poses(pose_file):

        with open(pose_file, "r") as file:
                for line in file:
                        if not line.isspace():
                                current_line = line.rstrip()
                                current_line = line.split(",")
                                translation = np.fromstring(current_line[0], sep = ' ')
                                rotation = np.fromstring(current_line[1], sep = ' ')
                                print(rotation)
#                                print(translation[0]) print(rotation)

def sensor_coords(n_el, el_width, kerf, elevation, numpts, sampling_scheme = None):
        azimuth = n_el*(el_width + kerf) - kerf
        if sampling_scheme == "centroid":
                sensor_z_coords = 0
        else:
                sensor_z_coords = np.linspace(elevation/2, -elevation/2, num = numpts)
        sensor_y_coords = np.transpose(np.linspace(-azimuth/2 + el_width/2, azimuth/2 - el_width/2, num = n_el))
        sensor_coords = np.zeros((n_el, numpts, 3))
        for col in range(numpts):
            sensor_coords[:, col, 1] = sensor_y_coords
        for row in range(n_el):
            sensor_coords[row, :, 2] = sensor_z_coords
        return sensor_coords


def remove_worm(index = None, label = None): # remove the transducer at position i
        if label is not None:
            index = None
            for i in range(len(wormlist)):
                if wormlist[i].author == label:
                    index = i
                    break
        if index is None:
            raise Exception("Please supply either a valid index or a valid label in order to remove a worm")
        else:
            del wormlist[index]


class Baby:

    def __init__(self, my_dict, author='Richard', idnum=None, vehicle = None):
        for key, value in my_dict.items():
            setattr(self, key, value)
        self.author = author
        if not hasattr(self, 'idnum'):
            self.idnum = idnum
        if not 'vehicle' in locals():
            print("no vehicle found")


