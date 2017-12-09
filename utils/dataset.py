import argparse
import pprint
import time, os, sys
import tensorflow as tf
import glob
import numpy as np
import pickle as p
import os.path
import os

from sklearn.utils import shuffle

num_sample_points = 2500

scale = 100

D = 4.5
H = 4.0
W = 4.0

#origin shift
shiftX = D
shiftY = W / 2
shiftZ = H / 2

D_size = D * scale
H_size = H * scale
W_size = W * scale

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Reading a point cloud file')
    parser.add_argument('--filepath', dest='filepath', help='path for files containg point cloud data',
                        default='cpu', type=str)

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    _args = parser.parse_args()
    return _args


#The point cloud data will be stored as voxels
def load_train(train_path):

    file_path = train_path + 'pointclouds.p'
    if not os.path.exists(file_path):
        pointclouds = []
        positions = []
        orientations = []
        labels = []
        print('Going to read training images')
        print(train_path)
        for file_name in os.listdir(train_path):
            print(file_name)
            if not file_name.endswith('.txt'):
                continue
            pointcloud = []
            position = []
            orientation = []
            label = []
            with open(train_path + '/' + file_name) as f:
                line_arr = f.readlines()
                point_lines = []
                for line in line_arr:
                    line_sp = line.strip().split(':')
                    if len(line_sp) == 1:
                        line_split = np.array(line.strip().split(','), dtype=np.float32)
                        line_split[0] += shiftX
                        line_split[1] += shiftY
                        line_split[2] += shiftZ

                        line_split *= scale
                        line_split = np.round(line_split)
                        line_split = line_split.astype(np.int32)

                        if line_split[1] > W_size - 1:
                            line_split[1] = W_size - 1

                        if line_split[2] > H_size - 1:
                            line_split[2] = H_size - 1

                        if line_split[1] < 0:
                            line_split[1] = 0

                        if line_split[2] < 0:
                            line_split[2] = 0

                        pointcloud.append(line_split)
                    else:
                        if line_sp[0] == "Label":
                            label.append(line_sp[1])
                            #print line_arr[1]
                        elif line_sp[0] == "Position":
                            position.append(line_sp[1])
                        elif line_sp[0] == "Orientation (Quaternion)":
                            orientation.append(line_sp[1])

            pointclouds.append(pointcloud)
            positions.append(position)
            orientations.append(orientation)
            labels.append(label)
        pointclouds = np.array(pointclouds)
        #print(pointclouds[0])
        labels = np.array(labels)
        orientations = np.array(orientations)
        positions = np.array(positions)

        with open(train_path + 'pointclouds.p', 'w') as fw1:
            p.dump(pointclouds, fw1)
        with open(train_path + 'positions.p', 'w') as fw2:
            p.dump(positions, fw2)
        with open(train_path + 'orientations.p', 'w') as fw3:
            p.dump(orientations, fw3)
        with open(train_path + 'labels.p', 'w') as fw4:
            p.dump(labels, fw4)
    else:
        with open(train_path + 'pointclouds.p', 'r') as f1:
            pointclouds = p.load(f1)
        with open(train_path + 'positions.p', 'r') as f2:
            positions = p.load(f2)
        with open(train_path + 'orientations.p', 'r') as f3:
            orientations = p.load(f3)
        with open(train_path + 'labels.p', 'r') as f4:
            labels = p.load(f4)

    return pointclouds, positions, orientations, labels


class DataSet(object):

    def __init__(self, pointclouds, positions, orientations, labels):
        self._num_examples = pointclouds.shape[0]
        self._pointclouds = pointclouds
        self._labels = labels
        self._positions = positions
        self._orientations = orientations
        self._epochs_done = 0
        self._index_in_epoch = 0

    @property
    def pointclouds(self):
        return self._pointclouds

    @property
    def labels(self):
        return self._labels

    @property
    def positions(self):
        return self._positions

    @property
    def orientations(self):
        return self._orientations

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_done(self):
        return self._epochs_done

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # After each epoch we update this
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._pointclouds[start:end], self._positions[start:end], self._orientations[start:end], self._labels[start:end]


def read_train_sets(train_path, validation_size):
    class DataSets(object):
        pass
    data_sets = DataSets()

    pointclouds, positions, orientations, labels = load_train(train_path)
    pointclouds, positions, orientations, labels = shuffle(pointclouds, positions, orientations, labels)

    print(pointclouds.shape[0])

    if isinstance(validation_size, float):
        validation_size = int(validation_size * pointclouds.shape[0])

    validation_pointclouds = pointclouds[:validation_size]
    validation_labels = labels[:validation_size]
    validation_positions = positions[:validation_size]
    validation_orientations = orientations[:validation_size]

    train_pointclouds = pointclouds[validation_size:]
    train_labels = labels[validation_size:]
    train_positions = positions[validation_size:]
    train_orientations = orientations[validation_size:]

    data_sets.train = DataSet(train_pointclouds,  train_positions, train_orientations, train_labels)
    data_sets.valid = DataSet(validation_pointclouds, validation_positions, validation_orientations, validation_labels)

    return data_sets

if __name__ == "__main__":
    args = parse_args()

    if len(args.filepath) == 0:
        filepath = "data/example_data/"

    filepath = "data/example_data/"
    print(filepath)
    load_train(filepath)
