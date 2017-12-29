from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

#variable declarations

#voxel dimensions in meters (length,width,height of each voxel)
vox_dim = np.array([1.5, 0.4, 0.4])

#overall Dimensions
space_dim = np.array([4.5, 4.0, 4.0])

#scale factor look at cm instead of m
scale = 100

#origin shift required to have all values positive
shift = np.array([space_dim[0], space_dim[1] / 2.0, space_dim[2] / 2.0])

#how many voxels in each dimension
vox_blocks = (space_dim / vox_dim)

#voxDx in terms of scaling value
vox_dim = (vox_dim * scale).astype(int)

#maximum allowable points per voxel
max_pts = 30
feature_length = 6


def get_data_sets():

    #set the range of files we read in
    #read in for training data

    data_sets = []
    data_labels = []

    #training samples of boxes
    for file_num in range(0,100):
        filename = "data/training_data/sample"+str(file_num)+".txt"
        data = np.genfromtxt(filename, delimiter=',', skip_header=3)

        if len(data_sets) == 0:
            #print("Length 0")
            data_sets = np.array([data])
            data_labels = np.array([1])
        else:
            #print("Length > 0")
            data_sets = np.concatenate((data_sets, [data]), axis=0)
            data_labels = np.append(data_labels, 1)

    #training samples of spheres
    for file_num in range(100, 200):
        filename = "data/training_data/sample"+str(file_num)+".txt"
        data = np.genfromtxt(filename, delimiter=',', skip_header=3)

        if len(data_sets) == 0:
            data_sets = np.array([data])
            data_labels = np.array([0])
        else:
            data_sets = np.concatenate((data_sets, [data]), axis=0)
            data_labels = np.append(data_labels, 0)

    data_sets = data_sets.astype(np.float32)
    data_labels = np.asarray(data_labels, dtype=np.int32)

    return data_sets, data_labels


# function for turning 2500x3 matrix into nx6 dimensional feature vector
def find_feature_vec(pointcloud):
    # print("point cloud: "+str(pointcloud.shape))

    num_voxels = int(np.prod(vox_blocks))
    # print("Number of voxels: "+str(num_voxels))
    #full data starts as all 0's --> (number of voxels) x (number of points per voxel) x (pt length)
    #full_data = np.zeros((num_voxels,max_pts_voxel,3))

    voxel_list = [np.array([])]*(num_voxels)

    all_pts = pointcloud

    pt_num = 0
    #sort the points into voxels
    for pt in pointcloud:
        #scale and shift the point
        pt = pt + shift
        pt = (pt*scale).astype(int)

        all_pts[pt_num]=pt
        pt_num += 1

        local_pos = pt % vox_dim
        block_pos = (pt / vox_dim).astype(int)
        vox_id = int(block_pos[2] + block_pos[1] * vox_blocks[2] + block_pos[0] * vox_blocks[2] * vox_blocks[1])

        if len(voxel_list[vox_id]) == 0:
            voxel_list[vox_id] = [local_pos]
        else:
            voxel_list[vox_id] = np.concatenate((voxel_list[vox_id], [local_pos]), axis=0)

    mean_shift_list = list(voxel_list)
    for id_ in range(num_voxels):
        if len(voxel_list[id_]) > 0:
            mean_shift_list[id_] = voxel_list[id_] - np.mean(voxel_list[id_], axis=0).astype(int)

    full_data = np.zeros((num_voxels,max_pts,feature_length))
    for voxel in range(num_voxels):
        if len(voxel_list[voxel]) > 0:
            #add all points if less than max_pts
            if len(voxel_list[voxel]) <= max_pts:
                for pts in range(len(voxel_list[voxel])):
                    full_data[voxel][pts] = np.append(voxel_list[voxel][pts],mean_shift_list[voxel][pts])

                #add local pos and mean shifted pos to full_data if exists
            else:
                combined = np.zeros((len(voxel_list[voxel]),feature_length))
                #go through and combine local pos and mean shifted pos
                for pts in range(len(voxel_list[voxel])):
                    combined[pts] = np.append(voxel_list[voxel][pts],mean_shift_list[voxel][pts])
                #shuffle up the points
                shuffled = combined
                np.random.shuffle(shuffled)

                for pts in range(max_pts):
                    full_data[voxel][pts] = shuffled[pts]

    #reshape full_data to be (num_voxels*max_pts)x(feature_length)
    full_data = np.reshape(full_data, (num_voxels*max_pts, feature_length))

    #print("Shape of full data: "+str(full_data.shape))

    #apply 2 conv filters to bring to n x 32 and then n x 128

    full_data = full_data.astype(np.float16)

    return full_data


if __name__ == "__main__":
    # Load training and eval data

    data_sets, data_labels = get_data_sets()

    # convert point cloud into feature vectors
    for filenum in range(data_sets.shape[0]):
        np.savetxt("featureVecs/train_" + str(filenum) + ".txt", find_feature_vec(data_sets[filenum]))
        print("Saved file: " + str(filenum))
