from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np

#import read data function
import read_data as rd

#variable declarations

#voxel dimensions in meters (length,width,height of each voxel)
voxDim = np.array([1.5, 0.4, 0.4])

#overall Dimensions
spaceDim = np.array([4.5,4.0,4.0])

#scale factor look at cm, or m instead of m
scale = 100

#origin shift required to have all values positive
shift = np.array([spaceDim[0],spaceDim[1]/2.0,spaceDim[2]/2.0])

#how many voxels in each dimension
voxBlocks = (spaceDim / voxDim)
#voxDx in terms of scaling value
voxDim = (voxDim*scale).astype(int)

#maximum allowable points per voxel
max_pts = 30
feature_length = 6



# function for turning 2500x3 matrix into nx6 dimensional feature vector
def findFeatureVec(pointcloud):
    # print("point cloud: "+str(pointcloud.shape))

    num_voxels = int(np.prod(voxBlocks))

    voxel_list = [np.array([])]*(num_voxels)

    all_pts = pointcloud

    pt_num = 0
    #sort the points into voxels
    for pt in pointcloud:
        #scale and shift the point
        pt = pt + shift
        pt = (pt*scale).astype(int)

        all_pts[pt_num]=pt
        pt_num+=1

        localPos = pt%voxDim
        blockPos = (pt/voxDim).astype(int)
        voxID = int(blockPos[2] + blockPos[1]*voxBlocks[2] + blockPos[0]*voxBlocks[2]*voxBlocks[1])

        if len(voxel_list[voxID]) == 0:
             voxel_list[voxID] = [localPos]
        else:
            voxel_list[voxID] = np.concatenate((voxel_list[voxID], [localPos]), axis=0)

    meanShiftList = list(voxel_list)
    for id in range(num_voxels):
        #print id
        if len(voxel_list[id]) > 0:
            # print("voxel list at index [" +str(id)+"]: "+str(voxel_list[id]))
            # print("mean of voxel list at index: "+str(np.mean(voxel_list[id],axis=0).astype(int)))
            # print("local1: "+str(voxel_list[id]))
            meanShiftList[id] = voxel_list[id] - np.mean(voxel_list[id],axis=0).astype(int)

    full_data = np.zeros((num_voxels,max_pts,feature_length))
    for voxel in range(num_voxels):
        if len(voxel_list[voxel]) > 0:
            #add all points if less than max_pts
            if len(voxel_list[voxel]) <= max_pts:
                for pts in range(len(voxel_list[voxel])):
                    full_data[voxel][pts] = np.append(voxel_list[voxel][pts],meanShiftList[voxel][pts])

                    #add local pos and mean shifted pos to full_data if exists
                #print("full data at given voxel: "+str(full_data[voxel]))
            else:
                # print("need to downsample this voxel")
                combined = np.zeros((len(voxel_list[voxel]),feature_length))
                #go through and combine local pos and mean shifted pos
                for pts in range(len(voxel_list[voxel])):
                    combined[pts] = np.append(voxel_list[voxel][pts],meanShiftList[voxel][pts])
                # print("combined: "+str(combined))
                #shuffle up the points
                shuffled = combined
                np.random.shuffle(shuffled)
                #print("num pts: "+str(len(voxel_list[voxel])))
                for pts in range(max_pts):
                    full_data[voxel][pts] = shuffled[pts]

    full_data = np.reshape(full_data, (num_voxels*max_pts,feature_length))

    full_data = full_data.astype(np.float16)
    print("Full data: "+str(full_data))

    return full_data


if __name__ == "__main__":
    data_sets, data_labels = rd.get_data_sets()

    # train_data = mnist.train.images # Returns np.array
    # train_data = np.random.rand(100,2500).astype(np.float32)
    print("training: " + str(data_sets.shape))
    # #train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    # train_labels = np.asarray(np.random.randint(2, size=100), dtype=np.int32)
    print("train_labels: " + str(data_labels.shape))

    # convert point cloud into feature vectors
    for filenum in range(data_sets.shape[0]):
        np.savetxt("featureVecs/train_" + str(filenum) + ".txt", findFeatureVec(data_sets[filenum]))
        print("Saved file: " + str(filenum))

