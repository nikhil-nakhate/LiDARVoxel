import numpy as np


def get_data_sets():

    #set the range of files we read in
    #read in for training data

    data_sets = []
    data_labels = []

    #training samples of boxes
    for fileNum in range(0,100):
        filename = "data/training_data/sample"+str(fileNum)+".txt"
        data= np.genfromtxt(filename, delimiter=',', skip_header=3)

        if len(data_sets) == 0:
            #print("Length 0")
            data_sets = np.array([data])
            data_labels = np.array([1])
        else:
            #print("Length > 0")
            data_sets = np.concatenate((data_sets,[data]), axis=0)
            data_labels = np.append(data_labels,1)

        #print("train_data shape: " + str(train_data.shape))

    #training samples of spheres
    for fileNum in range(100,200):
        filename = "data/training_data/sample"+str(fileNum)+".txt"
        data= np.genfromtxt(filename, delimiter=',', skip_header=3)

        if len(data_sets) == 0:
            data_sets = np.array([data])
            data_labels = np.array([0])
        else:
            data_sets = np.concatenate((data_sets,[data]), axis=0)
            data_labels = np.append(data_labels,0)


    print("READ IN DATA FROM FILES")
    data_sets = data_sets.astype(np.float32)
    data_labels = np.asarray(data_labels, dtype=np.int32)

    return data_sets, data_labels


def get_feature_sets():
    train_features = []

    for fileNum in range(0,200):
        filename = "featureVecs/train_"+str(fileNum)+".txt"
        data = np.loadtxt(filename)
        #print("shape of data: "+str(data.shape))
        #data = np.reshape(data,(-1,50))

        if len(train_features) == 0:
            #print("Length 0")
            train_features = np.array([data])
        else:
            #print("Length > 0")
            train_features = np.concatenate((train_features,[data]), axis=0)

    train_labels = np.append(np.ones(100),np.zeros(100))
    print("training labels: "+str(train_labels))
    print("shape of samples: "+str(train_features.shape))

    #shufle and pull out evaluation samples
    p = np.random.permutation(len(train_labels))
    train_features = train_features[p]
    train_labels = train_labels[p]

    eval_features = train_features[180:200].astype(np.float16)
    eval_labels = train_labels[180:200]
    train_features = train_features[0:180].astype(np.float16)
    train_labels = train_labels[0:180]

    return train_features, train_labels, eval_features, eval_labels


def get_depth_data_sets():
    train_data = []

    for fileNum in range(0,1500):
        filename = "data/depth_data/sample"+str(fileNum)+".txt"
        data = np.loadtxt(filename,skiprows=3)

        if fileNum%100==0:
            print("File "+str(fileNum))

        if len(train_data) == 0:
            #print("Length 0")
            train_data = np.array([data])
        else:
            #print("Length > 0")
            train_data = np.concatenate((train_data,[data]), axis=0)
    for fileNum in range(10000,11500):
        filename = "data/depth_data/sample"+str(fileNum)+".txt"
        data = np.loadtxt(filename,skiprows=3)
        if fileNum%100==0:
            print("File "+str(fileNum))

        if len(train_data) == 0:
            #print("Length 0")
            train_data = np.array([data])
        else:
            #print("Length > 0")
            train_data = np.concatenate((train_data,[data]), axis=0)

    train_labels = np.append(np.ones(1500),np.zeros(1500))

    i_train = np.r_[0:1000,1500:2500]
    i_eval = np.r_[1000:1500,2500:3000]
    eval_data = train_data[i_eval].astype(np.float16)
    eval_labels = train_labels[i_eval]
    train_data = train_data[i_train].astype(np.float16)
    train_labels = train_labels[i_train]

    p = np.random.permutation(len(train_labels))
    train_data = train_data[p]
    train_labels = train_labels[p]

    p = np.random.permutation(len(eval_labels))
    eval_data = eval_data[p]
    eval_labels = eval_labels[p]

    return train_data, train_labels, eval_data, eval_labels

