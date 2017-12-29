import numpy as np
import scipy.misc


# specified
num_ftvec_files = 200
num_classes = 2
eval_split = 0.1

# computed - do not change
ftvec_per_class = num_ftvec_files / num_classes
train_split = 1 - eval_split

# pc = per class
num_train_pc = train_split * ftvec_per_class
num_eval_pc = eval_split * ftvec_per_class


def get_feature_sets():
    train_features = []

    for file_num in range(0, num_ftvec_files):
        filename = "featureVecs/train_"+str(file_num)+".txt"
        data = np.loadtxt(filename)

        if len(train_features) == 0:
            train_features = np.array([data])
        else:
            train_features = np.concatenate((train_features,[data]), axis=0)

    train_labels = np.append(np.ones(100), np.zeros(100))

    i_train = np.r_[0:num_train_pc, ftvec_per_class:ftvec_per_class + num_train_pc]
    i_eval = np.r_[num_train_pc:num_train_pc + num_eval_pc, ftvec_per_class + num_train_pc: ftvec_per_class +
                                                                                            num_train_pc + num_eval_pc]
    eval_features = train_features[i_eval].astype(np.float16)
    eval_labels = train_labels[i_eval]
    train_features = train_features[i_train].astype(np.float16)
    train_labels = train_labels[i_train]

    p = np.random.permutation(len(train_labels))
    train_features = train_features[p]
    train_labels = train_labels[p]

    p = np.random.permutation(len(eval_labels))
    eval_features = eval_features[p]
    eval_labels = eval_labels[p]

    return train_features, train_labels, eval_features, eval_labels
