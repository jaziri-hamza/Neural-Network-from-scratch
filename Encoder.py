import numpy as np

"""
    A class for encode multiclass target label
    ....

"""
class Encoder:


    def __init__(self):
        pass

    def encode(self, label):
        self._unique_labels = np.unique(label)
        new_label = np.zeros((label.shape[0], len(self._unique_labels) ) )
        for i in range(label.shape[0]):
            new_label[i, np.squeeze(np.where( np.squeeze(label[i]) == self._unique_labels )) ] = 1
        return new_label
        
        # for i in range(label.shape[0]):


    def decode(self, encoded_label):
        new_label = np.zeros((encoded_label.shape[0], 1))
        for i in range(encoded_label.shape[0]):
            # raise Exception("")
            new_label[i,:] = self._unique_labels[ np.squeeze(np.where( np.squeeze(encoded_label[i,:]) == 1 )) ]
        return new_label

