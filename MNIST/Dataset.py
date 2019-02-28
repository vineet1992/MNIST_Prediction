from pathlib import Path
import numpy as np
from keras.utils.np_utils import to_categorical

###We define a dataset as a set of images and a set of labels for each image###

class Dataset:


    def __init__(self,imageFile,labelFile,partition = None):
        '''Parameters:
         imageFile - The training image file (absolute path)
         labelFile - The label file (absolute path)
         partition - proportion of the dataset to be allocated to the Training Set
         '''
        import random
        self.random = random
        self.random.seed = random.seed(42)
        self.train = imageFile
        self.labels = labelFile
        if(partition==None):
            self.partition = 0.9
        else:
            self.partition = partition
        data = self._load_data()
        labels = self._load_labels()
        allData = np.column_stack((data,labels))
        self.random.shuffle(allData)
        self.allData = allData

    def _load_data(self):
        '''Returns a shuffled list of 3D Matrices (x) and a list of output labels (y)
        Parses the image and label zip files via binary reading
        Note that for optimal efficiency this assumes that the file can be read into memory (no buffering of data)'''

        import struct
        import gzip
        x = []
        y = []
        with gzip.open(self.train, 'rb') as f:
            bytes_read = f.read()

        currByte = 0
        ##Read the first byte (magic number)
        magic = struct.unpack('>i', bytes_read[currByte:(currByte+4)])[0]
        currByte +=4

        # Magic number should equal 2051 if read properly
        assert (magic == 2051)

        ##Read number of images
        numImages = struct.unpack('>i', bytes_read[currByte:(currByte+4)])[0]

        currByte+=4
        ###Read number of rows in each image
        rows = struct.unpack('>i', bytes_read[currByte:(currByte+4)])[0]
        currByte+=4

        ##Read number of columns in each image
        cols = struct.unpack('>i', bytes_read[currByte:(currByte+4)])[0]
        currByte+=4

        ###Read Pixel level data (for now each image is a full vector)
        totalPixels = rows*cols ##Total number of pixels in an image
        allImages = []
        for i in range(0, numImages):
            format = '>' + ('B' * totalPixels )
            image = struct.unpack(format, bytes_read[currByte:(currByte+totalPixels)])
            currByte = currByte + (rows*cols)
            allImages.append(image)
        return np.array(allImages)


    def _load_labels(self):
        import struct
        import gzip
        with gzip.open(self.labels, 'rb') as f:
            bytes_read = f.read()

        currByte = 0
        ##Read the first byte (magic number)
        magic = struct.unpack('>i', bytes_read[currByte:(currByte+4)])[0]
        currByte +=4

        # Magic number should equal 2051 if read properly
        assert (magic == 2049)

        ##Read number of images
        numImages = struct.unpack('>i', bytes_read[currByte:(currByte+4)])[0]

        currByte+=4
        ###Read each label and append to output vector
        y = struct.unpack('>' + 'B' * numImages, bytes_read[currByte:(currByte+numImages)])


        return np.array(y)

    def get_full_set(self):
        '''Returns the entire dataset'''
        return self.allData


    def get_train_set(self):
        ''' Returns the first <partition> % of the dataset (training data)'''
        data = self.get_full_set()
        return data[0:int(self.partition*len(data))]

    def getFullX(self):
        ''' Returns the full set of images'''
        data = self.get_full_set()
        return data[:,0:(data.shape[1]-1)]

    def getFullY(self):
        '''Returns the full set of labels'''
        data = self.get_full_set()
        Y = to_categorical(data[:,data.shape[1]-1])
        return Y


    def getTrainX(self):
        ''' Returns just the training images from the dataset '''
        data = self.get_train_set()
        return data[:,0:(data.shape[1]-1)]

    def getTrainY(self):
        ''' Returns just the training labels in one-hot encoded format'''
        data = self.get_train_set()
        trainY = to_categorical(data[:, data.shape[1] - 1])
        return trainY

    def getTestX(self):
        ''' Returns just the testing images from the dataset'''
        data = self.get_test_set()
        testX = data[:0:(data.shape[1]-1)]
        return testX

    def getTestY(self):
        ''' Returns just the testing labels from the dataset'''
        data = self.get_test_set()
        testY = to_categorical(data[:,data.shape[1]-1])
        return testY

    def get_test_set(self):
        ''' Returns the last <1-partition> % of the dataset (testing data)'''
        data = self.get_full_set()
        return data[int(self.partition*len(data)):len(data)]
