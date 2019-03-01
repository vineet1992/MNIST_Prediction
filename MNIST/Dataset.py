import numpy as np
from keras.utils.np_utils import to_categorical
import random
import struct
import gzip
import math


###We define a dataset as a set of images and a set of labels for each image###

magicLabel = 2049 ###Magic number for reading label files
magicImages = 2051 ###Magic number for reading image files

class Dataset:


    def __init__(self,imageFile,labelFile,partition = None):
        '''Parameters:
         imageFile - The training image file (absolute path)
         labelFile - The label file (absolute path)
         partition - proportion of the dataset to be allocated to the Training Set
         '''

        ###Initialize random object and seed
        self.random = random
        self.random.seed = random.seed(42)

        ###Save image and label files
        self.train = imageFile
        self.labels = labelFile

        ###Default partition value is 90% train-test split
        if(partition==None):
            self.partition = 0.9
        else:
            self.partition = partition

        ###Load data and labels from file
        data = self._load_data()
        labels = self._load_labels()

        ###Merge datasets together
        allData = np.column_stack((data,labels))

        ###Shuffle the data according to the random seed
        self.random.shuffle(allData)

        ###Save the shuffled dataset
        self.allData = allData


        ###Ensure correctness of the data
        self.runTests()

    def _load_data(self):
        '''Returns a shuffled list of Numpy 3D Matrices (x)
        Parses the image and label zip files via binary reading
        Note that for optimal efficiency this assumes that the file can be read into memory (no buffering of data)'''
        with gzip.open(self.train, 'rb') as f:
            bytes_read = f.read()

        currByte = 0
        ##Read the first byte (magic number)
        magic = struct.unpack('>i', bytes_read[currByte:(currByte+4)])[0]
        currByte +=4

        # Magic number should equal 2051 if read properly
        assert magic == magicImages, "Incorrect download of the image file, please re-download and ensure connectivity"

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
            ###Read one byte per pixel
            format = '>' + ('B' * totalPixels )

            ###Convert this to integers
            image = struct.unpack(format, bytes_read[currByte:(currByte+totalPixels)])

            ###Increment the file pointer to the next reading area
            currByte = currByte + (rows*cols)

            ###Append the image to the fullset of images
            allImages.append(image)

        ###Return a numpy array format of all images
        return np.array(allImages)


    def _load_labels(self):
        '''
        Loads the labels for each image from the downloaded dataset
        Note that download.py must be run prior to loading data or labels
        :return: A numpy array of image labels as integers
        '''

        with gzip.open(self.labels, 'rb') as f:
            bytes_read = f.read()

        currByte = 0
        ##Read the first byte (magic number)
        magic = struct.unpack('>i', bytes_read[currByte:(currByte+4)])[0]
        currByte +=4

        # Magic number should equal 2049 if read properly
        assert magic == magicLabel, "Incorrect download of the label file, please repeat the download and ensure connectivity"

        ##Read number of images
        numImages = struct.unpack('>i', bytes_read[currByte:(currByte+4)])[0]

        ###Increment one integer ahead (4 bytes)
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
        '''Returns the full set of labels as one-hot encoded categorical vectors'''
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
        testX = data[:,0:(data.shape[1]-1)]
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

    def runTests(self):
        rows = self.allData.shape[0] ###number of samples
        cols = self.allData.shape[1] ###number of pixels + 1

        ###Squareroot of cols-1 should be a whole number

        if not math.sqrt(cols-1).is_integer():
            print("Failed image is a square test")
            exit(-1)

        ###confirm dimensions of trainX,testX,trainY,testY based on value of partition
        trainX = self.getTrainX()
        trainY = self.getTrainY()
        testX = self.getTestX()
        testY = self.getTestY()
        trainRows = self.partition*rows
        testRows = rows-trainRows

        if not trainX.shape[0]==trainRows:
            print("Training set has the wrong number of rows based on partition = " + str(self.partition) + ", Found: " + str(trainX.shape[0]) + ", expected: " + str(trainRows))
            exit(-1)
        if not testX.shape[0]==testRows:
            print("Testing set has the wrong number of rows based on partition = " + str(self.partition) + ", Found: " + str(testX.shape[0]) + ", expected: " + str(testRows))
            exit(-1)
        if not trainX.shape[0]==len(trainY):
            print("Training set and training labels do not match")
            exit(-1)
        if not testX.shape[0]==len(testX):
            print("Testing set and testing labels do not match")
            exit(-1)


        print("All tests passed for checking dataset consistency")