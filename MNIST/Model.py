###Takes as input a Dataset object (training) or a serialized model filename
### Methods for training and testing (which returns predictions given test data)


from keras.utils.np_utils import to_categorical #To one-hot encode the output labels
from keras.models import load_model
from MNIST.ModelParser import ModelParser
from MNIST.Exploration_Model import exploration_model
import math
import numpy as np
import os
import matplotlib.pyplot as mp
import talos as ta


class Model:

    def __init__(self, data, modelName,normalize=True):
        '''
        Default constructor for a model object
        :param data: Dataset object representing all of the data together
        :param modelName: The name for this model (used for output file generation)
        :param normalize: Should the data be normalized?
        '''

        self.data = data
        ###If the partitioning isn't 0 percent (only test set), then extract the training set
        if not data.partition==0:

            ###Extract training set
            trainSet = data.get_train_set()

            ###Extract only training images
            self.trainX = trainSet[:,0:(trainSet.shape[1]-1)]

            ###Extract number of rows in the images
            self.rows = int(math.sqrt(self.trainX.shape[1]))

            ###Normalize the images
            if(normalize):
                self.trainX = self.trainX/255.0

            ###Convert to categorical one-hot encoding
            self.trainY = to_categorical(trainSet[:,trainSet.shape[1]-1])


        if not data.partition==1:
            testSet = data.get_test_set()
            self.testX = testSet[:,0:(testSet.shape[1]-1)]
            self.rows = int(math.sqrt(self.testX.shape[1]))
            if(normalize):
                self.testX = self.testX/255.0
            self.testY = to_categorical(testSet[:,testSet.shape[1]-1])

        self.name = modelName
        self.normalize = normalize


    def train(self,modelFile):
        '''Parameters:
        modelFile - Model specification file (see README for correct syntax)
        Trains the model according to the specification file and outputs a trained keras model
        '''

        ###Create model parser object to handle model parsing
        parser = ModelParser(modelFile,self.trainX)

        ###Create CNN model using the parser
        output = parser.parse()
        self.model = output[0]
        self.epochs = output[1]
        self.batchSize = output[2]

        ###Reshape the data to work with Keras
        self.trainX = np.reshape(self.trainX,(-1,self.rows, self.rows, 1))

        self.model.fit(self.trainX, self.trainY,
                                 batch_size=self.batchSize,epochs=self.epochs)


    def reshape(self,X):
        '''Reshapes images array to fit MNIST format for Keras'''
        return np.reshape(X,(-1,self.rows,self.rows,1))

    def test(self):
        '''
        Creates predictions on the test set based on this trained model
        :return: Prediction accuracy and cross-entropy loss on the test set
        '''
        self.testX = np.reshape(self.testX,(-1,self.rows,self.rows,1))
        return self.model.evaluate(self.testX,self.testY)

    def saveModel(self):
        '''
        Serializes model to JSON file format, and serializes weights to MD5 file format
        First checks whether or not the Model Output directory exists
        :return:  Returns true if the model was successfully serialized, and false otherwise
        '''
        script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

        dir = script_dir + "/../Model_Output"
        if not os.path.isdir(dir):
            os.makedirs(dir)

        output_file = dir + "/" + self.name + "_Model.hdf5"

        self.model.save(output_file)

        print("Saved model to disk")


    def exploreByFile(self,fileName):
        '''
        Explores hyperparameter space and outputs diagnostic plots (given
        :param fileName: The name of results file
        :return: None
        '''
        ###TODO Complete creation of directory and writing out plots to the directory


        ###Load in scan file
        r = ta.Reporting(fileName)

        ###Extract absolute directory path of this script
        #script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

        ###Get output directory from user-specified argument
        #dataDir = script_dir + "/../" + arguments['<dataset-dir>']

        ###Create data directory if doesn't exist
        #try:
         #   os.mkdir(dataDir)
        #except:
         #   print("Directory already exists, not creating...")

        ###Plot correlation heatmap
        r.plot_corr()
        mp.savefig('Correlation.png')

        ###Plot histogram of metric accuracy
        r.plot_hist()
        mp.savefig('Histogram.png')

        ###Bar Plots multi-dimensional
        r.plot_bars('--epochs', 'val_acc', '--batch', '--dense')
        mp.savefig('Training.png')

        r.plot_bars('--epochs','val_acc','--convlayers','--conv')
        mp.savefig('Convolutions.png')

        r.plot_bars('--epochs', 'val_acc', '--layers', '--dense')
        mp.savefig('Fully_Connected.png')


    def explore(self,params):
        '''
        Explores hyperparameter space and outputs diagnostic plots to file
        :param params: A dictionary of hyperparameters to explore
        :return: The talos scan history
        '''
        x = self.reshape(self.data.getFullX())
        y = self.data.getFullY()
        h = ta.Scan(x, y, params=params,
                    model=exploration_model,
                    dataset_name=self.name,
                    experiment_no='1',
                    grid_downsample=.1)

        return h


    def loadModel(self):
        '''
        Loads serialized model from file to enable predictions on new data
        :return: True if the model was successfully loaded, and false otherwise
        '''


        ###Path to this script
        script_dir = os.path.dirname(__file__)

        ###Path to Model Output directory
        dir = script_dir + "/../Model_Output"

        ###If the path to this directory doesn't exist then the model was never trained
        if not os.path.isdir(dir):
            return False

        ###Open the HDF5 file
        modelFile = dir + "/" + self.name + "_Model.hdf5"



        ###Create the model based on the structure from the HDF5 File
        mdl = load_model(modelFile)

        ###Save the model to the class
        self.model = mdl

        ###If we reached the end of this function then return True
        return True




