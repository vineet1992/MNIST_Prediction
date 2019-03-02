###Takes as input a Dataset object (training) or a serialized model filename
### Methods for training and testing (which returns predictions given test data)

'''
This file defines a model for training and testing
Required input is a Dataset object and a name for the model as a String representation
This object contains methods for training a model, testing a model, loading and saving a model, and exploring hyperparameters of the model

'''

from keras.models import load_model
from ModelParser import ModelParser
from Exploration_Model import exploration_model
import math
import numpy as np
import os
import matplotlib.pyplot as mp
import talos as ta


mdlOutputDir = "Model_Output" ###Name of the output directory for saved models
hyprOutputDir = "Hyperparam_Search" ###Name of the output directory for hyperparameter explorations
script_dir = os.path.dirname(__file__)  ####Absolute dir the script is in


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

            ###Extract only training images
            self.trainX = data.getTrainX()

            ###Extract number of rows in the images
            self.rows = int(math.sqrt(self.trainX.shape[1]))

            ###Normalize the images
            if(normalize):
                self.trainX = self.trainX/255.0

            ###Extract one-hot encoded categorical labels
            self.trainY = data.getTrainY()

        ###If the partitioning is 1, this means we only have a training set
        if not data.partition==1:

            ###Extract the testing images
            self.testX = data.getTestX()
            ###Compute the number of rows in the testing images
            self.rows = int(math.sqrt(self.testX.shape[1]))
            if(normalize):
                self.testX = self.testX/255.0

            ###Extract the testing labels
            self.testY = data.getTestY()

        ###Set the model name
        self.name = modelName

        ###Set whether or not the pixels are normalized to 0-1
        self.normalize = normalize


    def train(self,modelFile):
        '''Parameters:
        modelFile - Model specification file (see README for correct syntax)
        Trains the model according to the specification file and outputs a trained keras model
        :return: Cross-entropy loss and Prediction accuracy on the training set after the last epoch
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

        ###Fit the model and collect the history object
        h = self.model.fit(self.trainX, self.trainY,
                                 batch_size=self.batchSize,epochs=self.epochs)

        ###Return the loss and the accuracy on the training set after the final epoch
        return (h.history["loss"][self.epochs-1],h.history["acc"][self.epochs-1])


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

        dir = os.path.join(os.path.dirname(script_dir),"Model_Output")
        if not os.path.isdir(dir):
            os.makedirs(dir)

        output_file = os.path.join(dir,self.name + "_Model.hdf5")

        self.model.save(output_file)

        print("Saved model to disk")


    def exploreByFile(self,fileName):
        '''
        Explores hyperparameter space and outputs diagnostic plots (given
        :param fileName: The name of results file
        :return: None
        '''

        dir = os.path.join(os.path.dirname(script_dir),hyprOutputDir)
        dir = os.path.join(dir,self.name)

        ###Load in scan file
        r = ta.Reporting(os.path.join(dir,fileName))



        ###Plot correlation heatmap
        r.plot_corr()
        mp.savefig(os.path.join(dir,'Correlation.png'))

        ###Plot histogram of metric accuracy
        r.plot_hist()
        mp.savefig(os.path.join(dir, 'Histogram.png'))

        ###Bar Plots multi-dimensional
        r.plot_bars('--epochs', 'val_acc', '--batch', '--dense')
        mp.savefig(os.path.join(dir,'Training.png'))

        r.plot_bars('--epochs','val_acc','--convlayers','--conv')
        mp.savefig(os.path.join(dir,'Convolutions.png'))

        r.plot_bars('--epochs', 'val_acc', '--layers', '--dense')
        mp.savefig(os.path.join(dir,'Fully_Connected.png'))


    def explore(self,params):
        '''
        Explores hyperparameter space and outputs diagnostic plots to file
        :param params: A dictionary of hyperparameters to explore
        :return: The talos scan history
        '''

        dir = os.path.join(os.path.dirname(script_dir),hyprOutputDir)
        if not os.path.isdir(dir):
            os.makedirs(dir)

        dir = os.path.join(dir,self.name)
        if not os.path.isdir(dir):
            os.makedirs(dir)


        name = os.path.join(dir,self.name)
        x = self.reshape(self.data.getFullX())
        y = self.data.getFullY()
        h = ta.Scan(x, y, params=params,
                    model=exploration_model,
                    dataset_name=name,
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
        dir = os.path.join(os.path.dirname(script_dir),"Model_Output")

        ###If the path to this directory doesn't exist then the model was never trained
        if not os.path.isdir(dir):
            print("Please ensure that the specified model file exists in the Model_Output directory")
            return False

        ###Open the HDF5 file
        modelFile = os.path.join(dir,self.name + "_Model.hdf5")



        ###Create the model based on the structure from the HDF5 File
        mdl = load_model(modelFile)

        ###Save the model to the class
        self.model = mdl

        ###If we reached the end of this function then return True
        return True




