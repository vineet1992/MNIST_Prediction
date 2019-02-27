###Takes as input a Dataset object (training) or a serialized model filename
### Methods for training and testing (which returns predictions given test data)


from keras.utils.np_utils import to_categorical #To one-hot encode the output labels
from keras import Sequential
from keras import layers
from keras import Model
from keras.models import model_from_json
from keras.models import load_model
from MNIST.ModelParser import ModelParser
import math
import numpy as np
import os
import matplotlib.pyplot as mp

class Model:

    def __init__(self, data, modelName,normalize=True):
        '''
        Default constructor for a model object
        :param data: Dataset object representing all of the data together
        :param modelName: The name for this model (used for output file generation)
        :param normalize: Should the data be normalized?
        '''

        if not data.partition==0:
            trainSet = data.get_train_set()

            self.trainX = trainSet[:,0:(trainSet.shape[1]-1)]
            self.rows = int(math.sqrt(self.trainX.shape[1]))
            if(normalize):
                self.trainX = self.trainX/255.0
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



    def explore(self,params):
        '''
        Explores hyperparameter space and outputs diagnostic plots to file
        :param params: A dictionary of hyperparameters to explore
        :return:
        '''


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




