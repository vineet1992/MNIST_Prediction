###This class defines how to parse a user-specified model input file to create a model

from keras import Sequential
from keras import layers
from keras.optimizers import RMSprop
import math

class ModelParser:

    def __init__(self, modelFile,trainSet):
        '''Parameters:
         modelFile - Model specification file (see README for details)
         trainSet - The training set for this model (rows are samples, column
        '''

        self.modelFile = modelFile
        self.rows = int(math.sqrt(trainSet.shape[1]))

    def parse(self):
        '''This method parses the model file and returns a keras CNN model
        according to the specifications by the user'''
        ##Create sequential CNN model
        model = Sequential()

        ###Open model file for reading
        text_file = open(self.modelFile, "r")

        ##Search for beginning of model arguments
        begin = False

        ###True until we have created our first layer
        firstLayer = True

        ###Loop through the model specification file
        while (text_file.readable()):

            ###Read the next line
            line = text_file.readline()
            params = line.replace('\n','').split(",")
            ##If we have already read the "Layers:" line
            if (begin):

                ###Conv2D Layer
                if (line.startswith("Conv")):

                    ###If this is the first layer, include the input shape
                    if (firstLayer):
                        model.add(layers.Conv2D(
                            int(params[1]), kernel_size=(int(params[2]), int(params[3])),
                            activation=params[4], input_shape=(self.rows, self.rows, 1)))
                        firstLayer = False

                    ###Otherwise, just include the specified params
                    else:
                        model.add(layers.Conv2D(
                            int(params[1]), kernel_size=(int(params[2]), int(params[3])),
                            activation=params[4], ))


                ##Max pooling layer
                elif (line.startswith("MaxPool")):
                    model.add(layers.MaxPool2D(pool_size=(int(params[1]), int(params[2]))))

                ###Dropout layer
                elif (line.startswith("Dropout")):
                    model.add(layers.Dropout(float(params[1])))
                ###Flatten layer
                elif (line.startswith("Flatten")):
                    model.add(layers.Flatten())

                ###Dense layer
                elif (line.startswith("Dense")):
                    model.add(layers.Dense(int(params[1]), activation=params[2]))

                elif(line.startswith("Optimizer")):
                    break
                ###Unable to understand this layer
                else:
                    print("Can't understand the line " + line)
                    ##TODO Add invalid model file error message here

            elif (line.startswith("Layers")):
                begin = True
        ###get the optimizer parametetrs
        optParams = text_file.readline().split(",")
        optimizer = RMSprop(float(optParams[0]),float(optParams[1]),float(optParams[2]),float(optParams[3]))
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        epochs = int(text_file.readline())
        batchSize = int(text_file.readline())
        text_file.close()
        return [model,epochs,batchSize]

