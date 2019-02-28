from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam
from keras.optimizers import RMSprop



###Define a set of keys going to be used in the exploration model here

params_used = ["--dense","--batch","--conv","--decay","--dropout","--epochs","--kernel","--layers","--pool","--rate","--convlayers"]

def exploration_model(trainX, trainY, testX, testY, params):
    '''
    Defines a CNN model to classify the MNIST dataset with hyperparameters left out for tuning using Talos.
    Params should be loaded from the dataset object
    :param trainX: The training images
    :param trainY: The training labels
    :param testX: The testing images
    :param testY: The testing labels
    :param params: Mapping from arguments to possible values
    :return: A parametrized model that can be passed to Talos
    '''

    ###Initialize CNN model
    model = Sequential()

    ###Add convolutional layer
    for i in range(0,int(params['--convlayers'])):
        if i==0:
            model.add(layers.Conv2D(
                            int(params['--conv']), kernel_size=(int(params["--kernel"]), int(params["--kernel"])),
                            activation='relu', input_shape=(trainX.shape[1], trainX.shape[1], 1)))
        else:
            model.add(layers.Conv2D(
            int(params['--conv']), kernel_size=(int(params["--kernel"]), int(params["--kernel"])),
            activation='relu'))

    ###Add max pooling layer
    model.add(layers.MaxPool2D((int(params['--pool']),int(params['--pool']))))

    ###Add dropout to max pooling
    model.add(layers.Dropout(float(params['--dropout'])))

    ###Flatten Max pooling layer for input into dense layers
    model.add(layers.Flatten())

    ###Add fully connected layers
    for i in range(0,int(params['--layers'])):
        model.add(layers.Dense(int(params['--dense']),
                    activation='relu'))
    ###Include dropout optimziation
    model.add(layers.Dropout(float(params['--dropout'])))

    ###Add softmax output layer to predict probabilities
    model.add(layers.Dense(10,activation="softmax"))

    ##TODO Use optimizer type as a hyperparameter

    ###Compile model
    model.compile(optimizer=Adam(lr=float(params['--rate']),
                                            decay=float(params['--decay'])),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    print(model.summary())
    ###Fit the model to the training data and evaluate on the validation set
    out = model.fit(trainX, trainY,
                    batch_size=int(params['--batch']),
                    epochs=int(params['--epochs']),
                    verbose=1,
                    validation_data=[testX, testY])
    return out,model

def convertParams(arguments):
    '''

    :param arguments: Full list of arguments from docopt
    :return: Only the arguments necessary for the exploration model


    '''
    params = {}

    ###Identify all keys that match the parameters of interest in the model
    for key,value in arguments.items():
        if key in params_used:
            params[key] = value[0].rsplit(",")
    return params


def temp_test(trainX, trainY, testX, testY, params):
    '''
    Defines a CNN model to classify the MNIST dataset with hyperparameters left out for tuning using Talos.
    Params should be loaded from the dataset object
    :param trainX: The training images
    :param trainY: The training labels
    :param testX: The testing images
    :param testY: The testing labels
    :param params: Mapping from arguments to possible values
    :return: A parametrized model that can be passed to Talos
    '''

    ###Initialize CNN model
    model = Sequential()
    model.add(layers.Conv2D(32, (5,5),
                            activation='relu', input_shape=(trainX.shape[1], trainX.shape[1], 1)))
    model.add(layers.Conv2D(32, kernel_size=(5,5),
            activation='relu' ))

    ###Add max pooling layer
    model.add(layers.MaxPool2D((2,2)))

    ###Add dropout to max pooling
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(64, kernel_size=(5, 5),
                            activation='relu'))
    model.add(layers.Conv2D(64, kernel_size=(5,5),
            activation='relu' ))
    model.add(layers.MaxPool2D((2,2)))


    ###Flatten Max pooling layer for input into dense layers
    model.add(layers.Flatten())

    ###Add fully connected layers
    model.add(layers.Dense(256,
                    activation='relu'))
    ###Include dropout optimziation
    model.add(layers.Dropout(0.5))

    ###Add softmax output layer to predict probabilities
    model.add(layers.Dense(10,activation="softmax"))

    ##TODO Use optimizer type as a hyperparameter

    ###Compile model
    model.compile(optimizer=RMSprop(0.0001,0.5,1e-8,0.0),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    print(model.summary())
    ###Fit the model to the training data and evaluate on the validation set
    out = model.fit(trainX, trainY,
                    batch_size=100,
                    epochs=1,
                    verbose=1,
                    validation_data=[testX, testY])
    return out,model
