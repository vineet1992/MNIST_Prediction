from keras.models import Sequential
from keras import layers




###Define a set of keys going to be used in the exploration model here

params_used = ["--dense","--batch","--conv","--decay","--dropout","--epochs","--kernel","--layers","--mom","--opt","--pool","--rate","--convlayers"]

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
    for i in range(0,params['--convlayers']):
        if i==0:
            model.add(layers.Conv2D(
                            params['--conv'], kernel_size=(params["--kernel"], params["--kernel"]),
                            activation='relu', input_shape=(trainX.shape[1], trainX.shape[1], 1)))
        else:
            model.add(layers.Conv2D(
            params['--conv'], kernel_size=(params["--kernel"], params["--kernel"]),
            activation='relu', input_shape=(trainX.shape[1], trainX.shape[1], 1)))

    ###Add max pooling layer
    model.add(layers.MaxPool2D((params['--pool'],params['--pool'])))

    ###Add dropout to max pooling
    model.add(layers.Dropout(params['--dropout']))

    ###Flatten Max pooling layer for input into dense layers
    model.add(layers.Flatten())

    ###Add fully connected layers
    for i in range(0,params['--layers']):
        model.add(layers.Dense(params['--dense'],
                    activation='relu'))

    ###Add softmax output layer to predict probabilities
    model.add(layers.Dense(10,activation="softmax"))

    ###Compile model
    model.compile(optimizer=params['--opt'](lr=lr_normalizer(params['--rate'], params['--opt']),decay=params['--decay'],momentum=params['--mom']),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    ###Fit the model to the training data and evaluate on the validation set
    out = model.fit(trainX, trainX,
                    batch_size=params['--batch'],
                    epochs=params['--epochs'],
                    verbose=0,
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
            params[key] = value
    return params