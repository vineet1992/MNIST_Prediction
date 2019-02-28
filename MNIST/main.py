
#########################################
#    Main File for MNIST Prediction     #
#                                       #
#########################################



"""MNIST - Complete package to train and test a Convolutional Neural Network on the MNIST dataset
Usage:
    MNIST.py download <dataset-dir>
    MNIST.py train <dataset-dir> <model-name> <model-description-file> [-s SPLIT]
    MNIST.py test <comparison-name> <dataset-dir> <model-names>
    MNIST.py explore <dataset-dir> <model-name> [-f FILE] [-z SIZE] [-c CONV] [-v CLAYERS] [-r RATE] [-b BATCH] [-l LAYERS] [-o OPTIMIZER] [-d DROPOUT] [-k KERNEL] [-p POOL] [-y DECAY] [-m MOMENTUM] [-e EPOCHS]
    MNIST.py (-h | --help)
Arguments:
    <dataset-dir>  Directory to look for dataset. This should be created by using download
    <model-name> Name of the model to be trained to create output directory
    <model-names> Comma separated list of model names to load for evaluation
    <model-description-file> File with model specification (see README for details)
    <comparison-name> Name of test comparison (for output file creation)
Options:
    -s, --split-percent SPLIT       Proportion of samples to send to train set [default: 0.9]
    -r, --rate RATE                 Comma separated list of learning rates to explore [default: 0.001]
    -b, --batch BATCH               Comma separated list of batch sizes to explore [default: 1000]
    -l, --layers LAYERS             Comma separated list of number of fully connected layers to explore [default: 1,2,3]
    -c, --conv CONV                 Comma separated list of number of convolutional filters to explore[default: 16,32]
    -o, --opt OPTIMIZER             Comma separated list of optimizers to try [default: Adam,RMSProp]
    -d, --dropout DROPOUT           Comma separated list of dropout percentage [default: 0.1,0.25,0.5]
    -e, --epochs EPOCHS             Comma separated list of epoch sizes for training [default: 5,10,15]
    -z, --dense SIZE                Comma separated list of neurons in output of hidden layer [default: 128,256]
    -k, --kernel KERNEL             Comma separated list of kernel sizes for convolutional layers [default: 3,5]
    -p, --pool POOL                 Comma separated list of sizes for max pooling layer [default: 2,3]
    -y, --decay DECAY               Comma separated list of decay values for optimization [default: 0.0001]
    -m, --mom MOMENTUM              Comma separated list of momentum values for optimization [default: 0.25,0.5,0.75]
    -v, --convlayers CLAYERS        Comma separated list of number of convolutions before max pooling [default: 1,2,3]
    -f, --file FILE                 Filename pointing to a file with scan results from Talos
    -h, --help                      Show this screen.
"""


from docopt import docopt
from Data.download import load_data
from Dataset import Dataset
from Model import Model
from Exploration_Model import convertParams
import os

def main():
    arguments = docopt(__doc__)

    ###Download dataset to the directory specified
    if(arguments['download']):

        ###Extract absolute directory path of this script
        script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

        ###Get output directory from user-specified argument
        dataDir = script_dir + "/../" + arguments['<dataset-dir>']

        ###Create data directory if doesn't exist
        try:
            os.mkdir(dataDir)
        except:
            print("Directory already exists, not creating...")

        ###Remote directory location of train and test files
        remoteDir = "http://yann.lecun.com/exdb/mnist/"
        urls = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz",
                "t10k-labels-idx1-ubyte.gz"]
        for i in range(0, len(urls)):
            urls[i] = remoteDir + "/" + urls[i]


        ###Local directory location of each of the files
        filenames = ["Train.gz", "Train_Labels.gz", "Test.gz", "Test_Labels.gz"]
        for i in range(0, len(filenames)):
            filenames[i] = dataDir + "/" + filenames[i]

        ###Run load data function from download
        load_data(urls,filenames)


    elif(arguments['train']):
        ###Create dataset object using directory and partition amount in arguments
        data = Dataset(arguments['<dataset-dir>'] + "/Train.gz",arguments['<dataset-dir>'] + "/Train_Labels.gz",partition = float(arguments['--split-percent'][0]))

        ###Create model object from training data
        mdl = Model(data,arguments['<model-name>'])

        ###Train the model based upon the
        mdl.train(arguments['<model-description-file>'])

        ###Apply the model to the dev set
        print(mdl.test())

        ###Save the model to a serialized Keras file
        mdl.saveModel()

    elif(arguments['test']):

        ###Load the testing dataset
        data = Dataset(arguments['<dataset-dir>'] + "/Test.gz",arguments['<dataset-dir>'] + "/Test_Labels.gz",0)

        ###TODO Incorporate ability to load multiple models and write results to a file

        ###Create the model object from the specified
        mdl = Model(data,arguments['<model-names>'])

        ###Load model from serialized model file
        mdl.loadModel()

        ###Get Output statistics on the testing set
        print(mdl.test())

    elif(arguments['explore']):

        print(arguments)


        ###Create dataset object using directory and partition amount in arguments
        data = Dataset(arguments['<dataset-dir>'] + "/Train.gz", arguments['<dataset-dir>'] + "/Train_Labels.gz",
                       partition=float(arguments['--split-percent'][0]))

        ###Create model object from training data
        mdl = Model(data, arguments['<model-name>'])

        ###Convert the full arguments list to only those relevant to parameters
        hyperparams = convertParams(arguments)

        ###Explore hyperparameter space
        if len(arguments['--file']) > 0:
            mdl.exploreByFile(arguments['--file'][0])
        else:
            result = mdl.explore(hyperparams)

        ###Output relevant plots to file TODO


if __name__ == '__main__':
    main()




