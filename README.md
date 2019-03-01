# MNIST Digit Classification System
MNIST Digit Classification using Convolutional Neural Networks

This project presents a system for training, tuning hyperparameters, and evaluating a Convolutional Neural Network for MNIST digit classification.

# Best Model and Process to Reproduce

My best model had (accuracy, categorical cross-entropy losses) of 

Training: 
Validation:
Testing:

## Steps to Reproduce

First, clone the repository and set up a virtual environment. Then install all dependencies using the pip command listed under the dependencies heading. Next, ensure you are in the project directory ("MNIST_Prediction") and follow the commands below.

The exact commands to reproduce this result are:

    python MNIST download Data
    python MNIST train Data Optimal optimized_model
    python MNIST test Results Data Optimal

This will produce two files, one is "Model_Output/Optimal.txt." This contains the training and validation accuracy. 
The second file is "Results.txt" which contains the Testing accuracy. To inspect and alter the model architecture, please see the Models/optimized_model.txt file and the Model Specification markdown file for instructions.

# How did I arrive at this model?

TODO





# Full System for MNIST Classification

The MNIST handwritten dataset is a popular benchmark for image classification algorithms, especially CNN's. The purpose of this project is to have an enclosed system for beginners to experiment with CNN architectures on the MNIST dataset without doing any extra work. The project includes commands for downloading the data, parsing a CNN model from a text file and training the CNN, Hyperparameter exploration using Talos, and Evaluating proposed CNN's on the test set and serializing output models.

## Dependencies

The full list of dependencies for the project are laid out in the requirements.txt file. For use, please first create a python virtual enviornment on your machine, and then use: 

    pip install -r requirements.txt 

Ensure that you are the main directory of the project in order to access requirements.txt

# Downloading Data

The data is downloaded directly from this link: "http://yann.lecun.com/exdb/mnist/" To do so, please use the following command

    python MNIST download <dataset-dir>
  
The argument specifies the name of the directory to download the data to. Please keep this consistent throughout all commands of the project. The datasets will be downloaded as gzipped tarballs to save space on disk. They will be parsed when training/testing the model.

# Training a Model

In order to train, a model specified using our model file format. Please use the following command:

    python MNIST train <dataset-dir> <model-name> <model-description-file> [-s SPLIT]

Here, \<dataset-dir> should match the name used in the downloading step. \<model-name> will be the identifier used for the model for any output files. \<model-description-file> should be the filename of your specified model file (Please do not include any path information, just the file name).

This command will output a serialized Keras model file along with a simple text file with accuracy metrics on the train and validation sets. The -s argument can be used to alter the proportion of data sent to the training set (Default: 0.9). 

# Testing a Model

    python MNIST test <comparison-name> <dataset-dir> <model-names>

Again, \<dataset-dir> should be the same as the previous commands. \<comparison-name> is an identifier used to produce the results, and \<model-names> is a comma separated list of model names that had been trained using the train command. The output of this command is a single text file with results for all of the models on the testing set (includes categorical-cross entropy loss and prediction accuracy).

# Exploring Hyperparameters

    python MNIST explore <dataset-dir> <model-name> [-f FILE] [-z SIZE] [-c CONV] [-v CLAYERS] [-r RATE] [-b BATCH] [-l LAYERS] [-o OPTIMIZER] [-d DROPOUT] [-k KERNEL] [-p POOL] [-y DECAY] [-m MOMENTUM] [-e EPOCHS]

One more time, \<dataset-dir> should be the same as in any other command. \<model-name> is an identifer used to create an output directory for the results of the hyperparameter scan. The definitions of all of the optional parameters that can be tested are as follows:

    -r, --rate RATE                 Comma separated list of learning rates to explore [default: 0.001]
    -b, --batch BATCH               Comma separated list of batch sizes to explore [default: 1000]
    -l, --layers LAYERS             Comma separated list of number of fully connected layers to explore [default: 1,2]
    -c, --conv CONV                 Comma separated list of number of convolutional filters to explore[default: 16,32]
    -o, --opt OPTIMIZER             Comma separated list of optimizers to try [default: Adam,RMSProp]
    -d, --dropout DROPOUT           Comma separated list of dropout percentage [default: 0.1,0.25]
    -e, --epochs EPOCHS             Comma separated list of epoch sizes for training [default: 5,10,15]
    -z, --dense SIZE                Comma separated list of neurons in output of hidden layer [default: 32,64]
    -k, --kernel KERNEL             Comma separated list of kernel sizes for convolutional layers [default: 3,5]
    -p, --pool POOL                 Comma separated list of sizes for max pooling layer [default: 2,3]
    -y, --decay DECAY               Comma separated list of decay values for optimization [default: 0.0001]
    -m, --mom MOMENTUM              Comma separated list of momentum values for optimization [default: 0.25,0.5,0.75]
    -v, --convlayers CLAYERS        Comma separated list of number of convolutions before max pooling [default: 1,2]

For this exploration the architecture was fixed at Convolutional layers first, followed by Maxpooling and dropout, and then flattening. Finally fully connected layers and softmax output with 10 categories (one per digit).
 
## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

This is not an officially supported Google product.
