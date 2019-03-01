# MNIST Digit Classification System
MNIST Digit Classification using Convolutional Neural Networks

This project presents a system for training, tuning hyperparameters, and evaluating a Convolutional Neural Network for MNIST digit classification.


# Production System for MNIST Classification

The MNIST handwritten dataset is a popular benchmark for image classification algorithms, especially CNN's. The purpose of this project is to have an enclosed system for beginners to experiment with CNN architectures on the MNIST dataset without doing any extra work. The project includes commands for downloading the data, parsing a CNN model from a text file and training the CNN, Hyperparameter exploration using Talos, and Evaluating proposed CNN's on the test set and serializing output models.

## Dependencies

The full list of dependencies for the project are laid out in the requirements.txt file. For use, please first create a python virtual enviornment on your machine, and then use: 

pip install -r requirements.txt 

Ensure that you are the main directory of the project in order to access requirements.txt

# Downloading Data

The data is downloaded directly from this link: "http://yann.lecun.com/exdb/mnist/" To do so, please use the following command

python main.py download \<dataset-dir>
  
The argument specifies the name of the directory to download the data to. Please keep this consistent throughout all commands of the project. The datasets will be downloaded as gzipped tarballs to save space on disk. They will be parsed when training/testing the model.

# Training a Model

In order to train, a model specified using our model file format. Please use the following command:

python main.py train \<dataset-dir> \<model-name> \<model-description-file> [-s SPLIT]

Here, \<dataset-dir> should match the name used in the downloading step. \<model-name> will be the identifier used for the model for any output files. \<model-description-file> should be the filename of your specified model file (Please do not include any path information, just the file name).

This command will output a serialized Keras model file along with a simple text file with accuracy metrics on the train and validation sets. The -s argument can be used to alter the proportion of data sent to the training set (Default: 0.9). 

# Testing a Model

  python main.py test \<comparison-name> \<dataset-dir> \<model-names>

Again, \<dataset-dir> should be the same as the previous commands. \<comparison-name> is an identifier used to produce the results, and \<model-names> is a comma separated list of model names that had been trained using the train command. The output of this command is a single text file with results for all of the models on the testing set (includes categorical-cross entropy loss and prediction accuracy).

# Exploring Hyperparameters




 
## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

This is not an officially supported Google product.
